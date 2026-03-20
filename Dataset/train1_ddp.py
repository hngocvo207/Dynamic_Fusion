"""
DDP-enabled training script for ETH_GBert.
Launch via: torchrun --nproc_per_node=<NUM_GPUS> train1_ddp.py [args]
Or use the provided run_train1_ddp.sh tmux helper script.
"""
import argparse
import gc
import json
import os

os.environ["WANDB_START_METHOD"] = "thread"
import pickle as pkl
import pickle
import random
import time
import datetime
import warnings

import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import wandb

from env_config import env_config
from ETH_GBert import ETH_GBertModel
from utils import (
    InputExample,
    CorpusDataset,
    normalize_adj,
    sparse_scipy2torch,
    get_class_count_and_weight,
    WeightedRandomSampler,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize the distributed process group and return local_rank, rank, world_size."""
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def load_data(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def adjust_matrix_size(adj_matrix, target_size):
    current_size = adj_matrix.shape[0]
    if current_size == target_size:
        return adj_matrix
    if current_size > target_size:
        adj_matrix = adj_matrix[:target_size, :target_size]
    else:
        padding = np.zeros((target_size - current_size, target_size - current_size))
        adj_matrix = np.block([
            [adj_matrix, np.zeros((current_size, target_size - current_size))],
            [np.zeros((target_size - current_size, current_size)), padding],
        ])
    return adj_matrix


# ---------------------------------------------------------------------------
# Data-loading utilities (DDP-aware)
# ---------------------------------------------------------------------------

def get_pytorch_dataloader(
    examples,
    tokenizer,
    address_to_index,
    max_seq_length,
    gcn_embedding_dim,
    batch_size,
    shuffle_choice,
    rank,
    world_size,
    classes_weight=None,
    total_resample_size=-1,
):
    """Return a DataLoader that uses DistributedSampler when world_size > 1."""
    ds = CorpusDataset(examples, tokenizer, address_to_index, max_seq_length, gcn_embedding_dim)

    use_distributed = world_size > 1

    if shuffle_choice == 0:  # no shuffle
        if use_distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            sampler = None
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 1:  # shuffle
        if use_distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            sampler = None
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 2:  # weighted resample (falls back to non-distributed)
        assert classes_weight is not None
        assert total_resample_size > 0
        weights = [
            classes_weight[0] if label == 0
            else classes_weight[1] if label == 1
            else classes_weight[2]
            for _, _, _, _, label in ds
        ]
        sampler = WeightedRandomSampler(weights, num_samples=total_resample_size, replacement=True)
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=ds.pad,
        )


# ---------------------------------------------------------------------------
# Predict / Evaluate
# ---------------------------------------------------------------------------

def _unwrap_model(model):
    """Return the underlying model, stripping the DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


def predict(model, examples, tokenizer, address_to_index, max_seq_length,
            gcn_embedding_dim, batch_size, device, gcn_adj_list,
            cfg_loss_criterion, do_softmax_before_mse, rank, world_size):
    dataloader = get_pytorch_dataloader(
        examples, tokenizer, address_to_index, max_seq_length, gcn_embedding_dim,
        batch_size, shuffle_choice=0, rank=rank, world_size=world_size,
    )
    predict_out = []
    confidence_out = []
    raw_model = _unwrap_model(model)
    raw_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, _, label_ids, gcn_swop_eye = batch
            score_out = raw_model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
            if cfg_loss_criterion == "mse" and do_softmax_before_mse:
                score_out = F.softmax(score_out, dim=-1)
            predict_out.extend(score_out.max(1)[1].tolist())
            confidence_out.extend(score_out.max(1)[0].tolist())
    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(-1)


def evaluate(model, gcn_adj_list, predict_dataloader, batch_size, epoch,
             dataset_name, device, num_classes, cfg_loss_criterion,
             do_softmax_before_mse, loss_weight, perform_metrics_str, rank):
    # Use the unwrapped model for evaluation to avoid DDP reducer issues
    raw_model = _unwrap_model(model)
    raw_model.eval()
    predict_out = []
    all_label_ids = []
    ev_loss = 0.0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch
            logits = raw_model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)

            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            eval_accuracy = predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

    f1_metrics = f1_score(np.array(all_label_ids).reshape(-1),
                          np.array(predict_out).reshape(-1), average="weighted")
    pre = precision_score(np.array(all_label_ids).reshape(-1),
                          np.array(predict_out).reshape(-1), average="weighted")
    rec = recall_score(np.array(all_label_ids).reshape(-1),
                       np.array(predict_out).reshape(-1), average="weighted")

    if is_main_process(rank):
        print("Report:\n" + classification_report(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1), digits=4,
        ))

    ev_acc = correct / total if total > 0 else 0
    end = time.time()

    if is_main_process(rank):
        print(
            "Epoch : %d, %s: %.3f, Pre: %.3f, Rec: %.3f, Acc : %.3f on %s, Spend:%.3f minutes for evaluation"
            % (epoch, " ".join(perform_metrics_str), 100 * f1_metrics,
               100 * pre, 100 * rec, 100.0 * ev_acc, dataset_name,
               (end - start) / 60.0)
        )
        print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_metrics, pre, rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- DDP setup ----
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # ---- Seeds ----
    random.seed(env_config.GLOBAL_SEED)
    np.random.seed(env_config.GLOBAL_SEED)
    torch.manual_seed(env_config.GLOBAL_SEED)
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)

    # ---- Args ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="Dataset")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--sw", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="ETH_GBert")
    parser.add_argument("--validate_program", action="store_true")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience: number of epochs without validation F1 improvement before stopping")
    args = parser.parse_args()

    # Only rank-0 initializes wandb
    if is_main_process(rank):
        wandb.login(key="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")
        wandb.init(project="Dynamic_Feature_Training_for_b4e", config=args)

    cfg_model_type = args.model
    cfg_stop_words = True if args.sw == 1 else False
    will_train_mode_from_checkpoint = True if args.load == 1 else False
    gcn_embedding_dim = args.dim
    learning_rate0 = args.lr
    l2_decay = args.l2
    total_train_epochs = 100
    dropout_rate = 0.2

    if args.ds == "Dataset":
        batch_size = 8
        learning_rate0 = 8e-6
        l2_decay = 0.001

    MAX_SEQ_LENGTH = 200 + gcn_embedding_dim
    gradient_accumulation_steps = 1
    bert_model_scale = "bert-base-uncased"

    if env_config.TRANSFORMERS_OFFLINE == 1:
        bert_model_scale = os.path.join(
            env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
            f"hf-maintainers_{bert_model_scale}",
        )

    do_lower_case = True
    warmup_proportion = 0.1
    data_dir = "data/preprocessed/b4e_processed_data_1"
    output_dir = "./output/"
    if is_main_process(rank) and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    perform_metrics_str = ["weighted avg", "f1-score"]
    classifier_act_func = nn.ReLU()
    resample_train_set = False
    do_softmax_before_mse = True
    cfg_loss_criterion = "cle"
    model_file_4save = (
        f"{cfg_model_type}{gcn_embedding_dim}_model_{args.ds}_{cfg_loss_criterion}"
        f"_sw{int(cfg_stop_words)}.pt"
    )

    if args.validate_program:
        total_train_epochs = 1

    if is_main_process(rank):
        print(cfg_model_type + " Start at:", time.asctime())
        print(
            "\n----- Configure -----",
            f"\n  DDP world_size: {world_size}",
            f"\n  args.ds: {args.ds}",
            f"\n  stop_words: {cfg_stop_words}",
            f"\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> {str(gcn_embedding_dim)}",
            f"\n  Learning_rate0: {learning_rate0}",
            f"\n  weight_decay: {l2_decay}",
            f"\n  Loss_criterion {cfg_loss_criterion}",
            f"\n  softmax_before_mse: {do_softmax_before_mse}",
            f"\n  Dropout: {dropout_rate}",
            f"\n  gcn_act_func: Relu",
            f"\n  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}",
            f"\n  perform_metrics_str: {perform_metrics_str}",
            f"\n  model_file_4save: {model_file_4save}",
            f"\n  validate_program: {args.validate_program}",
        )

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------
    if is_main_process(rank):
        print("\n----- Prepare data set -----")
        print(f"  Load/shuffle/separate {args.ds} dataset, and vocabulary graph adjacent matrix")

    objects = []
    names = [
        "labels", "train_y", "train_y_prob", "valid_y", "valid_y_prob",
        "test_y", "test_y_prob", "shuffled_clean_docs", "address_to_index",
    ]
    for name in names:
        datafile = f"/home/ngochv/Dynamic_Feature/{data_dir}/data_{args.ds}.{name}"
        with open(datafile, "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))

    (lables_list, train_y, train_y_prob, valid_y, valid_y_prob,
     test_y, test_y_prob, shuffled_clean_docs, address_to_index) = tuple(objects)

    label2idx = lables_list[0]
    idx2label = lables_list[1]

    y = np.hstack((train_y, valid_y, test_y))
    y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

    examples = []
    for i, ts in enumerate(shuffled_clean_docs):
        ex = InputExample(i, ts.strip(), confidence=y_prob[i], label=y[i])
        examples.append(ex)

    num_classes = len(label2idx)
    gcn_vocab_size = len(address_to_index)
    train_size = len(train_y)
    valid_size = len(valid_y)
    test_size = len(test_y)

    indexs = np.arange(0, len(examples))
    train_examples = [examples[i] for i in indexs[:train_size]]
    valid_examples = [examples[i] for i in indexs[train_size:train_size + valid_size]]
    test_examples = [examples[i] for i in indexs[train_size + valid_size:train_size + valid_size + test_size]]

    # GCN adjacency
    weighted_adj_matrix = load_data(
        "/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/weighted_adjacency_matrix.pkl"
    )
    adjusted_adj_matrix = adjust_matrix_size(weighted_adj_matrix, gcn_vocab_size)
    gcn_vocab_adj = csr_matrix(adjusted_adj_matrix)
    gcn_adj_list = [normalize_adj(gcn_vocab_adj).tocoo()]
    gcn_adj_list = [sparse_scipy2torch(adj).to(device) for adj in gcn_adj_list]

    gc.collect()

    train_classes_num, train_classes_weight = get_class_count_and_weight(train_y, len(label2idx))
    loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

    tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

    # ds size=1 for validating the program
    if args.validate_program:
        train_examples = [train_examples[0]]
        valid_examples = [valid_examples[0]]
        test_examples = [test_examples[0]]

    dl_kwargs = dict(
        tokenizer=tokenizer,
        address_to_index=address_to_index,
        max_seq_length=MAX_SEQ_LENGTH,
        gcn_embedding_dim=gcn_embedding_dim,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )

    train_dataloader = get_pytorch_dataloader(train_examples, shuffle_choice=0, **dl_kwargs)
    valid_dataloader = get_pytorch_dataloader(valid_examples, shuffle_choice=0, **dl_kwargs)
    test_dataloader = get_pytorch_dataloader(test_examples, shuffle_choice=0, **dl_kwargs)

    total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * total_train_epochs)

    if is_main_process(rank):
        print("  Train_classes count:", train_classes_num)
        print(f"  Num examples for train = {len(train_examples)}"
              f", after weight sample: {len(train_dataloader) * batch_size}")
        print("  Num examples for validate = %d" % len(valid_examples))
        print("  Batch size = %d" % batch_size)
        print("  Num steps = %d" % total_train_steps)

    # ------------------------------------------------------------------
    # Build / Load model
    # ------------------------------------------------------------------
    if is_main_process(rank):
        print("\n----- Running training -----")

    if will_train_mode_from_checkpoint and os.path.exists(os.path.join(output_dir, model_file_4save)):
        checkpoint = torch.load(os.path.join(output_dir, model_file_4save), map_location="cpu")
        if "step" in checkpoint:
            prev_save_step = checkpoint["step"]
            start_epoch = checkpoint["epoch"]
        else:
            prev_save_step = -1
            start_epoch = checkpoint["epoch"] + 1
        valid_acc_prev = checkpoint["valid_acc"]
        perform_metrics_prev = checkpoint["perform_metrics"]
        model = ETH_GBertModel.from_pretrained(
            bert_model_scale,
            state_dict=checkpoint["model_state"],
            gcn_adj_dim=gcn_vocab_size,
            gcn_adj_num=len(gcn_adj_list),
            gcn_embedding_dim=gcn_embedding_dim,
            num_labels=len(label2idx),
        )
        pretrained_dict = checkpoint["model_state"]
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        if is_main_process(rank):
            print(
                f"Loaded the pretrain model: {model_file_4save}",
                f", epoch: {checkpoint['epoch']}",
                f"step: {prev_save_step}",
                f"valid acc: {checkpoint['valid_acc']}",
                f"{' '.join(perform_metrics_str)}_valid: {checkpoint['perform_metrics']}",
            )
    else:
        start_epoch = 0
        valid_acc_prev = 0
        perform_metrics_prev = 0
        model = ETH_GBertModel.from_pretrained(
            bert_model_scale,
            gcn_adj_dim=gcn_vocab_size,
            gcn_adj_num=len(gcn_adj_list),
            gcn_embedding_dim=gcn_embedding_dim,
            num_labels=len(label2idx),
        )
        prev_save_step = -1

    model.to(device)

    # ---- Wrap model with DDP ----
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = BertAdam(
        model.parameters(),
        lr=learning_rate0,
        warmup=warmup_proportion,
        t_total=total_train_steps,
        weight_decay=l2_decay,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_start = time.time()
    global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

    all_loss_list = {"train": [], "valid": [], "test": []}
    all_f1_list = {"train": [], "valid": [], "test": []}

    valid_f1_best_epoch = start_epoch
    test_f1_when_valid_best = 0
    test_recall_when_valid_best = 0
    test_precision_when_valid_best = 0

    # Early stopping state
    early_stop_counter = 0
    early_stop_flag = torch.zeros(1, dtype=torch.int32, device=device)

    for epoch in range(start_epoch, total_train_epochs):
        # Set epoch on ALL distributed samplers so data ordering differs each epoch
        for dl in (train_dataloader, valid_dataloader, test_dataloader):
            if hasattr(dl, "sampler") and isinstance(dl.sampler, DistributedSampler):
                dl.sampler.set_epoch(epoch)

        tr_loss = 0
        ep_train_start = time.time()
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            if prev_save_step > -1:
                if step <= prev_save_step:
                    continue
            if prev_save_step > -1:
                prev_save_step = -1

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, y_prob_batch, label_ids, gcn_swop_eye = batch

            logits = model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)

            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob_batch)
            else:
                if loss_weight is None:
                    loss = F.cross_entropy(logits, label_ids)
                else:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, loss_weight)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

            if step % 40 == 0 and is_main_process(rank):
                print(
                    "Epoch:{}-{}/{}, Train {} Loss: {}, Cumulated time: {}m ".format(
                        epoch, step, len(train_dataloader),
                        cfg_loss_criterion, loss.item(),
                        (time.time() - train_start) / 60.0,
                    )
                )

        if is_main_process(rank):
            print("--------------------------------------------------------------")

        # ---- Evaluation (all ranks compute, only rank 0 prints / saves) ----
        valid_loss, valid_acc, perform_metrics, valid_recall, valid_precision = evaluate(
            model, gcn_adj_list, valid_dataloader, batch_size, epoch, "Valid_set",
            device, num_classes, cfg_loss_criterion, do_softmax_before_mse,
            loss_weight, perform_metrics_str, rank,
        )
        test_loss, test_acc, test_f1, test_recall, test_precision = evaluate(
            model, gcn_adj_list, test_dataloader, batch_size, epoch, "Test_set",
            device, num_classes, cfg_loss_criterion, do_softmax_before_mse,
            loss_weight, perform_metrics_str, rank,
        )

        all_loss_list["train"].append(tr_loss)
        all_loss_list["valid"].append(valid_loss)
        all_loss_list["test"].append(test_loss)
        all_f1_list["valid"].append(perform_metrics)
        all_f1_list["test"].append(test_f1)

        # Log metrics to WandB (rank 0 only)
        if is_main_process(rank):
            wandb.log({
                "epoch": epoch,
                "train_loss": tr_loss,
                "valid_loss": valid_loss,
                "valid_f1": perform_metrics,
                "valid_recall": valid_recall,
                "valid_precision": valid_precision,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_recall": test_recall,
                "test_precision": test_precision,
            })

            print(
                "Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(
                    epoch, tr_loss, valid_loss, (time.time() - train_start) / 60.0,
                )
            )

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Save checkpoint (rank 0 only) ----
        if is_main_process(rank):
            if perform_metrics > perform_metrics_prev:
                # Access underlying model (unwrap DDP)
                model_state = model.module.state_dict()
                to_save = {
                    "epoch": epoch,
                    "model_state": model_state,
                    "valid_acc": valid_acc,
                    "lower_case": do_lower_case,
                    "perform_metrics": perform_metrics,
                    "recall": valid_recall,
                    "precision": valid_precision,
                }
                torch.save(to_save, os.path.join(output_dir, model_file_4save))
                perform_metrics_prev = perform_metrics
                test_f1_when_valid_best = test_f1
                test_recall_when_valid_best = test_recall
                test_precision_when_valid_best = test_precision
                valid_f1_best_epoch = epoch
                early_stop_counter = 0
                # Save best validation results to a JSON file
                best_results = {
                    "best_epoch": epoch,
                    "valid_f1": round(float(perform_metrics), 6),
                    "valid_recall": round(float(valid_recall), 6),
                    "valid_precision": round(float(valid_precision), 6),
                    "valid_acc": round(float(valid_acc), 6),
                    "test_f1_at_best_valid": round(float(test_f1), 6),
                    "test_recall_at_best_valid": round(float(test_recall), 6),
                    "test_precision_at_best_valid": round(float(test_precision), 6),
                    "test_acc_at_best_valid": round(float(test_acc), 6),
                    "model_file": model_file_4save,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                results_file = os.path.join(
                    output_dir,
                    model_file_4save.replace(".pt", "_best_results.json"),
                )
                with open(results_file, "w") as rf:
                    json.dump(best_results, rf, indent=4)
                print(
                    f"New best model saved at epoch {epoch} with "
                    f"F1: {perform_metrics:.4f}, Recall: {valid_recall:.4f}, "
                    f"Precision: {valid_precision:.4f}"
                )
                print(f"Best results saved to: {results_file}")
            else:
                early_stop_counter += 1
                print(
                    f"No validation F1 improvement for {early_stop_counter}/{args.patience} epoch(s)."
                )
                if early_stop_counter >= args.patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {args.patience} consecutive epochs)."
                    )
                    early_stop_flag.fill_(1)

        # Broadcast early-stop decision from rank 0 to all other ranks
        if world_size > 1:
            dist.broadcast(early_stop_flag, src=0)

        # Synchronise all processes before next epoch (only needed for multi-GPU)
        if world_size > 1:
            dist.barrier()

        # All ranks exit the training loop if early stopping was triggered
        if early_stop_flag.item() == 1:
            break

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    if is_main_process(rank):
        print(
            "\n**Optimization Finished!, Total spend:",
            (time.time() - train_start) / 60.0,
        )
        print(
            "**Valid weighted F1: %.3f at %d epoch."
            % (100 * perform_metrics_prev, valid_f1_best_epoch)
        )
        print(
            "**Test weighted F1 when valid best: %.3f, Recall: %.3f, Precision: %.3f"
            % (100 * test_f1_when_valid_best, 100 * test_recall_when_valid_best,
               100 * test_precision_when_valid_best)
        )
        # Save final training summary to JSON
        final_summary = {
            "best_valid_f1": round(float(perform_metrics_prev), 6),
            "best_valid_epoch": valid_f1_best_epoch,
            "test_f1_at_best_valid": round(float(test_f1_when_valid_best), 6),
            "test_recall_at_best_valid": round(float(test_recall_when_valid_best), 6),
            "test_precision_at_best_valid": round(float(test_precision_when_valid_best), 6),
            "total_train_time_min": round((time.time() - train_start) / 60.0, 2),
            "model_file": model_file_4save,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_file = os.path.join(
            output_dir,
            model_file_4save.replace(".pt", "_training_summary.json"),
        )
        with open(summary_file, "w") as sf:
            json.dump(final_summary, sf, indent=4)
        print(f"Training summary saved to: {summary_file}")
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
