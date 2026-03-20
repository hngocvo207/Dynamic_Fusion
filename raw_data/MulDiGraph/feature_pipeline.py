"""
Expert-Engineered Feature Processing Pipeline
==============================================
Handles: segregation → imputation → scaling → PyTorch tensor output.

Dataset: /home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output1.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

# Identifier columns kept as metadata (not used as input features)
METADATA_COLS: list[str] = ["node", "sample_idx", "n_nodes", "n_edges"]

# Binary classification target
LABEL_COL: str = "is_phisher"

# All remaining columns become the numerical feature matrix X
# (derived automatically from the CSV so the pipeline is schema-agnostic)


class PhisherFeaturePipeline:
    """
    End-to-end preprocessing pipeline for the blockchain phisher dataset.

    Workflow
    --------
    Training split:
        pipeline = PhisherFeaturePipeline()
        X_train, y_train, meta_train = pipeline.fit_transform(df_train)

    Validation / Test split  (uses statistics fitted on train set only):
        X_val,   y_val,   meta_val   = pipeline.transform(df_val)
        X_test,  y_test,  meta_test  = pipeline.transform(df_test)

    Parameters
    ----------
    impute_strategy : str
        Strategy for SimpleImputer.  'median' is robust to outliers and
        preferred for financial / graph features where extreme values are
        common.  Alternatives: 'mean', 'most_frequent', 'constant'.

    Notes on scaler choice (RobustScaler)
    --------------------------------------
    Graph-derived financial features (degree counts, transaction amounts,
    centrality scores) often exhibit:
      • Heavy tails / extreme outliers (e.g., out-degree of hub nodes can
        exceed 10 000 while median is ~5).
      • Zero-inflated distributions (many nodes have no outgoing transfers).

    RobustScaler centres on the median and scales by the IQR, making it
    resilient to both extremes.  StandardScaler would be distorted by
    outliers; MinMaxScaler would compress the majority of values into a
    tiny band near 0.
    """

    def __init__(self, impute_strategy: str = "median") -> None:
        self.impute_strategy = impute_strategy

        # Fitted artefacts (populated during fit_transform)
        self._feature_cols: list[str] | None = None
        self._imputer: SimpleImputer | None = None
        self._scaler: RobustScaler | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Fit the imputer & scaler on *df* then transform it.

        Use this method **only on the training split** to avoid data leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame loaded from the CSV (or a training-split slice).

        Returns
        -------
        X : torch.FloatTensor  – shape (N, F)
            Scaled numerical feature matrix.
        y : torch.LongTensor   – shape (N,)
            Binary labels  [0 = legitimate, 1 = phisher].
        meta : pd.DataFrame
            Metadata columns (node, sample_idx, n_nodes, n_edges).
        """
        df = self._validate_and_copy(df)
        meta, y_raw, X_raw = self._segregate(df)

        # Derive feature column list from the first call
        self._feature_cols = X_raw.columns.tolist()

        # --- Step 1: Imputation -----------------------------------------------
        # Missing values can arise from nodes with zero transactions (division
        # by zero in ratio features) or from computation failures.
        # We fit the imputer only once on training data.
        self._imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imputed = self._imputer.fit_transform(X_raw)

        # --- Step 2: Scaling (RobustScaler) -----------------------------------
        # See class-level docstring for rationale.  Fitted only on training set.
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X_imputed)

        self._is_fitted = True

        # --- Step 3: Convert to PyTorch tensors --------------------------------
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)          # (N, F)
        y_tensor = self._encode_label(y_raw)                             # (N,)

        return X_tensor, y_tensor, meta

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Apply already-fitted imputer & scaler to a new split.

        Use this method on **validation and test splits**.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame for validation or test.

        Returns
        -------
        X : torch.FloatTensor  – shape (N, F)
        y : torch.LongTensor   – shape (N,)
        meta : pd.DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline has not been fitted yet. "
                "Call fit_transform() on the training split first."
            )

        df = self._validate_and_copy(df)
        meta, y_raw, X_raw = self._segregate(df)

        # Enforce the exact same feature column order as training
        X_raw = X_raw.reindex(columns=self._feature_cols)

        # --- Impute with training-derived statistics --------------------------
        X_imputed = self._imputer.transform(X_raw)

        # --- Scale with training-derived median/IQR --------------------------
        X_scaled = self._scaler.transform(X_imputed)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = self._encode_label(y_raw)

        return X_tensor, y_tensor, meta

    # ------------------------------------------------------------------
    # Properties (read-only access to fitted state)
    # ------------------------------------------------------------------

    @property
    def feature_cols(self) -> list[str]:
        """List of feature column names (in order) after fitting."""
        if self._feature_cols is None:
            raise RuntimeError("Pipeline not fitted yet.")
        return self._feature_cols

    @property
    def n_features(self) -> int:
        """Dimensionality of the feature vector."""
        return len(self.feature_cols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_copy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic schema validation + defensive copy so the caller's DataFrame
        is never mutated.
        """
        df = df.copy()
        missing = [c for c in METADATA_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Required columns missing from DataFrame: {missing}"
            )
        return df

    @staticmethod
    def _segregate(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Split the raw DataFrame into three disjoint parts.

        Returns
        -------
        meta   : pd.DataFrame  – identifier / graph-size metadata
        y_raw  : pd.Series     – raw label column (True/False strings or bool)
        X_raw  : pd.DataFrame  – numerical feature columns
        """
        meta = df[METADATA_COLS].reset_index(drop=True)
        y_raw = df[LABEL_COL].reset_index(drop=True)

        # Everything that is neither metadata nor label → feature matrix
        feature_cols = [
            c for c in df.columns
            if c not in METADATA_COLS and c != LABEL_COL
        ]
        X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

        return meta, y_raw, X_raw

    @staticmethod
    def _encode_label(y_raw: pd.Series) -> torch.Tensor:
        """
        Convert the raw 'is_phisher' column to a binary LongTensor.

        Handles multiple source formats:
          • Python bool  (True / False)
          • String 'True' / 'False'  (common after CSV round-trip)
          • Integer 1 / 0
        """
        # Coerce to string first for uniformity, then map
        y_str = y_raw.astype(str).str.strip().str.lower()
        y_int = y_str.map({"true": 1, "false": 0, "1": 1, "0": 0})

        if y_int.isna().any():
            bad = y_raw[y_int.isna()].unique()
            raise ValueError(
                f"Unrecognised label values (cannot map to 0/1): {bad}"
            )

        return torch.tensor(y_int.values.astype(int), dtype=torch.long)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load the raw feature CSV from disk."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script: python feature_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split

    DATA_PATH = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output1.csv"

    print("=" * 60)
    print("PhisherFeaturePipeline – smoke test")
    print("=" * 60)

    # 1. Load raw data
    df = load_csv(DATA_PATH)
    print(f"\n[1] Raw data shape       : {df.shape}")
    print(f"    Columns              : {list(df.columns)}")

    # 2. Train / val / test split (stratified on label)
    #    — convert label to int first so stratify works
    label_int = df[LABEL_COL].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    df_train_val, df_test = train_test_split(
        df, test_size=0.1, stratify=label_int, random_state=42
    )
    label_trainval = label_int.loc[df_train_val.index]
    df_train, df_val = train_test_split(
        df_train_val, test_size=1/9, stratify=label_trainval, random_state=42
    )  # ~0.15 of total

    print(f"\n[2] Split sizes          : train={len(df_train)}, "
          f"val={len(df_val)}, test={len(df_test)}")

    # 3. Fit on train
    pipeline = PhisherFeaturePipeline(impute_strategy="median")
    X_train, y_train, meta_train = pipeline.fit_transform(df_train)

    print(f"\n[3] Feature dimensionality: {pipeline.n_features}")
    print(f"    Feature columns (first 5): {pipeline.feature_cols[:5]} …")

    print(f"\n[4] Train tensors:")
    print(f"    X_train  : {X_train.shape}  dtype={X_train.dtype}")
    print(f"    y_train  : {y_train.shape}  dtype={y_train.dtype}")
    print(f"    label dist: {dict(zip(*y_train.unique(return_counts=True)))}")

    # 4. Transform val / test (NO re-fitting)
    X_val,  y_val,  meta_val  = pipeline.transform(df_val)
    X_test, y_test, meta_test = pipeline.transform(df_test)

    print(f"\n[5] Val  tensors : X={X_val.shape}, y={y_val.shape}")
    print(f"    Test tensors : X={X_test.shape}, y={y_test.shape}")

    # 5. Sanity checks
    assert X_train.dtype == torch.float32, "X must be FloatTensor"
    assert y_train.dtype == torch.long,    "y must be LongTensor"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        "Feature dim mismatch between splits"
    assert not torch.isnan(X_train).any(), "NaN detected in X_train after scaling"

    # 6. Show metadata sample
    print(f"\n[6] Metadata sample (train, first 3 rows):")
    print(meta_train.head(3).to_string(index=False))

    print("\n✅  All assertions passed – pipeline is working correctly.")
    # Ví dụ thêm vào cuối block if __name__ == "__main__":
    torch.save({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }, "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/processed_data.pt")
    print("✅ Đã lưu tensors vào processed_data.pt")
    meta_train.to_csv("/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/meta_train.csv", index=False)
    sys.exit(0)
