"""
Day 1 - Preprocess Pipeline
Dataset: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_DIR = "data/processed"
RANDOM_SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"[1/6] Loading data from {path} ...")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    # Strip whitespace khỏi column names (CIC dataset hay bị lỗi này)
    df.columns = df.columns.str.strip()
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns[:5])} ... (85 total)")
    return df


def inspect_labels(df: pd.DataFrame) -> None:
    print("\n[2/6] Label distribution:")
    label_col = " Label" if " Label" in df.columns else "Label"
    counts = df[label_col].value_counts()
    for label, count in counts.items():
        print(f"      {label:<30} {count:>8,}  ({count/len(df)*100:.1f}%)")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/6] Cleaning ...")
    label_col = " Label" if " Label" in df.columns else "Label"

    # Bỏ các cột metadata (tên cột có thể có hoặc không có space prefix)
    drop_cols = [
        "Flow ID", " Flow ID",
        "Source IP", " Source IP",
        "Destination IP", " Destination IP",
        "Timestamp", " Timestamp",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"      Dropped {len(drop_cols)} metadata columns")

    # Drop bất kỳ cột string nào còn sót (tránh lỗi float conversion)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # Giữ lại label_col nếu chưa encode
    label_col_check = " Label" if " Label" in df.columns else "Label"
    non_numeric_to_drop = [c for c in non_numeric if c != label_col_check]
    if non_numeric_to_drop:
        print(f"      Dropping non-numeric cols: {non_numeric_to_drop}")
        df = df.drop(columns=non_numeric_to_drop)

    # Replace inf với NaN rồi drop
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"      Dropped {before - len(df):,} rows with NaN/Inf")

    # Encode label: BENIGN = 0, DDoS = 1
    df[label_col] = df[label_col].str.strip()
    df["label"] = (df[label_col] != "BENIGN").astype(int)
    df = df.drop(columns=[label_col])
    print(f"      Labels encoded: BENIGN=0, DDoS=1")
    print(f"      Final shape: {df.shape}")
    return df


def balance_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Balancing classes (RandomUnderSampler) ...")
    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"      Before - BENIGN: {(y==0).sum():,} | DDoS: {(y==1).sum():,}")

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X, y)

    print(f"      After  - BENIGN: {(y_res==0).sum():,} | DDoS: {(y_res==1).sum():,}")

    df_balanced = pd.DataFrame(X_res, columns=X.columns)
    df_balanced["label"] = y_res.values
    return df_balanced


def split_and_scale(df: pd.DataFrame):
    print("\n[5/6] Train/Val/Test split + StandardScaler ...")
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    # Scale dựa trên train set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"      Train : {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names):
    print(f"\n[6/6] Saving to {OUTPUT_DIR}/ ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy",   X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)

    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{OUTPUT_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("      Saved: X_train, X_val, X_test, y_train, y_val, y_test")
    print("      Saved: scaler.pkl, feature_names.pkl")
    print("\n✅ Preprocessing complete!")
    print(f"   Input dim  : {X_train.shape[1]} features")
    print(f"   Train size : {len(X_train):,}")
    print(f"   Val size   : {len(X_val):,}")
    print(f"   Test size  : {len(X_test):,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data(DATA_PATH)
    inspect_labels(df)
    df = clean_data(df)
    df = balance_data(df)

    feature_names = list(df.drop(columns=["label"]).columns)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)


if __name__ == "__main__":
    main()