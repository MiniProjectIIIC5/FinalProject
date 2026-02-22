"""
train_model.py — Multi-Dataset Fake Profile Classifier
=======================================================
Downloads TWO Kaggle datasets, normalises them into a single
unified schema, engineers 20 features, trains a Random Forest
with proper class balance, and saves:
    ml/model.pkl        — trained classifier
    ml/scaler.pkl       — fitted StandardScaler
    ml/features.json    — feature names & metadata

Run:
    python ml/train_model.py
    python ml/train_model.py --skip-download   (if datasets already cached)
    python ml/train_model.py --data-dir ./data (custom data folder)

Requirements:
    pip install kagglehub pandas scikit-learn joblib numpy
"""

import os
import sys
import json
import math
import logging
import argparse
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import kagglehub

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Output paths (relative to this script's directory) ──────────────────────
SCRIPT_DIR   = Path(__file__).parent
MODEL_PATH   = SCRIPT_DIR / "model.pkl"
SCALER_PATH  = SCRIPT_DIR / "scaler.pkl"
FEATURES_PATH = SCRIPT_DIR / "features.json"


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — whoseaspects/genuinefake-user-profile-dataset
#  Twitter-style data with screen_name, followers_count, verified, etc.
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_twitter(skip_download=False, data_dir=None):
    log.info("── Dataset 1: whoseaspects/genuinefake-user-profile-dataset ──")
    if data_dir:
        path = Path(data_dir) / "twitter"
    else:
        path = Path(kagglehub.dataset_download(
            "whoseaspects/genuinefake-user-profile-dataset"
        ))
    log.info(f"  Path: {path}")

    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}")

    frames = []
    for f in csv_files:
        log.info(f"  Reading {f.name} …")
        try:
            df = pd.read_csv(f, encoding="latin1", on_bad_lines="skip", low_memory=False)
            frames.append(df)
        except Exception as e:
            log.warning(f"  Could not read {f.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    log.info(f"  Raw shape: {df.shape}")
    log.info(f"  Columns: {list(df.columns)}")

    # ── Determine label ──
    if "dataset" in df.columns:
        df["label"] = df["dataset"].apply(
            lambda x: 1 if str(x).strip().lower() in ("fake", "1", "bot") else 0
        )
    elif "fake" in df.columns:
        df["label"] = df["fake"].astype(int)
    elif "label" in df.columns:
        df["label"] = df["label"].astype(int)
    else:
        raise ValueError("Dataset 1: cannot determine label column")

    log.info(f"  Label dist: {dict(df['label'].value_counts())}")

    # ── Username column ──
    for col in ("screen_name", "username", "user_name", "name"):
        if col in df.columns:
            df["_username"] = df[col].fillna("").astype(str)
            break
    else:
        raise ValueError("Dataset 1: no username column found")

    # ── Optional numeric columns ──
    def safe_col(df, col, default=0):
        return df[col].fillna(default).astype(float) if col in df.columns else pd.Series(default, index=df.index)

    df["_followers"]   = safe_col(df, "followers_count")
    df["_following"]   = safe_col(df, "friends_count")
    df["_posts"]       = safe_col(df, "statuses_count")
    df["_favourites"]  = safe_col(df, "favourites_count")
    df["_listed"]      = safe_col(df, "listed_count")
    df["_verified"]    = safe_col(df, "verified")
    df["_protected"]   = safe_col(df, "protected")
    df["_has_pic"]     = (
        df["profile_image_url_https"].notnull().astype(float)
        if "profile_image_url_https" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    df["_bio_len"]     = (
        df["description"].fillna("").apply(len)
        if "description" in df.columns
        else pd.Series(0, index=df.index)
    )

    result = df[["_username","_followers","_following","_posts","_favourites",
                 "_listed","_verified","_protected","_has_pic","_bio_len","label"]].copy()
    result = result[result["_username"].str.len() > 0].dropna(subset=["label"])
    log.info(f"  Clean shape: {result.shape}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — rajumavinmar/fake-instagram-profile-dataset
#  Instagram data: profile pic, posts, followers, following, bio, label
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_instagram(skip_download=False, data_dir=None):
    log.info("── Dataset 2: rajumavinmar/fake-instagram-profile-dataset ──")
    if data_dir:
        path = Path(data_dir) / "instagram"
    else:
        path = Path(kagglehub.dataset_download(
            "rajumavinmar/fake-instagram-profile-dataset"
        ))
    log.info(f"  Path: {path}")

    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}")

    frames = []
    for f in csv_files:
        log.info(f"  Reading {f.name} …")
        try:
            df = pd.read_csv(f, encoding="latin1", on_bad_lines="skip", low_memory=False)
            frames.append(df)
        except Exception as e:
            log.warning(f"  Could not read {f.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    log.info(f"  Raw shape: {df.shape}")
    log.info(f"  Columns: {list(df.columns)}")

    # ── Determine label ──
    label_candidates = ["fake", "label", "is_fake", "class", "target", "fake_account"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        # Last column heuristic
        last = df.columns[-1]
        if df[last].nunique() <= 2:
            label_col = last
            log.warning(f"  Using last column '{last}' as label")
        else:
            raise ValueError("Dataset 2: cannot determine label column")

    df["label"] = df[label_col].astype(str).apply(
        lambda x: 1 if x.strip().lower() in ("1", "fake", "true", "yes") else 0
    )
    log.info(f"  Label dist: {dict(df['label'].value_counts())}")

    # ── Username column ──
    for col in ("username", "screen_name", "user_name", "profile_username", "userid"):
        if col in df.columns:
            df["_username"] = df[col].fillna("").astype(str)
            break
    else:
        df["_username"] = ""   # Instagram dataset may not have usernames

    def safe_col(df, col, default=0):
        return df[col].fillna(default).astype(float) if col in df.columns else pd.Series(default, index=df.index)

    # Instagram column name variants
    df["_followers"] = safe_col(df, next((c for c in ("follower_count","followers","followers_count","#followers") if c in df.columns), "__"), 0)
    df["_following"] = safe_col(df, next((c for c in ("following_count","following","friends_count","#follows")    if c in df.columns), "__"), 0)
    df["_posts"]     = safe_col(df, next((c for c in ("post_count","posts","num_posts","statuses_count","#posts")  if c in df.columns), "__"), 0)
    df["_favourites"] = pd.Series(0, index=df.index)
    df["_listed"]     = pd.Series(0, index=df.index)
    df["_verified"]   = safe_col(df, next((c for c in ("is_verified","verified") if c in df.columns), "__"), 0)
    df["_protected"]  = pd.Series(0, index=df.index)
    df["_has_pic"]    = safe_col(df, next((c for c in ("profile_pic","has_profile_pic","profile_picture","pic") if c in df.columns), "__"), 0)
    df["_bio_len"]    = (
        df[next((c for c in ("biography","bio","description") if c in df.columns), "__")].fillna("").apply(len)
        if any(c in df.columns for c in ("biography","bio","description"))
        else pd.Series(0, index=df.index)
    )

    result = df[["_username","_followers","_following","_posts","_favourites",
                 "_listed","_verified","_protected","_has_pic","_bio_len","label"]].copy()
    result = result.dropna(subset=["label"])
    log.info(f"  Clean shape: {result.shape}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (20 features — all derivable from a profile URL)
# ══════════════════════════════════════════════════════════════════════════════
SUSPICIOUS_KW = {
    "bot","fake","scam","spam","hack","temp","xyz","anon",
    "xxx","adult","promo","giveaway","f4f","l4l","onlyfan",
    "follow4follow","like4like","viral","offi","official2",
}
VOWELS = set("aeiouAEIOU")


def _entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s.lower():
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v/n) * math.log2(v/n) for v in freq.values())


def extract_features(row) -> dict:
    """Return dict of 20 numeric features for one row."""
    u = str(row["_username"]).strip()
    n = len(u)
    lower = u.lower()

    digits   = sum(c.isdigit() for c in u)
    specials = sum(c in "._-" for c in u)
    alpha    = sum(c.isalpha() for c in u)
    vowels_n = sum(c in VOWELS for c in u)
    d_ratio  = digits / (n + 1e-5)

    f = {}

    # ── Username signals ────────────────────────────────────────────────────
    f["username_len"]         = min(n, 40)
    f["digit_count"]          = digits
    f["digit_ratio"]          = round(d_ratio, 4)
    f["special_count"]        = specials
    f["alpha_count"]          = alpha
    f["vowel_ratio"]          = round(vowels_n / (n + 1e-5), 4)
    f["entropy"]              = round(_entropy(u), 4)
    f["all_digits"]           = int(u.isdigit())
    f["no_alpha"]             = int(alpha == 0)
    f["no_vowels"]            = int(n >= 6 and vowels_n == 0)
    f["trailing_digits4"]     = int(bool(re.search(r"\d{4,}$", u)))
    f["digit_block"]          = int(bool(re.search(r"[A-Za-z]{2,}\d{4,}", u)))
    f["repeating_chars"]      = int(bool(re.search(r"(.)\1{3,}", u)))
    f["suspicious_keyword"]   = int(any(kw in lower for kw in SUSPICIOUS_KW))
    f["uuid_like"]            = int(bool(re.match(r"^[a-f0-9]{8,}$", lower)))

    # ── Profile / social signals ────────────────────────────────────────────
    followers = float(row.get("_followers", 0) or 0)
    following = float(row.get("_following", 0) or 0)
    posts     = float(row.get("_posts", 0)     or 0)

    f["has_profile_pic"]      = float(row.get("_has_pic",   0) or 0)
    f["verified"]             = float(row.get("_verified",  0) or 0)
    f["bio_length"]           = min(float(row.get("_bio_len", 0) or 0), 500)

    # follower/following ratio capped at 50
    ff_ratio = followers / (following + 1)
    f["follower_following_ratio"] = round(min(ff_ratio, 50), 4)

    # engagement proxy: posts per follower (capped)
    f["posts_per_follower"]   = round(min(posts / (followers + 1), 50), 4)

    return f


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE & PREPARE
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(df: pd.DataFrame):
    log.info("Extracting features …")
    rows = [extract_features(row) for _, row in df.iterrows()]
    X = pd.DataFrame(rows)
    y = df["label"].astype(int).values

    # Sanity-check: both classes must be present
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Only class(es) {classes} found — dataset has no balance. "
            "Check your label column mapping."
        )

    log.info(f"Feature matrix: {X.shape}")
    log.info(f"Class distribution — Real(0): {(y==0).sum()}  Fake(1): {(y==1).sum()}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(X: pd.DataFrame, y: np.ndarray):
    log.info("Splitting dataset …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    log.info("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)

    # ── Evaluate ──
    y_pred  = rf.predict(X_te_s)
    y_proba = rf.predict_proba(X_te_s)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    cm  = confusion_matrix(y_te, y_pred)

    log.info(f"\n{'='*55}")
    log.info(f"  Test Accuracy : {acc:.4f}")
    log.info(f"  ROC-AUC       : {auc:.4f}")
    log.info(f"  Confusion Matrix:\n{cm}")
    log.info("\n" + classification_report(
        y_te, y_pred,
        target_names=["Real(0)", "Fake(1)"],
        zero_division=0
    ))
    log.info(f"{'='*55}\n")

    # ── Cross-validation ──
    log.info("Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all_s = scaler.transform(X)
    cv_scores = cross_val_score(rf, X_all_s, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Feature importance ──
    importances = dict(zip(X.columns, rf.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
    log.info("  Top-8 features:")
    for name, imp in top:
        log.info(f"    {name:<35} {imp:.4f}")

    return rf, scaler, {
        "accuracy": round(acc, 4),
        "roc_auc":  round(auc, 4),
        "cv_mean":  round(float(cv_scores.mean()), 4),
        "cv_std":   round(float(cv_scores.std()),  4),
        "n_train":  int(len(X_tr)),
        "n_test":   int(len(X_te)),
        "class_dist": {"real": int((y==0).sum()), "fake": int((y==1).sum())},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(model, scaler, feature_names, meta):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    payload = {
        "features":    feature_names,
        "n_features":  len(feature_names),
        "model_type":  "RandomForestClassifier",
        "metrics":     meta,
    }
    with open(FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    log.info(f"✅  model.pkl  → {MODEL_PATH}")
    log.info(f"✅  scaler.pkl → {SCALER_PATH}")
    log.info(f"✅  features.json → {FEATURES_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Train fake-profile classifier on two Kaggle datasets.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Use locally-cached kagglehub downloads")
    ap.add_argument("--data-dir", default=None,
                    help="Override dataset path (must contain twitter/ and instagram/ sub-dirs)")
    ap.add_argument("--dataset", choices=["both","twitter","instagram"], default="both",
                    help="Which dataset(s) to use for training (default: both)")
    args = ap.parse_args()

    dfs = []

    if args.dataset in ("both", "twitter"):
        try:
            dfs.append(load_dataset_twitter(skip_download=args.skip_download, data_dir=args.data_dir))
        except Exception as e:
            log.error(f"Failed to load Twitter dataset: {e}")
            if args.dataset == "twitter":
                sys.exit(1)

    if args.dataset in ("both", "instagram"):
        try:
            dfs.append(load_dataset_instagram(skip_download=args.skip_download, data_dir=args.data_dir))
        except Exception as e:
            log.error(f"Failed to load Instagram dataset: {e}")
            if args.dataset == "instagram":
                sys.exit(1)

    if not dfs:
        log.error("No datasets loaded. Exiting.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"\nCombined dataset: {combined.shape[0]:,} rows")

    X, y = build_feature_matrix(combined)
    model, scaler, meta = train(X, y)
    save_artifacts(model, scaler, list(X.columns), meta)

    log.info("\n🎉  Training complete!")
    log.info("   The project will now use this model for predictions.")
    log.info("   Re-run `python ml/train_model.py` any time to retrain.\n")


if __name__ == "__main__":
    main()