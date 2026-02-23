"""
train_model.py — Multi-Dataset Fake Profile Classifier
=======================================================
Downloads FOUR Kaggle datasets, normalises them into a single
unified schema, engineers 20 features, trains a Random Forest
with proper class balance, and saves:
    ml/model.pkl        — trained classifier
    ml/scaler.pkl       — fitted StandardScaler
    ml/features.json    — feature names & metadata

Datasets:
  1. whoseaspects/genuinefake-user-profile-dataset     (Twitter-style)
  2. rajumavinmar/fake-instagram-profile-dataset       (Instagram)
  3. vibodhbhosure/twitter-fake-profile-dataset        (Twitter v2)
  4. bitandatom/social-network-fake-account-dataset    (Social network)

Run:
    python ml/train_model.py
    python ml/train_model.py --skip-download                (if datasets already cached)
    python ml/train_model.py --data-dir ./data              (custom data folder)
    python ml/train_model.py --dataset twitter              (single dataset)
    python ml/train_model.py --dataset twitter,instagram    (subset of datasets)

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

from sklearn.ensemble import RandomForestClassifier
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
SCRIPT_DIR    = Path(__file__).parent
MODEL_PATH    = SCRIPT_DIR / "model.pkl"
SCALER_PATH   = SCRIPT_DIR / "scaler.pkl"
FEATURES_PATH = SCRIPT_DIR / "features.json"


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_col(df, col, default=0):
    """Return df[col] filled with default, or a constant Series if col absent."""
    return df[col].fillna(default).astype(float) if col in df.columns else pd.Series(default, index=df.index)


def _first_col(df, candidates, default=0):
    """Return the first matching column from candidates, else a constant Series."""
    col = next((c for c in candidates if c in df.columns), None)
    return _safe_col(df, col, default) if col else pd.Series(default, index=df.index)


def _read_csvs(path: Path, label: str) -> pd.DataFrame:
    """Read all CSVs under path, concatenate, and lowercase column names."""
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
    log.info(f"  [{label}] Raw shape: {df.shape}")
    log.info(f"  [{label}] Columns:   {list(df.columns)}")
    return df


def _resolve_label(df: pd.DataFrame, label: str, candidates=None) -> pd.Series:
    """
    Try common label column names; fall back to last column if binary.
    Returns an integer Series (0 = real, 1 = fake).
    """
    candidates = candidates or ["dataset", "fake", "label", "is_fake", "class",
                                 "target", "fake_account", "account_type"]
    for col in candidates:
        if col in df.columns:
            return df[col].astype(str).apply(
                lambda x: 1 if x.strip().lower() in ("fake", "1", "bot", "true", "yes") else 0
            )
    # last column heuristic
    last = df.columns[-1]
    if df[last].nunique() <= 2:
        log.warning(f"  [{label}] Using last column '{last}' as label")
        return df[last].astype(str).apply(
            lambda x: 1 if x.strip().lower() in ("fake", "1", "bot", "true", "yes") else 0
        )
    raise ValueError(f"[{label}] Cannot determine label column")


def _resolve_username(df: pd.DataFrame, label: str) -> pd.Series:
    for col in ("screen_name", "username", "user_name", "name", "profile_username", "userid"):
        if col in df.columns:
            return df[col].fillna("").astype(str)
    log.warning(f"  [{label}] No username column found; using empty strings")
    return pd.Series("", index=df.index)


def _to_unified(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Project a dataset-specific DataFrame to the unified 11-column schema."""
    return df[["_username", "_followers", "_following", "_posts", "_favourites",
               "_listed", "_verified", "_protected", "_has_pic", "_bio_len", "label"]].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — whoseaspects/genuinefake-user-profile-dataset  (Twitter)
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_twitter(skip_download=False, data_dir=None):
    LABEL = "DS1-Twitter"
    log.info(f"── {LABEL}: whoseaspects/genuinefake-user-profile-dataset ──")
    path = (Path(data_dir) / "twitter") if data_dir else Path(
        kagglehub.dataset_download("whoseaspects/genuinefake-user-profile-dataset")
    )
    log.info(f"  Path: {path}")

    df = _read_csvs(path, LABEL)
    df["label"]      = _resolve_label(df, LABEL)
    df["_username"]  = _resolve_username(df, LABEL)
    df["_followers"] = _safe_col(df, "followers_count")
    df["_following"] = _safe_col(df, "friends_count")
    df["_posts"]     = _safe_col(df, "statuses_count")
    df["_favourites"]= _safe_col(df, "favourites_count")
    df["_listed"]    = _safe_col(df, "listed_count")
    df["_verified"]  = _safe_col(df, "verified")
    df["_protected"] = _safe_col(df, "protected")
    df["_has_pic"]   = (
        df["profile_image_url_https"].notnull().astype(float)
        if "profile_image_url_https" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    df["_bio_len"]   = (
        df["description"].fillna("").apply(len)
        if "description" in df.columns
        else pd.Series(0, index=df.index)
    )

    result = _to_unified(df, LABEL)
    result = result[result["_username"].str.len() > 0].dropna(subset=["label"])
    log.info(f"  [{LABEL}] Clean shape: {result.shape}  |  label dist: {dict(result['label'].value_counts())}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — rajumavinmar/fake-instagram-profile-dataset  (Instagram)
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_instagram(skip_download=False, data_dir=None):
    LABEL = "DS2-Instagram"
    log.info(f"── {LABEL}: rajumavinmar/fake-instagram-profile-dataset ──")
    path = (Path(data_dir) / "instagram") if data_dir else Path(
        kagglehub.dataset_download("rajumavinmar/fake-instagram-profile-dataset")
    )
    log.info(f"  Path: {path}")

    df = _read_csvs(path, LABEL)
    df["label"]      = _resolve_label(df, LABEL)
    df["_username"]  = _resolve_username(df, LABEL)
    df["_followers"] = _first_col(df, ("follower_count", "followers", "followers_count", "#followers"))
    df["_following"] = _first_col(df, ("following_count", "following", "friends_count", "#follows"))
    df["_posts"]     = _first_col(df, ("post_count", "posts", "num_posts", "statuses_count", "#posts"))
    df["_favourites"]= pd.Series(0, index=df.index)
    df["_listed"]    = pd.Series(0, index=df.index)
    df["_verified"]  = _first_col(df, ("is_verified", "verified"))
    df["_protected"] = pd.Series(0, index=df.index)
    df["_has_pic"]   = _first_col(df, ("profile_pic", "has_profile_pic", "profile_picture", "pic"))
    bio_col          = next((c for c in ("biography", "bio", "description") if c in df.columns), None)
    df["_bio_len"]   = df[bio_col].fillna("").apply(len) if bio_col else pd.Series(0, index=df.index)

    result = _to_unified(df, LABEL).dropna(subset=["label"])
    log.info(f"  [{LABEL}] Clean shape: {result.shape}  |  label dist: {dict(result['label'].value_counts())}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — vibodhbhosure/twitter-fake-profile-dataset  (Twitter v2)
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_twitter_v2(skip_download=False, data_dir=None):
    LABEL = "DS3-TwitterV2"
    log.info(f"── {LABEL}: vibodhbhosure/twitter-fake-profile-dataset ──")
    path = (Path(data_dir) / "twitter_v2") if data_dir else Path(
        kagglehub.dataset_download("vibodhbhosure/twitter-fake-profile-dataset")
    )
    log.info(f"  Path: {path}")

    df = _read_csvs(path, LABEL)
    df["label"]      = _resolve_label(df, LABEL)
    df["_username"]  = _resolve_username(df, LABEL)

    # This dataset may use slightly different column names — cover common variants
    df["_followers"] = _first_col(df, ("followers_count", "followers", "follower_count", "num_followers"))
    df["_following"] = _first_col(df, ("friends_count",   "following", "following_count", "num_following"))
    df["_posts"]     = _first_col(df, ("statuses_count",  "tweets",    "tweet_count",     "num_tweets", "posts"))
    df["_favourites"]= _first_col(df, ("favourites_count","favorites_count","likes"))
    df["_listed"]    = _first_col(df, ("listed_count",    "listed"))
    df["_verified"]  = _first_col(df, ("verified",        "is_verified"))
    df["_protected"] = _first_col(df, ("protected",       "is_protected"))
    df["_has_pic"]   = (
        df["profile_image_url_https"].notnull().astype(float)
        if "profile_image_url_https" in df.columns
        else _first_col(df, ("has_profile_pic", "profile_pic", "default_profile_image"),
                        default=1)
    )
    bio_col = next((c for c in ("description", "biography", "bio") if c in df.columns), None)
    df["_bio_len"]   = df[bio_col].fillna("").apply(len) if bio_col else pd.Series(0, index=df.index)

    result = _to_unified(df, LABEL)
    result = result[result["_username"].str.len() > 0].dropna(subset=["label"])
    log.info(f"  [{LABEL}] Clean shape: {result.shape}  |  label dist: {dict(result['label'].value_counts())}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 4 — bitandatom/social-network-fake-account-dataset  (Social)
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_social(skip_download=False, data_dir=None):
    LABEL = "DS4-Social"
    log.info(f"── {LABEL}: bitandatom/social-network-fake-account-dataset ──")
    path = (Path(data_dir) / "social") if data_dir else Path(
        kagglehub.dataset_download("bitandatom/social-network-fake-account-dataset")
    )
    log.info(f"  Path: {path}")

    df = _read_csvs(path, LABEL)
    df["label"]      = _resolve_label(df, LABEL)
    df["_username"]  = _resolve_username(df, LABEL)

    df["_followers"] = _first_col(df, ("followers_count", "followers", "follower_count",
                                       "num_followers", "#followers"))
    df["_following"] = _first_col(df, ("friends_count",   "following", "following_count",
                                       "num_following",   "#follows"))
    df["_posts"]     = _first_col(df, ("statuses_count",  "posts",     "post_count",
                                       "num_posts",       "#posts",    "tweets", "tweet_count"))
    df["_favourites"]= _first_col(df, ("favourites_count","favorites_count","likes","like_count"))
    df["_listed"]    = _first_col(df, ("listed_count",    "listed"))
    df["_verified"]  = _first_col(df, ("verified",        "is_verified"))
    df["_protected"] = _first_col(df, ("protected",       "is_protected"))
    df["_has_pic"]   = _first_col(df, ("profile_pic",     "has_profile_pic",
                                       "profile_picture", "pic", "profile_image_url_https"))
    # For profile_image_url_https treat non-null as 1
    if "profile_image_url_https" in df.columns and "_has_pic" not in df.columns:
        df["_has_pic"] = df["profile_image_url_https"].notnull().astype(float)
    bio_col = next((c for c in ("description", "biography", "bio") if c in df.columns), None)
    df["_bio_len"]   = df[bio_col].fillna("").apply(len) if bio_col else pd.Series(0, index=df.index)

    result = _to_unified(df, LABEL).dropna(subset=["label"])
    log.info(f"  [{LABEL}] Clean shape: {result.shape}  |  label dist: {dict(result['label'].value_counts())}")
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
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def extract_features(row) -> dict:
    """Return dict of 20 numeric features for one row."""
    u     = str(row["_username"]).strip()
    n     = len(u)
    lower = u.lower()

    digits   = sum(c.isdigit() for c in u)
    specials = sum(c in "._-" for c in u)
    alpha    = sum(c.isalpha() for c in u)
    vowels_n = sum(c in VOWELS for c in u)
    d_ratio  = digits / (n + 1e-5)

    f = {}

    # ── Username signals ──────────────────────────────────────────────────
    f["username_len"]       = min(n, 40)
    f["digit_count"]        = digits
    f["digit_ratio"]        = round(d_ratio, 4)
    f["special_count"]      = specials
    f["alpha_count"]        = alpha
    f["vowel_ratio"]        = round(vowels_n / (n + 1e-5), 4)
    f["entropy"]            = round(_entropy(u), 4)
    f["all_digits"]         = int(u.isdigit())
    f["no_alpha"]           = int(alpha == 0)
    f["no_vowels"]          = int(n >= 6 and vowels_n == 0)
    f["trailing_digits4"]   = int(bool(re.search(r"\d{4,}$", u)))
    f["digit_block"]        = int(bool(re.search(r"[A-Za-z]{2,}\d{4,}", u)))
    f["repeating_chars"]    = int(bool(re.search(r"(.)\1{3,}", u)))
    f["suspicious_keyword"] = int(any(kw in lower for kw in SUSPICIOUS_KW))
    f["uuid_like"]          = int(bool(re.match(r"^[a-f0-9]{8,}$", lower)))

    # ── Profile / social signals ──────────────────────────────────────────
    followers = float(row.get("_followers", 0) or 0)
    following = float(row.get("_following", 0) or 0)
    posts     = float(row.get("_posts", 0)     or 0)

    f["has_profile_pic"]          = float(row.get("_has_pic",  0) or 0)
    f["verified"]                 = float(row.get("_verified", 0) or 0)
    f["bio_length"]               = min(float(row.get("_bio_len", 0) or 0), 500)
    f["follower_following_ratio"] = round(min(followers / (following + 1), 50), 4)
    f["posts_per_follower"]       = round(min(posts / (followers + 1), 50), 4)

    return f


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE & PREPARE
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(df: pd.DataFrame):
    log.info("Extracting features …")
    rows = [extract_features(row) for _, row in df.iterrows()]
    X    = pd.DataFrame(rows)
    y    = df["label"].astype(int).values

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Only class(es) {classes} found — dataset has no balance. "
            "Check your label column mapping."
        )

    log.info(f"Feature matrix: {X.shape}")
    log.info(f"Class distribution — Real(0): {(y==0).sum():,}  Fake(1): {(y==1).sum():,}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(X: pd.DataFrame, y: np.ndarray):
    log.info("Splitting dataset …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

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

    # ── Evaluate ──────────────────────────────────────────────────────────
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

    # ── Cross-validation ──────────────────────────────────────────────────
    log.info("Running 5-fold cross-validation …")
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all_s  = scaler.transform(X)
    cv_scores = cross_val_score(rf, X_all_s, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Feature importance ────────────────────────────────────────────────
    importances = dict(zip(X.columns, rf.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
    log.info("  Top-8 features:")
    for name, imp in top:
        log.info(f"    {name:<35} {imp:.4f}")

    return rf, scaler, {
        "accuracy":   round(acc, 4),
        "roc_auc":    round(auc, 4),
        "cv_mean":    round(float(cv_scores.mean()), 4),
        "cv_std":     round(float(cv_scores.std()),  4),
        "n_train":    int(len(X_tr)),
        "n_test":     int(len(X_te)),
        "class_dist": {"real": int((y==0).sum()), "fake": int((y==1).sum())},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(model, scaler, feature_names, meta):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    payload = {
        "features":   feature_names,
        "n_features": len(feature_names),
        "model_type": "RandomForestClassifier",
        "metrics":    meta,
    }
    with open(FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    log.info(f"✅  model.pkl     → {MODEL_PATH}")
    log.info(f"✅  scaler.pkl    → {SCALER_PATH}")
    log.info(f"✅  features.json → {FEATURES_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
DATASET_LOADERS = {
    "twitter":    load_dataset_twitter,
    "instagram":  load_dataset_instagram,
    "twitter_v2": load_dataset_twitter_v2,
    "social":     load_dataset_social,
}
ALL_DATASETS = list(DATASET_LOADERS.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Train fake-profile classifier on up to four Kaggle datasets."
    )
    ap.add_argument("--skip-download", action="store_true",
                    help="Use locally-cached kagglehub downloads")
    ap.add_argument("--data-dir", default=None,
                    help=(
                        "Override dataset path. Must contain sub-dirs: "
                        "twitter/, instagram/, twitter_v2/, social/"
                    ))
    ap.add_argument(
        "--dataset",
        default="all",
        help=(
            "Comma-separated list of datasets to include. "
            f"Choices: all, {', '.join(ALL_DATASETS)}. "
            "Example: --dataset twitter,instagram"
        ),
    )
    args = ap.parse_args()

    # Resolve which datasets to load
    if args.dataset.strip().lower() == "all":
        selected = ALL_DATASETS
    else:
        selected = [d.strip() for d in args.dataset.split(",")]
        unknown  = [d for d in selected if d not in DATASET_LOADERS]
        if unknown:
            log.error(f"Unknown dataset(s): {unknown}. Valid: {ALL_DATASETS}")
            sys.exit(1)

    dfs = []
    for name in selected:
        try:
            df = DATASET_LOADERS[name](
                skip_download=args.skip_download,
                data_dir=args.data_dir,
            )
            dfs.append(df)
        except Exception as e:
            log.error(f"Failed to load dataset '{name}': {e}")
            if len(selected) == 1:
                sys.exit(1)   # fatal only if it's the only requested dataset

    if not dfs:
        log.error("No datasets loaded. Exiting.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"\nCombined dataset: {combined.shape[0]:,} rows from {len(dfs)} source(s)")
    log.info(f"Overall label dist: {dict(combined['label'].value_counts())}")

    X, y = build_feature_matrix(combined)
    model, scaler, meta = train(X, y)
    save_artifacts(model, scaler, list(X.columns), meta)

    log.info("\n🎉  Training complete!")
    log.info("   The project will now use this model for predictions.")
    log.info("   Re-run `python ml/train_model.py` any time to retrain.\n")


if __name__ == "__main__":
    main()