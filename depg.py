#!/usr/bin/env python3
"""DepG: depression-signal detection in social media posts.

Trains a TF-IDF + logistic regression classifier on labeled Reddit posts and
scores new text, with the words that pushed the score up or down.

    python depg.py train                      # downloads the dataset, cross-validates, saves the model
    python depg.py predict "I can't get out of bed anymore and nothing feels worth it"
    python depg.py timeline posts.txt         # one post per line, scored in order

This is a research prototype, not a diagnostic tool. A high score means the
writing resembles posts from depression forums; it says nothing reliable about
a specific person. If you or someone you know is struggling, contact a local
crisis line (see README).
"""
import argparse
import csv
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "depression_reddit_cleaned_ds.csv"
MODEL = ROOT / "models" / "depg.pkl"
# mrjunos/depression-reddit-cleaned on Hugging Face (CC BY 4.0): ~7.7k posts, label 1 = from depression subreddits
DATA_URL = "https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned/resolve/main/depression_reddit_cleaned_ds.csv"

SUPPORT_MESSAGE = (
    "If this text is about you or someone close to you: you don't have to handle it alone. "
    "Reach out to someone you trust or a crisis line (e.g. 988 in the US, Mind 90101 in Sweden, "
    "116 123 Samaritans in the UK, findahelpline.com elsewhere)."
)


def load_dataset():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        print(f"downloading {DATA_URL}", file=sys.stderr)
        urllib.request.urlretrieve(DATA_URL, DATA)
    texts, labels = [], []
    with DATA.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # the CSV uses text/labels; the dataset card calls them clean_text/is_depression
            text = (row.get("text") or row.get("clean_text") or "").strip()
            label = row.get("labels", row.get("is_depression"))
            if text and label not in (None, ""):
                texts.append(text)
                labels.append(int(float(label)))
    return texts, np.array(labels)


def build_pipeline():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english")),
        # character n-grams catch misspellings and word fragments ("cant", "anymoreee")
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=50000)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])


def cmd_train(args):
    texts, y = load_dataset()
    print(f"{len(texts)} posts, {y.mean():.0%} labeled depression")

    # Deduplicate before splitting: reposts would otherwise leak between train and test
    seen, keep = set(), []
    for i, t in enumerate(texts):
        key = " ".join(t.lower().split())
        if key not in seen:
            seen.add(key)
            keep.append(i)
    texts = [texts[i] for i in keep]
    y = y[keep]
    print(f"{len(texts)} after removing duplicate posts")

    proba = cross_val_predict(build_pipeline(), texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    print(f"\n5-fold cross-validation ROC AUC: {roc_auc_score(y, proba):.3f}")
    print(classification_report(y, proba >= 0.5, target_names=["not depression", "depression"], digits=3))

    # Hold out a test set once more to check calibration-ish behaviour of the final model
    tr_x, te_x, tr_y, te_y = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=1)
    held = build_pipeline().fit(tr_x, tr_y)
    p = held.predict_proba(te_x)[:, 1]
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        mask = (p >= lo) & (p < hi)
        if mask.any():
            print(f"  score {lo:.1f}-{min(hi, 1):.1f}: {mask.sum():4d} posts, {te_y[mask].mean():.0%} actually labeled depression")

    model = build_pipeline().fit(texts, y)
    MODEL.parent.mkdir(exist_ok=True)
    with MODEL.open("wb") as f:
        pickle.dump(model, f)
    print(f"\nsaved {MODEL}")
    print("\nStrongest indicators:")
    words = model.named_steps["features"].transformer_list[0][1]
    n_words = len(words.vocabulary_)
    coefs = model.named_steps["clf"].coef_[0][:n_words]
    names = words.get_feature_names_out()
    order = np.argsort(coefs)
    print("  depression:", ", ".join(names[i] for i in order[::-1][:15]))
    print("  other:     ", ", ".join(names[i] for i in order[:15]))


def load_model():
    if not MODEL.exists():
        raise SystemExit("no model yet, run: python depg.py train")
    with MODEL.open("rb") as f:
        return pickle.load(f)


def explain(model, text, k=6):
    """Word n-grams in this text that contributed most to the score."""
    words = model.named_steps["features"].transformer_list[0][1]
    coefs = model.named_steps["clf"].coef_[0][:len(words.vocabulary_)]
    vec = words.transform([text]).toarray()[0]
    contrib = vec * coefs
    names = words.get_feature_names_out()
    up = [names[i] for i in np.argsort(contrib)[::-1][:k] if contrib[i] > 0]
    down = [names[i] for i in np.argsort(contrib)[:k] if contrib[i] < 0]
    return up, down


def band(p):
    return "high" if p >= 0.8 else "elevated" if p >= 0.5 else "low"


def cmd_predict(args):
    model = load_model()
    text = " ".join(args.text)
    p = model.predict_proba([text])[0, 1]
    up, down = explain(model, text)
    print(f"score {p:.2f} ({band(p)} resemblance to depression-forum posts)")
    if up:
        print("  raised by:", ", ".join(up))
    if down:
        print("  lowered by:", ", ".join(down))
    if p >= 0.5:
        print("\n" + SUPPORT_MESSAGE)


def cmd_timeline(args):
    model = load_model()
    posts = [line.strip() for line in Path(args.file).read_text().splitlines() if line.strip()]
    scores = model.predict_proba(posts)[:, 1]
    for i, (post, p) in enumerate(zip(posts, scores), 1):
        bar = "#" * int(round(p * 20))
        print(f"{i:3d}  {p:.2f} {bar:<20}  {post[:70]}")
    # a sustained rise matters more than a single bad day
    window = max(1, len(scores) // 4)
    trend = scores[-window:].mean() - scores[:window].mean()
    print(f"\nmean {scores.mean():.2f}, first-vs-last quarter change {trend:+.2f}")
    if scores[-window:].mean() >= 0.5:
        print("\n" + SUPPORT_MESSAGE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    p = sub.add_parser("predict")
    p.add_argument("text", nargs="+")
    p = sub.add_parser("timeline")
    p.add_argument("file")
    args = parser.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "timeline": cmd_timeline}[args.cmd](args)


if __name__ == "__main__":
    main()
