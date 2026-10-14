# DepG
 Sentiment analysis for depression based on social media post

> Research prototype, **not a diagnostic tool**. A score says how much a piece of writing resembles posts from depression forums; it is not a statement about a person's mental health. If you or someone you know is struggling, talk to someone you trust or a crisis line: 988 (US), Mind Självmordslinjen 90101 (Sweden), Samaritans 116 123 (UK), or [findahelpline.com](https://findahelpline.com).

## What it does
`depg.py` trains a TF-IDF (word 1–2-grams + character 3–5-grams) logistic regression on the [depression-reddit-cleaned](https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned) dataset (CC BY 4.0, ~7.7k posts, downloaded on first run), then:
- `predict` scores a post and lists the words that raised or lowered the score
- `timeline` scores a sequence of posts and reports the trend, since a sustained change says more than one bad day

```sh
nix-shell                 # or: pip install -r requirements.txt
python depg.py train
python depg.py predict "I am so tired of everything, I feel empty"
python depg.py timeline examples/timeline.txt
```

## Results, and why not to trust them yet
5-fold cross-validation gives ROC AUC ≈ 0.99, which is too good to be real. The learned indicators show why: the "not depression" class is dominated by Twitter artifacts (`quot`, `com`, `twitter`), while depression posts come from Reddit (`title`, `text`). Even "sad" and "depressing" lean toward the *non-depression* class. **The dataset's label is partly the platform, not the mental state**, so the model learns writing style as much as anything clinical.

Before this could mean anything:
- train on data where both classes come from the same platform and population (e.g. the CLPsych or eRisk shared-task data, which require a data agreement)
- evaluate on users rather than posts, with clinically grounded labels
- check behaviour across dialects, ages and languages
