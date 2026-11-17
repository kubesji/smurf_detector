from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import load, save
import pandas as pd
import numpy as np

# This script loads training dataframe, trains model and classifies test dataset.
# Results are then used to generate list of profiles to manually check and add to train dataset.

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.72,
        class_weight="balanced",
        max_iter=5000,
    ),
)

tiny_df = load("training_df")
excluded = tiny_df[tiny_df["smurf"].isin(["inconclusive", "map dodger"])].copy()
tiny_df = tiny_df[~tiny_df["smurf"].isin(["map dodger", "inconclusive"])]
tiny_df["target"] = tiny_df["smurf"].eq("smurf").astype(int)
feature_cols = tiny_df.columns.drop(["smurf", "target", "profile"])
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_score = cross_val_predict(
    model, tiny_df[feature_cols], tiny_df["target"], cv=cv, method="predict_proba"
)[:, 1]

print("ROC AUC:", roc_auc_score(tiny_df["target"], oof_score))
print("PR AUC:", average_precision_score(tiny_df["target"], oof_score))

model.fit(tiny_df[feature_cols], tiny_df["target"])
tiny_df["smurf_score"] = model.predict_proba(tiny_df[feature_cols])[:, 1]

print(tiny_df.groupby("smurf")["smurf_score"].describe())

excluded["smurf_score"] = model.predict_proba(excluded[feature_cols])[:, 1]
print(excluded.groupby("smurf")["smurf_score"].describe())

all_players = load('aoe2insights')
all_players = all_players.drop_duplicates("profile_id", keep="last")
all_players = all_players[all_players['games_total'] > 25]
unlabeled = all_players[~all_players["profile"].isin(tiny_df["profile"])].copy()
unlabeled["smurf_score"] = model.predict_proba(unlabeled[feature_cols])[:, 1]

top = unlabeled.nlargest(25, "smurf_score")
boundary = unlabeled.iloc[(unlabeled["smurf_score"] - 0.5).abs().argsort()[:25]]
random = unlabeled.drop(top.index.union(boundary.index)).sample(25, random_state=42)
low = unlabeled.nsmallest(25, "smurf_score")
to_review = pd.concat([top, boundary, random, low]).drop_duplicates()
for url in list(to_review["profile"]):
    print(url)