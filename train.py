from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

from utils import load, save
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = load('aoe2insights_september')

manual_labels = {}
with open("training_labels.txt", encoding="utf-8") as f:
    for line in f:
        url, label = line.strip().rsplit(" - ", 1)
        manual_labels[url] = label

mask = df["profile"].isin(manual_labels)
df.loc[mask, "smurf"] = df.loc[mask, "profile"].map(manual_labels)
df = df.drop(columns=["profile_id", "index", "game_id", "started", "map", "winner", "old_elo",
                      "duration", "short_game", "long_game", "opponent_id", "opponent_new_elo",
                      "opponent_old_elo"])

df = df[df["smurf"].isin(["legit", "smurf", "map dodger"])].copy()
df_sus = df[df["smurf"].isin(["smurf", "map dodger"])].copy()# Ignore inconclusive and NaN
df["suspicious"] = df["smurf"].ne("legit").astype(int)
df_sus["map_dodger"] = df_sus["smurf"].eq("map dodger").astype(int)

df = df.loc[:, df.nunique(dropna=False) > 1] # Remove constant columns
# df = df.loc[:, df.isna().mean() < 0.8] # Remove columns with too many NaNs
df = df.loc[:, ~df.T.duplicated()]

features = df.drop(columns=["smurf", "profile", "suspicious"])
corr = features.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [col for col in upper.columns if (upper[col] > 0.92).any()]
df = df.drop(columns=drop)
feature_cols = df.columns.drop(["smurf", "suspicious", "profile"])
save(feature_cols, "features")
# df = df.fillna(0)

lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=10,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )

baseline = make_pipeline(
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)

print("Suspicious profiles")
for name, model in { "logistic (baseline)": baseline, "lgbm": lgbm}.items():
    oof_score = cross_val_predict(model, df[feature_cols], df["suspicious"].to_numpy(), cv=cv, method="predict_proba")[:, 1]
    print(f"{name} ROC AUC:", roc_auc_score(df["suspicious"], oof_score))
    print(f"{name} PR AUC:", average_precision_score(df["suspicious"], oof_score))


lgbm.fit(df[feature_cols], df["suspicious"])
save(lgbm, "lgbm_classifier")
baseline.fit(df[feature_cols], df["suspicious"])
df["smurf_score"] = lgbm.predict_proba(df[feature_cols])[:, 1]

all_players = load('aoe2insights_september')
all_players = all_players.drop_duplicates("profile_id", keep="last")
all_players = all_players[all_players['games_total'] > 25]
unlabeled = all_players[~all_players["profile"].isin(df["profile"])].copy()
unlabeled["smurf_score"] = lgbm.predict_proba(unlabeled[feature_cols])[:, 1]

unlabeled["smurf_score"].hist(bins=np.arange(0, 1.1, 0.1))
plt.xlabel("Smurf score")
plt.ylabel("Players")
plt.yscale("log")
plt.show()

smurfs = unlabeled[unlabeled["smurf_score"] > 0.95].sort_values(by="smurf_score", ascending=False)
smurfs["profile"].to_csv("pantry/classified_smurfs.txt", index=False, header=False)

print('done')
