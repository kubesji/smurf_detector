# AoE2 Smurf Detector

Machine-learning project for detecting suspicious behaviour in RM 1v1 ladder of **Age of Empires II: Definitive Edition**.

The main targets are identification of:

- smurf accounts
- map dodging
- suspicious alternate-account behaviour

The model is intended mainly for **ranking suspicious players**, not for producing perfectly calibrated probabilities.

## Dataset

The raw dataset contains one row per player per match and includes data such as:

```text
profile id
game id
timestamp of the game
elo
winner
map
match duration
```

The full dataset may contain millions of match records, so memory usage is an important part of the pipeline. 
A 30-day window is used for the algorithm as Nili requested report for last 30 days.

## Features

There are hundreds or even thousands of features created.

Player-level features include:

```text
games_total
wins_total
losses_total
win_ratio

short_games_total
short_loss_total
short_wins_total
short_loss_to_total_ratio

long_games_total
long_loss_total
long_wins_total
long_loss_to_total_ratio

games_last_7_days
games_last_14_days
games_last_28_days

weeks_active
weeks_with_short_loss
weeks_with_short_loss_ratio
```

Elo-related features include:

```text
current_elo
max_elo
short_elo
mid_elo
long_elo

alt_avg_elo   # Elo information about alternative accounts
alt_max_elo
```

***NOTE:*** Elo computation takes extreme amount of time compared to rest of the features. 
If anyone would be able to come up with better solution, I would be glad.

Map-specific features are calculated mainly for important maps such as Arabia and Arena. Other maps are bundled 
into open/closed/water/... categories:

```text
arabia_games_total
arabia_short_loss_total
arabia_short_loss_to_total_ratio

arena_games_total
arena_short_loss_total
arena_short_loss_to_total_ratio
```

## Labels

Profiles are manually reviewed using AoE2Insights.

Current labels are:

```text
legit
smurf
map dodger
inconclusive
```

`inconclusive` profiles are stored but excluded from supervised training.

## Models

Two model types are currently used.

### Logistic Regression

Used mainly as a simple and interpretable baseline.

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5,
    C=0.72,
    class_weight="balanced",
    max_iter=5000,
)
```

### LightGBM

Currently the primary model.

```python
LGBMClassifier(
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
```

## Classification

Smurfs and map dodgers are currently bundled under `suspicious` umbrella. A second stage may then be used to 
distinguish between them but it is unreliable as of now.

## Evaluation

Models are evaluated using stratified cross-validation.

Main metrics:

```text
PR-AUC
ROC-AUC
precision@20
precision@50
precision@100
```

PR-AUC and top-N precision are the most useful because the practical goal is to inspect the most suspicious players first.

## Active Learning

Training dataset is created manually, each profiled checked and classified after review.
New profiles are selected for manual review from:

```text
highest scores
classification boundary (score around 0.4-0.6)
random players
very low scores
```

This gives more useful training data than reviewing only the highest-scoring accounts.

## Dataset

Dataset is not included as I get it directly from admin of aoe2insights.com. Big shout out. 
I can provide it via some file sharing service as it is quite big (~500MB) to store here.

## Current Status

The most imminent need is to make training dataset larger to improve reliability of classification.
Some features may be added but most stuff is covered.

Also, other means of classification can be explored.