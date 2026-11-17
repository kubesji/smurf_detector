from numpy.ma.extras import column_stack

from utils import load, save
import pandas as pd
import numpy as np

# Used to identify smurfs and legit players for initial training dataset. Now more or less obsolete

df = load('aoe2insights')
df = df.drop_duplicates("profile_id", keep="last")
df = df[df['games_total'] > 25]

potential_smurfs = df[((df['number_of_short_loss_5_streaks'] > 1) | (df['number_of_short_loss_3_streaks'] > 2) | (df[
    "short_loss_to_total_ratio"] > 0.5))]
save(potential_smurfs, "potential_smurfs")

non_smurfs = df[~df["profile_id"].isin(potential_smurfs["profile_id"])]
save(non_smurfs, "non_smurfs")

sub_smurfs = potential_smurfs.sample(50, random_state=42)
print("----SMURFS----")
for user in list(sub_smurfs['profile']):
    print(user)
save(sub_smurfs, "sub_smurfs")


print("----NON SMURFS----")
sub_non_smurfs = non_smurfs.sample(50, random_state=42)
for user in list(sub_non_smurfs['profile']):
    print(user)
save(sub_non_smurfs, "sub_non_smurfs")

sub_smurfs['smurf'] = "smurf"
sub_non_smurfs['smurf'] = "legit"
combined = pd.concat([sub_smurfs, sub_non_smurfs], ignore_index=True)
manual_labels = {}

with open("training_labels.txt", encoding="utf-8") as f:
    for line in f:
        url, label = line.strip().rsplit(" - ", 1)
        manual_labels[url] = label

mask = combined["profile"].isin(manual_labels)
combined.loc[mask, "smurf"] = combined.loc[mask, "profile"].map(manual_labels)

combined = combined.drop(columns=["profile_id", "index", "game_id", "started", "map", "winner", "old_elo",
                                  "duration", "short_game", "long_game", "opponent_id", "opponent_new_elo",
                                  "opponent_old_elo"])
combined = combined.copy()
combined = combined.loc[:, combined.nunique(dropna=False) > 1] # Remove constant columns
combined = combined.loc[:, combined.isna().mean() < 0.8] # Remove columns with too many NaNs
combined = combined.T.drop_duplicates().T

numerical = combined.drop(columns=["smurf", "profile"])
corr = numerical.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [col for col in upper.columns if (upper[col] > 0.95).any()]
combined.drop(columns=drop, inplace=True)

ignored = ['nomad_', 'megarandom_', 'hideout_', 'semiopen_', 'hybrid_', 'open_', 'closed_', 'chaos_', 'water_']
combined =combined.drop(columns=[c for c in combined.columns if any(x in c for x in ignored)])

save(combined, "training_df")

print('done')
