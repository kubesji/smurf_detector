import pickle
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler


WR_LIMIT = 0.25
MIN_GAMES = 10
SHORT_GAME = pd.Timedelta(minutes=5)
STANDARD_GAME = pd.Timedelta(minutes=10)


with open('pantry/match_players.pickle', 'rb') as file:
    players = pickle.load(file).drop(columns=["opening", "match_rating_diff", "team"]).set_index("game_id")

with open('pantry/match_info.pickle', 'rb') as file:
    matches = pickle.load(file)

complete = players.merge(matches, how="left", left_index=True, right_index=True).reset_index()
players_grouped = complete.groupby("profile_id")

df = pd.DataFrame()
df["n_games"] = players_grouped.size()
short_games = complete[complete['duration'] < SHORT_GAME].groupby("profile_id")
standard_games = complete[complete['duration'] > STANDARD_GAME].groupby("profile_id")
df["n_short_games"] = short_games.size()
df["short_games"] = df["n_short_games"] / df["n_games"]

df["short_games_wr"] = short_games["winner"].sum() / df["n_short_games"]
df["standard_games_wr"] = standard_games["winner"].sum() / standard_games.size()

df[f"duration_avg"] = players_grouped["duration"].mean().dt.total_seconds() / 60  # minutes
df[f"duration_std"] = players_grouped["duration"].std().dt.total_seconds() / 60  # minutes

for m in ["arabia", "arena", "other_maps"]:
    if m != "other_map":
        tmp = complete[complete["map"] == m]
    else:
        tmp = complete[~complete["map"].isin(["arena", "arabia"])]
    short_games = tmp[tmp['duration'] < SHORT_GAME]
    grouped = tmp.groupby("profile_id")
    short_games_grouped = short_games.groupby("profile_id")
    df[f"{m}_n_games"] = grouped.size()
    df[f"{m}_games"] = df[f"{m}_n_games"] / df["n_games"]
    df[f"{m}_wr"] = grouped['winner'].sum() / df[f"{m}_n_games"]
    df[f"{m}_short_games"] = short_games_grouped.size() / df[f"{m}_n_games"]
    df[f"{m}_short_games_wr"] = short_games_grouped['winner'].sum() / short_games_grouped.size()
    df[f"{m}_duration_avg"] = grouped["duration"].mean().dt.total_seconds() / 60 # minutes
    df[f"{m}_duration_std"] = grouped["duration"].std().dt.total_seconds() / 60 # minutes

top_civs = (
    complete
        .groupby("profile_id")["civ"]
        .value_counts()
        .groupby(level=0)
        .head(3)
        .reset_index(name="count")
        .assign(rank=lambda d: d.groupby("profile_id")["count"].rank(method="first", ascending=False))
        .pivot(index="profile_id", columns="rank", values="count")
        .rename(columns={1.0: "civ_top1", 2.0: "civ_top2", 3.0: "civ_top3"})
    )
top_civs = top_civs.div(df["n_games"], axis=0)
df = df.merge(top_civs, left_index=True, right_index=True, how='left')

print(f"Total size: {df.shape}")
df = df[df["n_games"] > MIN_GAMES]
print(f"Without less then {MIN_GAMES} games: {df.shape}")

# Win rate NaNs filled with zero mess clustering -> filling with 50% WR. Everything else filled with zero
cols = df.filter(regex=r"_wr$").columns.tolist()
df[cols] = df[cols].fillna(0.5)
df = df.fillna(0)

data = RobustScaler().fit_transform(df)
ids = df.index.tolist()

# It is relatively stable around 15-25, comes back as 636 smurfs no matter the number of clusters
model = AgglomerativeClustering(n_clusters=20)
labels = model.fit_predict(data, ids)

# Add info to dataframe
df["cluster"] = labels
df["elo_avg"] = players_grouped['new_rating'].mean()

centroids =df.groupby("cluster").mean()
cols = df.filter(regex=r"_wr$").columns.tolist()
smurf_cluster = centroids[centroids[cols].lt(WR_LIMIT).any(axis=1)]
smurf_cluster = smurf_cluster[smurf_cluster['short_games'].gt(WR_LIMIT)]
smurf_labels = smurf_cluster.index.tolist()

smurfs = df[df["cluster"].isin(smurf_labels)]
print(f"# of smurfs: {smurfs.shape[0]} out of {df.shape[0]} players, i.e. {100 * smurfs.shape[0] / df.shape[0]:.2f}%")
print(f"# of smurf games: {smurfs["n_games"].sum()} out of {df["n_games"].sum()} games, "
      f"i.e. {100 * smurfs["n_games"].sum() / df["n_games"].sum():.2f}%")
smurfs_ids = '\n'.join(smurfs.index.astype(str).tolist())

with open('pantry/smurfs.pickle', 'wb') as file:
    pickle.dump(smurfs, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('pantry/smurfs.txt', 'w') as file:
    file.write(smurfs_ids)