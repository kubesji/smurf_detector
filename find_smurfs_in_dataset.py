import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

import requests
import pycountry

from utils import load, save
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Classifies players in given dataset

def get_country(profile_id):
    url = "https://aoe-api.worldsedgelink.com/community/leaderboard/getPersonalStat"

    r = requests.get(
        url,
        params={
            "title": "age2",
            "profile_ids": f"[{profile_id}]",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()

    code = data['statGroups'][0]['members'][0]['country']
    country = pycountry.countries.get(alpha_2=code.upper())
    print(country.name)
    return country.name if country else None

df = load('aoe2insights')
cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=30)
df = df[df["started"] >= cutoff]
lgbm = load('lgbm_classifier')
feature_cols = load('features')

df["elo_bin"] = (df["old_elo"] // 100 * 100).astype(int)
df["games_bin"] = (df["games_total"] // 10 * 10).astype(int)

df = df.drop_duplicates("profile_id", keep="last")
df = df[df['games_total'] > 25]
df["smurf_score"] = lgbm.predict_proba(df[feature_cols])[:, 1]

smurfs = df[df["smurf_score"] > 0.90].sort_values(by="smurf_score", ascending=False)
smurfs["profile"].to_csv("pantry/classified_smurfs.txt", index=False, header=False)
control = df[df['smurf_score'] < 0.2].sample(n=smurfs.shape[0], random_state=27)

elo_counts = pd.DataFrame({
    "Smurfs": smurfs["elo_bin"].value_counts().sort_index(),
    "Legit": control["elo_bin"].value_counts().sort_index(),}).fillna(0)
elo_counts.index = [f"{x}+" for x in elo_counts.index]

game_counts = pd.DataFrame({
    "Smurfs": smurfs["games_bin"].clip(upper=250).value_counts().sort_index(),
    "Legit": control["games_bin"].clip(upper=250).value_counts().sort_index(),}).fillna(0)
game_counts.index = [f"{x}+" for x in game_counts.index]

smurfs["country"] = smurfs["profile_id"].apply(get_country) # Get countries and rename long ones
smurfs["country"] = smurfs["country"].replace({"Taiwan, Province of China": "Taiwan", "Russian Federation": "Russia"})
country_counts = smurfs["country"].value_counts()

fig, ax = plt.subplots(2, 2, figsize=(20, 12))

df["smurf_score"].hist(bins=20, ax=ax[0, 0])
ax[0, 0].set_title("Score")
ax[0, 0].set_xlabel("Smurf score")
ax[0, 0].set_ylabel("Number of players")
# ax[0, 0].set_yscale('log')

smurfs["country"].value_counts().head(25).plot.bar(ax=ax[0, 1])
ax[0, 1].set_title("Country")
ax[0, 1].set_xlabel("Smurf score")
ax[0, 1].set_ylabel("Number of players")
ax[0, 1].tick_params(axis="x", rotation=45)

elo_counts.plot.bar(ax=ax[1, 0])
ax[1, 0].set_title("Elo distribution")
ax[1, 0].set_xlabel("Elo range")
ax[1, 0].set_ylabel("Number of players")

game_counts.plot.bar(ax=ax[1, 1])
ax[1, 1].set_title("Activity")
ax[1, 1].set_xlabel("Number of games played")
ax[1, 1].set_ylabel("Number of players")

plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
plt.show()

