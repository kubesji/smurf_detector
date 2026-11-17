from utils import load
import pandas as pd


all_players = load('aoe2insights')
all_players = all_players.drop_duplicates("profile_id", keep="last")
all_players = all_players[all_players['games_total'] > 25]

short_loss_cols = [c for c in all_players.columns if "_short_loss_to_total_ratio" in c]

map_dodgers = all_players[
    pd.concat([
        (all_players[c] > 0.8) & (all_players[c.replace("_short_loss_to_total_ratio", "_short_loss_total")] >= 5)
        for c in short_loss_cols
    ], axis=1).any(axis=1)
]
map_dodgers = map_dodgers.sample(50, random_state=42)
short_losses = map_dodgers[short_loss_cols + [c.replace("_short_loss_to_total_ratio", "_short_loss_total") for c in short_loss_cols]]

for url in list(map_dodgers["profile"]):
    print(url)
