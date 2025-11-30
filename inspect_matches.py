import pandas as pd
import pickle
import matplotlib.pyplot as plt

def get_winrates(player_df, game_ids, min_count=2):
    selection = player_df[player_df["game_id"].isin(game_ids)]
    selection_loosers = selection[selection["winner"] == 0]
    selection_winners = selection[selection["winner"] == 1]
    selection_counts = selection["profile_id"].value_counts()
    selection_counts = selection_counts[selection_counts >= min_count]
    selection_wins = selection_winners["profile_id"].value_counts()
    selection_wins = selection_wins[selection_wins.index.isin(selection_counts.index)]
    selection_wr = (selection_wins / selection_counts).fillna(0)
    return selection_wr

def get_droprates(player_df, game_ids, min_count=5):
    game_per_player = player_df["profile_id"].value_counts()
    selection = player_df[player_df["game_id"].isin(game_ids)]
    drops = selection[selection["winner"] == 0]
    drops_counts = drops["profile_id"].value_counts()
    drops_counts = drops_counts[drops_counts >= min_count]
    dr = (drops_counts / game_per_player).fillna(0).sort_values(ascending=False)
    return dr

with open('pantry/match_players.pickle', 'rb') as file:
    players = pickle.load(file)

with open('pantry/match_info.pickle', 'rb') as file:
    matches = pickle.load(file)


total_matches = matches.shape[0]
matches["short_game"] = (matches["duration"] < pd.Timedelta(seconds=300))
matches["long_game"] = (matches["duration"] > pd.Timedelta(seconds=600))
matches["bracket"] = (matches["avg_elo"] / 100).astype(int) * 100
matches = matches.sort_values(by=['started_timestamp'])
short_game_ids = matches.index[matches["short_game"]].tolist()
long_game_ids = matches.index[matches["long_game"]].tolist()
n_players = players["profile_id"].value_counts().shape[0]

drop_rates = get_droprates(players, short_game_ids, min_count=5)
drop_rates = drop_rates[drop_rates > 0.2]
drop_wr = get_winrates(players, short_game_ids, min_count=5)
drop_wr = drop_wr[drop_wr < 0.3]
genuine_wr = get_winrates(players, long_game_ids)
genuine_wr = genuine_wr[genuine_wr > 0.6]
smurf_ids = drop_wr.index.intersection(genuine_wr.index)
smurf_ids = drop_rates.index.intersection(smurf_ids).tolist()
smurf_wr = genuine_wr[smurf_ids].sort_values(ascending=False)
smurf_matches = players[players["profile_id"].isin(smurf_ids)]["game_id"].tolist()
smurf_matches = matches.loc[smurf_matches]
abusive_matches = smurf_matches[smurf_matches["duration"] > pd.Timedelta(seconds=300)].index.tolist()
abusive_players = players[players["game_id"].isin(abusive_matches)]
abusive_players = abusive_players[abusive_players["profile_id"].isin(smurf_ids)]
abusive_winners = abusive_players[abusive_players["winner"]]

smurfing_brackets = smurf_matches[smurf_matches["duration"] > pd.Timedelta(seconds=300)]
smurf_games_per_bracket = smurfing_brackets["bracket"].value_counts()
total_games_per_bracket = matches["bracket"].value_counts()
smurfing_per_bracket = 100 * smurf_games_per_bracket / total_games_per_bracket
smurfing_per_bracket = smurfing_per_bracket[smurfing_per_bracket.index <= 2000]
smurfing_per_bracket.plot(kind='bar')
plt.xlabel("Elo")
plt.ylabel("% of smurf games")
plt.title("% of smurf games per Elo bracket")
plt.xticks(rotation=45)
plt.show()

print(f"{len(smurf_ids)} active smurfs of of {n_players}, i.e. {len(smurf_ids)/n_players * 100:.2f} %")
print(f"Games involving smurf: {smurf_matches.shape[0] / total_matches * 100:.2f} %")
print(f"Games actually smurf played: {len(abusive_matches) / total_matches * 100:.2f} % ({len(abusive_matches)}/{total_matches})")
print(f"Games smurf won: {abusive_winners.shape[0] / len(abusive_matches) * 100:.2f} % matches ({abusive_winners.shape[0]}/{len(abusive_matches)})")
print("end")