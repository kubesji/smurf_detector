import pickle
from player import Player
import matplotlib.pyplot as plt


with open('pantry/players.pickle', 'rb') as file:
    players = pickle.load(file)

players_brackets = {}
smurf_brackets = {}

for p in players:
    bracket = int(p.elo / 100) * 100
    if bracket in players_brackets:
        players_brackets[bracket] += 1
    else:
        players_brackets[bracket] = 1

    if p.is_potential_smurf(min_short_game_ratio=0.1, max_short_game_wins_ratio=0.3, min_long_game_wins_ratio=0.6):
        if bracket in smurf_brackets:
            smurf_brackets[bracket] += 1
        else:
            smurf_brackets[bracket] = 1

data = {}
keys = sorted(players_brackets.keys())
top_players, top_smurfs = 0, 0
for k in keys:
    if k < 2200:
        new_key = f"{k}-{k+99}"
        data[new_key] = 100 * smurf_brackets[k] / players_brackets[k] if k in smurf_brackets else 0
    else:
        top_players += players_brackets[k]
        if k in smurf_brackets:
            top_smurfs += smurf_brackets[k]
data["2200+"] = 100 * top_smurfs / top_players

plt.bar(list(data.keys()), list(data.values()))
plt.xlabel("Elo")
plt.ylabel("# of smurfs")
plt.title("# of smurfs per Elo bracket")
plt.xticks(rotation=45)

plt.show()
