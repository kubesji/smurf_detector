import pickle
import aoe2insights as aoe
from player import Player
import time


try:
    with open('pantry/players.pickle', 'rb') as file:
        players = pickle.load(file)
except:
    players = []

last_page = False
page = 0
while not last_page:
    start = time.time()
    page += 1

    try:
        board, last_page = aoe.get_leaderboard(page)
    except Exception as e:
        print(f"##error## Page {page} could not be retrieved: {e}. Waiting for 5 minutes and trying again")
        time.sleep(300)
        page-=1
        continue

    for row in board:
        name, player_id, elo = row
        stats = aoe.get_winrates(player_id)
        player = Player(name, player_id, elo, stats)
        if player not in players:
            players.append(player)

    # Save after every page
    with open('pantry/players.pickle', 'wb') as handle:
        pickle.dump(players, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Page {page}: total {len(players)} of players. Took {time.time() - start:.2f} s")
