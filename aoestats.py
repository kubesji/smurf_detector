from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import requests
from time import sleep
import pandas as pd
import os


base_url = "https://aoestats.io"

def get_list_of_dumps(retries=3, delay=1.0):

    for attempt in range(retries):
        try:
            response = requests.get(f"{base_url}/api/db_dumps/?format=json", timeout=10)
            response.raise_for_status()
            return response.json()["db_dumps"]

        except Exception as e:
            print(f"[Attempt {attempt + 1}/{retries}] Error: {e}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                raise  # re-raise final error

def get_dump(url, retries=3, delay=1.0):

    for attempt in range(retries):
        try:
            response = requests.get(f"{base_url}{url}", timeout=10, stream=True)
            response.raise_for_status()

            db_type = url.split("/")[-1].split(".")[0]
            with open(f"pantry/{db_type}.parquet", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except Exception as e:
            print(f"[Attempt {attempt + 1}/{retries}] Error: {e}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                raise  # re-raise final error

def get_1v1_matches(n_weeks=4):
    dumps = get_list_of_dumps()
    dumps = sorted(dumps, key=lambda d: d["start_date"], reverse=True)
    matches = []
    players = []

    today = dt.now()
    start = today - relativedelta(weeks=n_weeks)
    end_date = today.strftime("%Y-%m-%d")
    start_date = start.strftime("%Y-%m-%d")

    for db in dumps:
        if start_date > db["end_date"] or end_date < db["start_date"]:
            continue

        get_dump(db["matches_url"])
        get_dump(db["players_url"])

        new_matches = pd.read_parquet("pantry/matches.parquet", engine="pyarrow")
        new_players = pd.read_parquet("pantry/players.parquet", engine="pyarrow")
        new_matches = new_matches[new_matches["num_players"] == 2]
        new_matches = new_matches[new_matches["leaderboard"] == "random_map"]
        new_matches = new_matches[["game_id", "map", "started_timestamp", "duration", "avg_elo"]]
        new_players = new_players[new_players["game_id"].isin(new_matches["game_id"])]
        new_players = new_players.drop(columns=["feudal_age_uptime", "castle_age_uptime", "imperial_age_uptime", "replay_summary_raw"])
        matches.append(new_matches)
        players.append(new_players)

    matches = pd.concat(matches, ignore_index=True)
    matches = matches.set_index("game_id")
    players = pd.concat(players, ignore_index=True)

    return matches, players

matches, players = get_1v1_matches()
import pickle

with open("pantry/match_info.pickle", "wb") as handle:
    pickle.dump(matches, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("pantry/match_players.pickle", "wb") as handle:
    pickle.dump(players, handle, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.exists("pantry/players.parquet"):
    os.remove("pantry/players.parquet")

if os.path.exists("pantry/matches.parquet"):
    os.remove("pantry/matches.parquet")
