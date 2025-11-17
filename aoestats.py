from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import requests
from time import sleep
import pandas as pd


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
            with open(f"{db_type}.parquet", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except Exception as e:
            print(f"[Attempt {attempt + 1}/{retries}] Error: {e}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                raise  # re-raise final error

def get_matches_and_players(n_weeks=5, filter_1v1=True):
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

        if filter_1v1: # Filtered here to increase speed - player concat takes massive amount of time
            new_matches = new_matches[new_matches["num_players"] == 2]
            new_players = new_players[new_players["game_id"].isin(new_matches["game_id"])]
        matches.append(new_matches)
        players.append(new_players)

    matches = pd.concat(matches, ignore_index=True)
    players = pd.concat(players, ignore_index=True)


    return matches, players
