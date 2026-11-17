import requests
import numpy as np
import pandas as pd
from time import sleep
import pickle
import os
import sys

from utils import load, save

# Computes all features from aoe2insights dataset

maps = ["arena", "arabia", "nomad", "megarandom", "hideout"]
map_to_cat = {'enclosed': 'semiopen', 'african_clearing': 'nomadic', 'four_lakes': 'hybrid', 'gold_rush': 'open',
              'fortress': 'closed', 'lowland': 'open', 'haboob': 'chaos', 'land_nomad': 'nomadic', 'yucatan': 'semiopen',
              'valley': 'open', 'black_forest': 'closed', 'glade': 'semiopen', 'socotra': 'chaos', 'steppe': 'nomadic',
              'acropolis': 'open', 'scandanavia': 'open', 'golden_swamp': 'open', 'goldenpit': 'open',
              'land_madness': 'chaos', 'runestones': 'open', 'atacama': 'chaos', 'migration': 'water',
              'kawasan': 'open', 'wade': 'open', 'islands': 'water', 'marketplace': 'open', 'baltic': 'hybrid',
              'graveyards': 'open', 'cenotes': 'open', 'continental': 'hybrid', 'shoals': 'hybrid',
              'ghost_lake': 'open', 'fortified_clearing': 'semiopen', 'frigid_lake': 'hybrid', 'acclivity': 'open',
              'mediterranean': 'hybrid', 'golden_pit': 'open', 'highland': "hybrid", 'michi': 'closed',
              'pacific_islands': 'water', 'mountain_pass': 'nomadic', 'mongolia': 'open', 'amazon_tunnel': 'closed',
              'coastal': 'hybrid', 'hill_fort': 'closed', 'canals': 'hybrid', 'morass': 'chaos',
              'passage': 'hybrid', 'coastal_forest': 'hybrid', 'crossroads': 'hybrid', 'shrubland': 'open',
              'sherwood_forest': 'open', 'capricious': 'hybrid', 'paradise_island': 'hybrid', 'salt_marsh': 'hybrid',
              'team_islands': 'water', 'northern_isles': 'water', 'hamburger': 'hybrid', 'serengeti': 'open',
              'bog_islands': 'hybrid', 'prairie': 'open', 'kilimanjaro': 'open', 'greenland': 'hybrid',
              'budapest': 'chaos', 'oasis': 'open', 'archipelago': 'water', 'alpine_lakes': 'open', 'bogland': 'open',
              'city_of_lakes': 'hybrid', 'lombardia': 'open', 'hillfort': 'closed', 'qs_runestones': 'open',
              'qs_arena': 'closed'}


SHORT_GAME_THRESHOLD = pd.Timedelta(seconds=5 * 60)
LONG_GAME_THRESHOLD = pd.Timedelta(seconds=15 * 60)


def get_map_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == 'mega_random':
        subset = 'megarandom'

    if subset in maps:
        subset = [subset]
    else:
        subset = [m for m in map_to_cat.keys() if map_to_cat[m] == subset]

    return df[df["map"].isin(subset)]


def calc_expectation(player_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))


def map_specific_elo(df: pd.DataFrame, prefix: str, k_factor=32.0):
    grouped = df.groupby("game_id", sort=False)
    iterations = len(grouped)
    ratings: dict[int, float] = {}
    df[f"{prefix}old_elo"] = np.nan
    df[f"{prefix}new_elo"] = np.nan

    for i, (game_id, game) in enumerate(grouped):
        if (i + 1) % 1000 == 0:
            print(f"Calculating {prefix} Elo of game {i + 1} / {iterations} ....")

        row_1 = game.iloc[0]
        row_2 = game.iloc[1]

        player_1 = row_1["profile_id"]
        player_2 = row_2["profile_id"]

        if player_1 == player_2:
            raise ValueError(f"Game {game_id!r} contains the same player twice")

        old_elo_1 = ratings.get(player_1, float(row_1[f"old_elo"]))
        old_elo_2 = ratings.get(player_2, float(row_2[f"old_elo"]))

        result_1 = float(row_1["winner"])
        result_2 = float(row_2["winner"])

        expected_1 = calc_expectation(old_elo_1, old_elo_2)
        expected_2 = 1.0 - expected_1

        new_elo_1 = old_elo_1 + k_factor * (result_1 - expected_1)
        new_elo_2 = old_elo_2 + k_factor * (result_2 - expected_2)

        df.loc[row_1.name, f"{prefix}old_elo"] = old_elo_1
        df.loc[row_1.name, f"{prefix}new_elo"] = new_elo_1

        df.loc[row_2.name, f"{prefix}old_elo"] = old_elo_2
        df.loc[row_2.name, f"{prefix}new_elo"] = new_elo_2

        ratings[player_1] = new_elo_1
        ratings[player_2] = new_elo_2


def calculate_duration_elos(df: pd.DataFrame, k_factor: float = 32.0, prefix: str = "") -> pd.DataFrame:
    if prefix != "":
        prefix += "_"

    if f"{prefix}old_long_elo" in df.columns:
        print(f"Custom {prefix} Elos already in ....")
        return df

    print(f"Computing additional {prefix} Elos ....")

    required = {"game_id", "profile_id", "old_elo", "winner", "short_game", "long_game", "started"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # A game cannot be both short and long.
    invalid_duration = df["short_game"] & df["long_game"]
    if invalid_duration.any():
        raise ValueError(f"{invalid_duration.sum()} rows are marked as both short and long")

    # Each game must contain exactly two rows.
    game_sizes = df.groupby("game_id").size()
    invalid_games = game_sizes[game_sizes != 2]
    if not invalid_games.empty:
        raise ValueError(f"{len(invalid_games)} games do not contain exactly two player rows")

    # Determine the category once per row.
    df["duration_category"] = np.select([df["short_game"].to_numpy(dtype=np.bool_, na_value=False),
                                         df["long_game"].to_numpy(dtype=np.bool_, na_value=False)],
                                        ["short", "long"], default="mid")

    # Output columns.
    categories = ("short", "mid", "long")
    for category in categories:
        df[f"{prefix}old_{category}_elo"] = np.nan
        df[f"{prefix}new_{category}_elo"] = np.nan

    # Current category-specific Elo per player.
    ratings: dict[str, dict[object, float]] = {"short": {}, "mid": {}, "long": {}}

    grouped = df.groupby("game_id", sort=False)
    iterations = len(grouped)

    if prefix != "":
        map_specific_elo(df, prefix, k_factor=k_factor)

    for i, (game_id, game) in enumerate(grouped):
        if game["duration_category"].nunique() != 1:
            raise ValueError(f"Rows for game {game_id!r} have different duration categories")

        if (i + 1) % 1000 == 0:
            print(f"Calculating {prefix} Elo of game {i + 1} / {iterations} ....")

        category = game["duration_category"].iloc[0]

        row_1 = game.iloc[0]
        row_2 = game.iloc[1]

        player_1 = row_1["profile_id"]
        player_2 = row_2["profile_id"]

        if player_1 == player_2:
            raise ValueError(f"Game {game_id!r} contains the same player twice")

        old_elos_1 = {cat: ratings[cat].get(player_1, float(row_1[f"{prefix}old_elo"])) for cat in categories}
        old_elos_2 = {cat: ratings[cat].get(player_2, float(row_2[f"{prefix}old_elo"])) for cat in categories}

        result_1 = float(row_1["winner"])
        result_2 = float(row_2["winner"])

        try:
            expected_1 = calc_expectation(old_elos_1[category], old_elos_2[category])
            expected_2 = 1.0 - expected_1

            new_elo_1 = old_elos_1[category] + k_factor * (result_1 - expected_1)
            new_elo_2 = old_elos_2[category] + k_factor * (result_2 - expected_2)
        except:
            print("WTF")

        index_1 = row_1.name
        index_2 = row_2.name

        for cat in categories:
            df.loc[index_1, f"{prefix}old_{cat}_elo"] = old_elos_1[cat]
            df.loc[index_1, f"{prefix}new_{cat}_elo"] = new_elo_1 if cat == category else old_elos_1[cat]

            df.loc[index_2, f"{prefix}old_{cat}_elo"] = old_elos_2[cat]
            df.loc[index_2, f"{prefix}new_{cat}_elo"] = new_elo_2 if cat == category else old_elos_2[cat]

            ratings[cat][player_1] = new_elo_1 if cat == category else old_elos_1[cat]
            ratings[cat][player_2] = new_elo_2 if cat == category else old_elos_2[cat]

    df = df.drop(columns=["duration_category"])
    return df


def add_ratio(df: pd.DataFrame, numerator: str, denominator: str, new_name: str):
    df[new_name] = df[numerator] / df[denominator]


def add_game_counts(df: pd.DataFrame, n_days: list, prefix: str = "") -> pd.DataFrame:
    if prefix != "":
        prefix += "_"

    if f"{prefix}short_wins_last_{n_days[-1]}_days" in df.columns:
        print(f"Game counts {f'for {prefix}' if prefix != "" else ""} already in ....")
        return df

    old_cols = df.columns.copy()

    value_columns = ["_game", "_short", "_mid", "_long", "_win", "_loss", "_long_win", "_mid_win", "_short_win",
                     "_short_loss", "_mid_loss", "_long_loss"]
    # Convert Boolean indicators to 0/1.
    df["_game"] = np.int8(1)
    df["_win"] = df["winner"].astype(np.int8)
    df["_loss"] = 1 - df["winner"].astype(np.int8)
    df["_short"] = df["short_game"].astype(np.int8)
    df["_long"] = df["long_game"].astype(np.int8)
    df["_mid"] = (~(df["short_game"] | df["long_game"])).astype(np.int8)
    df["_short_win"] = (df["short_game"] & df["winner"]).astype(np.int8)
    df["_short_loss"] = (df["short_game"] & (~df["winner"])).astype(np.int8)
    df["_long_win"] = (df["long_game"] & df["winner"]).astype(np.int8)
    df["_long_loss"] = (df["long_game"] & (~df["winner"])).astype(np.int8)
    df["_mid_win"] = ((~(df["short_game"] | df["long_game"])) & df["winner"]).astype(np.int8)
    df["_mid_loss"] = ((~(df["short_game"] | df["long_game"])) & (~df["winner"])).astype(np.int8)
    df["_id"] = df.index

    player_groups = df.groupby("profile_id", sort=False)

    print("Computing cumulative game counts and wins up to and including the current row ....")
    df[f"{prefix}games_total"] = player_groups.cumcount() + 1
    df[f"{prefix}short_games_total"] = player_groups["_short"].cumsum()
    df[f"{prefix}mid_games_total"] = player_groups["_mid"].cumsum()
    df[f"{prefix}long_games_total"] = player_groups["_long"].cumsum()

    df[f"{prefix}wins_total"] = player_groups["_win"].cumsum()
    df[f"{prefix}losses_total"] = player_groups["_loss"].cumsum()
    df[f"{prefix}short_wins_total"] = player_groups["_short_win"].cumsum()
    df[f"{prefix}short_loss_total"] = player_groups["_short_loss"].cumsum()
    add_ratio(df, f"{prefix}short_loss_total", f"{prefix}losses_total", f"{prefix}short_loss_to_total_ratio")
    df[f"{prefix}mid_wins_total"] = player_groups["_mid_win"].cumsum()
    df[f"{prefix}mid_loss_total"] = player_groups["_mid_loss"].cumsum()
    add_ratio(df, f"{prefix}mid_loss_total", f"{prefix}losses_total", f"{prefix}mid_loss_to_total_ratio")
    df[f"{prefix}long_wins_total"] = player_groups["_long_win"].cumsum()
    df[f"{prefix}long_loss_total"] = player_groups["_long_loss"].cumsum()
    add_ratio(df, f"{prefix}long_loss_total", f"{prefix}losses_total", f"{prefix}long_loss_to_total_ratio")

    for n in n_days:
        print(f"Computing game counts and wins within last {n} days ....")
        rolling_counts = (
            df[["profile_id", 'started', *value_columns]]
            .groupby("profile_id", sort=False)
            .rolling(window=f"{n}D", on='started', closed="both", min_periods=1)
            .sum(numeric_only=True)
            .reset_index(level="profile_id", drop=True)[
                value_columns]  # Remove profile_id from the MultiIndex, leaving _id.
            .rename(columns={"_game": f"{prefix}games_last_{n}_days",
                             "_win": f"{prefix}wins_last_{n}_days",
                             "_loss": f"{prefix}losses_last_{n}_days",
                             "_short": f"{prefix}short_games_last_{n}_days",
                             "_mid": f"{prefix}mid_games_last_{n}_days",
                             "_long": f"{prefix}long_games_last_{n}_days",
                             "_short_win": f"{prefix}short_wins_last_{n}_days",
                             "_short_loss": f"{prefix}short_loss_last_{n}_days",
                             "_mid_win": f"{prefix}mid_wins_last_{n}_days",
                             "_mid_loss": f"{prefix}mid_loss_last_{n}_days",
                             "_long_win": f"{prefix}long_wins_last_{n}_days",
                             "_long_loss": f"{prefix}long_losses_last_{n}_days"})
        )

        df = df.join(rolling_counts, how="left", validate="one_to_one")
        add_ratio(df, f"{prefix}short_loss_last_{n}_days", f"{prefix}losses_last_{n}_days",
                  f"{prefix}short_to_total_loss_{n}_days")
        add_ratio(df, f"{prefix}short_wins_last_{n}_days", f"{prefix}wins_last_{n}_days",
                  f"{prefix}short_to_total_win_{n}_days")

    df = df.drop(columns=df.columns[df.columns.str.startswith("_")])
    new_cols = df.columns.difference(old_cols)
    df[new_cols] = df.groupby("profile_id", sort=False)[new_cols].ffill()

    return df


def add_short_loss_duration_variance(df: pd.DataFrame, prefix: str = ""):
    if prefix != "":
        prefix += "_"

    if "short_loss_duration_mean" in df.columns:
        print("Short loss variance, std, ... already in ....")
        return df

    print("Adding short loss median, std and variance coefficient ....")
    df["_short_loss"] = (df["short_game"] & (~df["winner"]))
    df["duration"] = pd.to_timedelta(df["duration"])
    df["_short_loss_dur"] = df["duration"].where(df["_short_loss"]).dt.total_seconds()
    grouped = df.groupby("profile_id", sort=False)
    df["_short_losses_so_far"] = grouped["_short_loss"].cumsum().astype(np.int64)
    df[f"{prefix}short_loss_duration_mean"] = (grouped["_short_loss_dur"].expanding(min_periods=1).mean()
                                               .droplevel("profile_id"))
    df[f"{prefix}short_loss_duration_std"] = (grouped["_short_loss_dur"].expanding(min_periods=1).std()
                                              .droplevel("profile_id"))
    df[f"{prefix}short_loss_duration_coef"] = (df[f"{prefix}short_loss_duration_std"] /
                                               df[f"{prefix}short_loss_duration_mean"])

    df = df.drop(columns=["_short_loss", "_short_loss_dur", "_short_losses_so_far"])
    return df


def add_streak(df: pd.DataFrame, streak_type: str, count_n_streaks: list = None, prefix: str = ""):
    if prefix != "":
        prefix += "_"

    if f"_{prefix}current_{streak_type}_streak" in df.columns:
        print(f"{prefix}{streak_type} already in ....")
        return df

    print(f"Adding {prefix}{streak_type} streak ....")

    if streak_type == "mid_long_win":
        df['_tmp'] = (~df["short_game"]) & df["winner"]
    elif streak_type == "long_win":
        df['_tmp'] = df["long_game"] & df["winner"]
    elif streak_type == "short_loss":
        df['_tmp'] = (df["short_game"] & (~df["winner"]))
    elif streak_type == "loss":
        df['_tmp'] = ~df["winner"]
    elif streak_type == "win":
        df['_tmp'] = df["winner"]
    else:
        raise ValueError("Invalid type")

    previous_value = df.groupby("profile_id")["_tmp"].shift(fill_value=False).astype(int)

    df['_tmp'] = df['_tmp'].astype(int)
    streak_start = df['_tmp'].gt(previous_value)

    df["_run_id"] = streak_start.groupby(df["profile_id"]).cumsum()
    df["_run_length"] = df['_tmp'].groupby([df["profile_id"], df["_run_id"]]).cumsum() * df['_tmp']

    df[f"_{prefix}current_{streak_type}_streak"] = (df['_tmp'].groupby([df["profile_id"], df["_run_id"]]).cumsum() *
                                                   df['_tmp'])

    if count_n_streaks is not None:
        for c in count_n_streaks:
            df["_n_streak"] = df[f"_{prefix}current_{streak_type}_streak"].eq(c)
            df[f"{prefix}number_of_{streak_type}_{c}_streaks"] = df["_n_streak"].groupby(df["profile_id"]).cumsum()
            # When the streak first exceeds N, count the whole streak so far.
            # For every additional game in that streak, add one.
            df["_streak_games"] = np.select([df["_run_length"].eq(c), df["_run_length"].gt(c)],
                                            [df["_run_length"], 1], default=0)
            df[f"{prefix}n_of_{streak_type}_in_{c}_streaks"] = df.groupby("profile_id")["_streak_games"].cumsum()
            df[f"{prefix}ratio_of_{streak_type}_in_{c}_streaks"] = (df[f"{prefix}n_of_{streak_type}_in_{c}_streaks"] /
                                                                    df[f"{prefix}games_total"])

    df = df.drop(columns=["_tmp", "_run_id", "_run_length", "_n_streak", "_streak_games"])
    return df


def add_activity(df: pd.DataFrame):
    if "weeks_active" in df.columns:
        print("Activity already in ....")
        return df

    print("Adding activity ....")

    df["_short_loss"] = (df["short_game"] & (~df["winner"]))

    df["_week"] = df["started"].dt.normalize() - pd.to_timedelta(df["started"].dt.weekday, unit="D")

    weekly = df.groupby(["profile_id", "_week"], as_index=False).agg(_short_loss=("_short_loss", "sum"))

    weekly["weeks_active"] = weekly.groupby("profile_id").cumcount() + 1
    weekly["weeks_with_short_loss"] = weekly["_short_loss"].gt(0).groupby(weekly["profile_id"]).cumsum()
    weekly["weeks_with_short_loss_ratio"] = weekly["weeks_with_short_loss"] / weekly["weeks_active"]

    # df = df.merge(weekly[["profile_id", "_week", "weeks_active", "weeks_with_short_loss"]],
    #               on=["profile_id", "_week"], how="left", validate="many_to_one")
    df = df.join(weekly.set_index(["profile_id", "_week"])[["weeks_active", "weeks_with_short_loss",
                                                            "weeks_with_short_loss_ratio"]],
                 on=["profile_id", "_week"],how="left",validate="many_to_one")

    df = df.drop(columns=["_short_loss", "_week"])
    return df


def get_alt_info(df: pd.DataFrame, alts: dict):
    links = pd.DataFrame(
        [(profile_id, alt_profile_id) for profile_id, alt_ids in alts.items() for alt_profile_id in alt_ids if
         alt_profile_id != profile_id],
        columns=["profile_id", "alt_profile_id"]).drop_duplicates()

    alt_games = df[["profile_id", "new_elo", "started"]].rename(
        columns={"profile_id": "alt_profile_id", "new_elo": "alt_elo", })
    alt_games["alt_elo"] = alt_games["alt_elo"]
    alt_events = links.merge(alt_games, on="alt_profile_id", how="inner", validate="many_to_many")
    alt_events["alt_profile_id"] = alt_events["alt_profile_id"].astype(int)

    alt_events = alt_events.sort_values(["profile_id", "started", "alt_profile_id"], kind="stable")

    grouped = alt_events.groupby("profile_id", sort=False)
    alt_events["alt_games_total"] = (grouped.cumcount() + 1).astype(int)
    valid_elo = alt_events["alt_elo"].notna()

    alt_events["_elo_count"] = (valid_elo.astype(np.int64).groupby(alt_events["profile_id"]).cumsum())
    alt_events["_elo_sum"] = (alt_events["alt_elo"].fillna(0).groupby(alt_events["profile_id"]).cumsum())
    alt_events["alt_avg_elo"] = alt_events["_elo_sum"] / alt_events["_elo_count"].replace(0, np.nan)
    alt_events["alt_max_elo"] = grouped["alt_elo"].cummax()
    alt_events["alt_max_elo"] = alt_events.groupby("profile_id")["alt_max_elo"].ffill()

    df["profile_id"] = df["profile_id"].astype(int)
    left = df[["profile_id", "started"]]
    left["profile_id"] = left["profile_id"].astype(int)
    right = alt_events[["profile_id", "started", "alt_games_total", "alt_avg_elo",
                        "alt_max_elo"]].sort_values(["started", "profile_id"], kind="stable")
    alt_metrics = pd.merge_asof(left, right, on="started", by="profile_id", direction="backward",
                                allow_exact_matches=False)

    df = df.merge(alt_metrics[["alt_games_total", "alt_avg_elo", "alt_max_elo"]], left_index=True, right_index=True,
                  how="left", validate="one_to_one")
    df["alt_games_total"] = df["alt_games_total"].fillna(0).astype(np.int64)
    df["alt_avg_elo"] = df["alt_avg_elo"].fillna(df["old_elo"])
    df["alt_max_elo"] = df["alt_max_elo"].fillna(df["max_elo"])
    df["alt_max_elo_vs_current_diff"] = df["alt_max_elo"] - df["new_elo"]
    df["alt_avg_elo_vs_current_diff"] = df["alt_avg_elo"] - df["new_elo"]
    return df


def find_alt_accounts(df: pd.DataFrame):
    if "alt_games_total" in df.columns:
        print("Alt accounts already in ....")
        return df

    print("Getting alt accounts ....")

    alt_accounts = load('alt_accounts')
    # alt_accounts = None # Force refresh
    alt_accounts = {} if alt_accounts is None else alt_accounts

    session = requests.Session()
    session.headers.update({"User-Agent": "https://github.com/kubesji/smurf_detector", "Accept": "application/json"})

    accounts = list(df['profile_id'].unique())

    for i, acc in enumerate(accounts):
        if acc in alt_accounts.keys():
            continue
        print(f"Getting account {i + 1} / {len(accounts)}")

        url = (f"https://data.aoe2companion.com/api/profiles/{acc}")
        tries = 10
        linked = []
        for _ in range(tries):
            try:
                response = session.get(url, timeout=30)
            except:
                continue

            if response.status_code == 429:
                print("Waiting 2.5s due to rate limiting ....")
                sleep(2.5)
            elif 200 <= response.status_code <= 300:
                profile = response.json()
                linked = profile.get("linked_profiles", profile.get("linkedProfiles", []))
                linked = [item['profileId'] for item in linked]
                break

        alt_accounts[acc] = linked

        if (i + 1) % 1000 == 0:
            save(alt_accounts, 'alt_accounts')

        sleep(0.1)

    save(alt_accounts, 'alt_accounts')
    return get_alt_info(df, alt_accounts)


def calc_elo_stats(df: pd.DataFrame, prefix: str = ""):
    if prefix != "":
        prefix += "_"

    if f"{prefix}max_elo" in df.columns:
        print("Elo stats already in ....")
        return df

    print("Calculating elo stats ....")

    df[f"{prefix}old_elo"] = df[f"{prefix}old_elo"]
    grouped = df.groupby("profile_id", sort=False)
    df[f"{prefix}max_elo"] = grouped[f"{prefix}old_elo"].cummax()
    df[f"{prefix}max_elo_25"] = df.groupby("profile_id", sort=False)[f"{prefix}old_elo"].transform(
        lambda x: x.rolling(window=25, min_periods=1).max())

    return df


def long_short(df: pd.DataFrame):
    if "short_game" in df.columns:
        print("Long/short indicators already in ....")
        return

    print("Adding short/long identifiers ....")
    df["short_game"] = df["duration"] < SHORT_GAME_THRESHOLD
    df["long_game"] = df["duration"] > LONG_GAME_THRESHOLD
    return


def add_opponents(df: pd.DataFrame):
    if "opponent_id" in df.columns:
        print("Opponent data already in ....")
        return df

    print("Adding opponent data ....")
    int_cols = ["game_id", "profile_id"] # TODO: add "old_elo", "new_elo"
    df[int_cols] = df[int_cols].astype(int)
    opponents = df[["game_id", "profile_id", "new_elo", "old_elo"]]
    opponents = opponents.rename(columns={"profile_id": "opponent_id",
                                          "new_elo": "opponent_new_elo",
                                          "old_elo": "opponent_old_elo"})

    df = df.merge(opponents, on="game_id", how="left")
    # Remove the row where the player was matched with themselves
    df = df.loc[df["profile_id"] != df["opponent_id"]]
    return df

map_subsets = ['', 'arabia', 'arena', 'nomad', 'megarandom', 'hideout', 'water', 'open', 'closed', 'hybrid', 'chaos',
               'semiopen', 'nomadic']

def load_aoeinsights_csv(filename):
    print("Loading CSV")
    df = pd.read_csv(f"pantry/{filename}.csv")


    df = df.dropna(subset=['profile', 'relic_id', 'won', 'started', 'finished', 'map'])
    print("Renaming")
    df = df.rename(columns={"player_rating": "old_elo", "match_id": "game_id", "relic_id": "profile_id"})
    print("Adding a few stuff")
    df['winner'] = df['won'] > 0.5
    df["old_elo"] = df.groupby("profile_id")["old_elo"].ffill()
    df["old_elo"] = df.groupby("profile_id")["old_elo"].bfill()
    df['new_elo'] = df['old_elo'] + df['player_rating_change']
    df['started'] = pd.to_datetime(df['started'])
    cutoff = df['started'].max() - pd.Timedelta(days=30)
    df = df[df["started"] >= cutoff]
    df['finished'] = pd.to_datetime(df['finished'])
    df['duration'] = df['finished'] - df['started']
    print("Sorting CSV")
    df = df.sort_values(['started', "game_id", "profile_id"], kind="stable")
    print("Renaming maps to snake_case")
    df['map'] = df['map'].str.lower()
    df['map'] = df['map'].str.replace(pat={" ": "_"})
    print("Removing unnecessary columns")
    df = df.drop(columns=["player_rating_change", "won", "finished"])
    return df.reset_index()

def main():
    df = load_aoeinsights_csv("matches_60d_september")
    filename = 'aoe2insights_september'

    for prefix in map_subsets:
        if prefix != '':
            subset = get_map_subset(df, prefix)
        else:
            long_short(df)
            df = add_opponents(df)
            df = calc_elo_stats(df)
            df = find_alt_accounts(df)
            subset = df.copy()

        subset = add_game_counts(subset, n_days=[7, 14, 28], prefix=prefix)
        subset = add_streak(subset, "loss", count_n_streaks=[5, 7, 10], prefix=prefix)
        subset = add_streak(subset, "win", count_n_streaks=[5, 7, 10], prefix=prefix)
        subset = add_streak(subset, "short_loss", count_n_streaks=[3, 5], prefix=prefix)
        subset = add_streak(subset, "mid_long_win", count_n_streaks=[3, 5], prefix=prefix)
        subset = add_streak(subset, "long_win", count_n_streaks=[3, 5], prefix=prefix)

        if prefix == "":
            subset = add_activity(subset)

        subset = add_short_loss_duration_variance(subset, prefix=prefix)
        subset = calculate_duration_elos(subset, k_factor=32, prefix=prefix)

        subset = subset.drop(columns=subset.columns.intersection(df.columns))
        subset = subset.drop(columns=subset.columns[subset.columns.str.startswith("_")])
        df = df.merge(subset, left_index=True, right_index=True, how="left")
        # save(df, filename)

    print("Dropping all matches except for the last one")
    # df = df.ffill()
    df = df.drop_duplicates("profile_id", keep="last")
    save(df, filename)
    print("done")


if __name__ == "__main__":
    main()
