class Player:

    def __init__(self, name, player_id, elo, match_stats):
        self.name = name
        self.elo = elo
        self.id = player_id

        self.match_stats = match_stats

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return self.id == other.id

    def _get_link(self, ):
        # self.id already contains backslashes
        return f"https://www.aoe2insights.com{self.id}stats/3/"

    def _short_game_ratio(self):
        total_games = 0
        for value in self.match_stats.values():
            total_games += value['matches']
        if total_games == 0:
            return 0
        return self.match_stats['<5 mins']['matches'] / total_games

    def _short_game_win_ratio(self):
        if self.match_stats['<5 mins']['matches'] == 0:
            return 0
        return self.match_stats['<5 mins']['wins'] / self.match_stats['<5 mins']['matches']

    def _long_game_win_ratio(self):
        long_games = self.match_stats['25 to 40 mins']['matches'] + self.match_stats['>40 mins']['matches']
        long_wins = self.match_stats['25 to 40 mins']['wins'] + self.match_stats['>40 mins']['wins']
        if long_games == 0:
            return 0
        return long_wins / long_games

    def is_potential_smurf(self, min_short_game_ratio=0.1, max_short_game_wins_ratio=0.3, min_long_game_wins_ratio=0.6):
        if self._short_game_ratio() < min_short_game_ratio:
            return False
        if self._short_game_win_ratio() > max_short_game_wins_ratio:
            return False
        return self._long_game_win_ratio() > min_long_game_wins_ratio