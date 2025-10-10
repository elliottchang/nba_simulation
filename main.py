# imports
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
import numpy as np
import pandas as pd

# ====================================================================
# helper functions
# ====================================================================

#nba teams mapping
nba_teams = pd.DataFrame(teams.get_teams())

# box score function
BOXCOLS = ["PTS","REB","AST","FGA","FGM","3PA","3PM","FTA","FTM","STL","BLK","TO","PF","+/-"]

def init_boxscore_from_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    box = roster_df[["PLAYER_ID","PLAYER","POSITION","NUM"]].copy()
    for c in BOXCOLS:
        box[c] = 0
    return box.set_index("PLAYER_ID")

# minutes to float function
def _mins_to_float(v):
    # "MM:SS" -> minutes as float; also handles NaN or numeric
    if pd.isna(v): return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v)
    if ":" in s:
        m, sec = s.split(":")
        return float(m) + float(sec)/60.0
    return float(s)


# nba team id, name, abbr resolver
def team_resolver(id=None, abbr=None, name=None) -> dict:

    # check to make sure only one value is present
    if sum(x is not None for x in (id, abbr, name)) != 1:
        raise ValueError("Provide exactly one of team_id, abbr, or name.")
    
    # by id
    if id is not None:
        team_name = nba_teams[nba_teams['id'] == id]['nickname'].iloc[0] # returns the name of the team
        team_abbr = nba_teams[nba_teams['id'] == id]['abbreviation'].iloc[0] # returns abbreviation
        team_id = id
    
    # by name
    elif name is not None:
        team_id = nba_teams[nba_teams['nickname'] == name]['id'].iloc[0] # returns the id of the team
        team_abbr = nba_teams[nba_teams['id'] == team_id]['abbreviation'].iloc[0] # returns abbreviation
        team_name = name

    # by abbreviation
    else:
        team_id = nba_teams[nba_teams['abbreviation'] == abbr]['id'].iloc[0] # returns the name of the team
        team_name = nba_teams[nba_teams['id'] == team_id]['nickname'].iloc[0] # returns abbreviation
        team_abbr = abbr

    location = nba_teams[nba_teams['id'] == team_id]['city'].iloc[0]
    
    return {'team_id': team_id, 'team_abbr': team_abbr, 'team_name': team_name, 'location': location}
    

# ====================================================
# game simmulation classes

# team class
class Team:
    def __init__(self, season: str, team_id: int = None, team_abbr: str = None, team_name: str = None):
        team_info = team_resolver(team_id, team_abbr, team_name)

        self.team_id = int(team_info['team_id'])
        self.team_name = str(team_info['team_name'])
        self.team_abbr = str(team_info['team_abbr'])
        self.location = str(team_info['location'])
        self.season = str(season)
        
        endpoint = commonteamroster.CommonTeamRoster(team_id=self.team_id, season=self.season)
        self.roster_df = endpoint.get_data_frames()[0]

        self.starters = self.pick_starters(self.roster_df, season)

    def __str__(self) -> str:
        header = f"{self.season} {self.location} {self.team_name} Summary:"
        meta = f"{self.team_abbr} | Team ID: {self.team_id}"
        roster_preview = self.roster_df[['PLAYER_ID', 'PLAYER', 'POSITION', 'NUM']].to_string(index=False)
        return f"{header}\n\n{meta}\n\nRoster:\n{roster_preview}"
    
    def pick_starters(self, roster_df: pd.DataFrame, season: str) -> list:
        records = []
        for pid in roster_df["PLAYER_ID"].tolist():
            try:
                # reuse your Player class
                p = Player(player_id=int(pid), season=season)
                df = p.gamelog
                if df.empty:
                    continue
                starts = int(df.get("GS", pd.Series([0]*len(df))).fillna(0).astype(int).sum())
                avg_min = float(pd.Series(df.get("MIN", [])).apply(_mins_to_float).mean()) if "MIN" in df.columns else 0.0
                records.append((int(pid), (starts, avg_min)))
            except Exception:
                # player may have no logs; skip quietly
                pass

        # sort by (starts desc, avg_min desc) and take top 5
        records.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
        return records[:5]
    

class Player:
    def __init__(self, player_id: int, season: str):
        player_info = players.find_player_by_id(player_id)

        self.id = player_id
        self.name = player_info['full_name']
        self.gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season').get_data_frames()[0]
    

class GameEngine:
    def __init__(self, hometeam_id: int, awayteam_id: int, season: str):
        self.hometeam = Team(team_id = hometeam_id, season = season)
        self.awayteam = Team(team_id = awayteam_id, season=season)

        # game vars
        self.quarter = 1
        self.possession = 1
        self.quarter_duration = 12*60 #seconds

        #boxscores
        self.hometeam_boxscore = init_boxscore_from_roster(self.hometeam.roster_df)
        self.awayteam_boxscore = init_boxscore_from_roster(self.awayteam.roster_df)

        # initialize lineup as teams starters
        self.home_lineup = self.hometeam.starters
        self.away_lineup = self.awayteam.starters

    def step(self):
        self.possession += 1

    def subsitute(self):
        pass
