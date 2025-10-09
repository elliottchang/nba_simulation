# imports
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
import pandas as pd

#nba teams mapping
nba_teams = pd.DataFrame(teams.get_teams())

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

    def __str__(self) -> str:
        header = f"{self.season} {self.location} {self.team_name} Summary"
        meta   = f"{self.team_abbr} | Team ID: {self.team_id}"
        roster_preview = self.roster_df[['PLAYER', 'POSITION', 'NUM']].to_string(index=False)
        return f"{header}\n\n{meta}\n\nRoster:\n{roster_preview}"
    

class GameEngine:
    def __init__(self, hometeam_id: int, awayteam_id: int, season: str):
        hometeam = Team(team_id = hometeam_id, season = season)
        awayteam = Team(team_id = awayteam_id, season=season)

        self.quarter = 1
        self.possession = 1
        self.quarter_duration = 12*60 #seconds
        self.hometeam_boxscore = hometeam.roster_df
        self.hometeam_boxscore['pts', 'reb', 'ast', 'fga', 'fgm', '3pa', '3pm', 'fta', 'ftm', 'stl', 'blk', 'to', 'pf', 'pm'] = 0
        self.awayteam_boxscore['pts', 'reb', 'ast', 'fga', 'fgm', '3pa', '3pm', 'fta', 'ftm', 'stl', 'blk', 'to', 'pf', 'pm'] = 0

    def step(self):
        self.possession += 1