from nba_api.stats.static import teams
import pandas as pd

nba_teams = pd.DataFrame(teams.get_teams())
print(nba_teams)