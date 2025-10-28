from data_sources.BallDontLie import BallDontLie
from data_sources.OddsAPI import OddsAPI
import time

#bdl = BallDontLie('e443567c-ddc6-41cb-8b66-7d55f09f4e90')
#print(bdl.get_all_teams())
#time.sleep(15)
#print(bdl.get_team_by_id(27))
#time.sleep(15)
#print(bdl.get_all_players())

oa = OddsAPI("57d81d16fc6c36b35f5d8bb5893e1d3d")
odds = oa.get_nfl_odds()
for odd in odds:
	print(f"{ odd['away_team'] } @ { odd['home_team'] } :: { odd['commence_time'] }")