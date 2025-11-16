from data_sources.BillSimmonsPodcast import BillSimmonsPodcast

bsp = BillSimmonsPodcast(week = 11, season = 2025)
gtl = bsp.transcribe_episode(episode_type = 'guess_the_lines')
print(gtl)