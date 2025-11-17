from data_sources.BillSimmonsPodcast import BillSimmonsPodcast

bsp = BillSimmonsPodcast(week = 11, season = 2025)
transcription = bsp.transcribe_episode(episode_type = 'guess_the_lines')
#print(len(transcription['result']))
chunks = bsp.chunk_transcription(transcription['result'])
#print(chunks)
#print(len(chunks))