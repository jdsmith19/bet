from dotenv import load_dotenv
import os
import requests
import xmltodict
import json
import time

load_dotenv()

class BillSimmonsPodcast:
    def __init__(self, week, season, podcast):
        self.whisper_server_url = os.getenv('WHISPER_SERVER_URL')
        self.week = week
        self.season = season
        self.rss_url = "https://feeds.megaphone.fm/the-bill-simmons-podcast"
        self.rss_obj = self.__load_rss()
        self.current_job_id = None
        self.current_episode_name = None
        self.current_episode_duration = None
        self.debug = False
        self.job_status = "pending"
    
    def __load_rss(self):
        r = requests.get(self.rss_url)
        data = xmltodict.parse(r.text)
        return data
    
    def __get_guess_the_lines(self):
        for item in self.rss_obj['rss']['channel']['item']:
            is_season = str(self.season) in item['pubDate']
            is_gtl = "guess the lines" in item['title'].lower()
            is_current_week = (f"week {self.week}" in item['title'].lower() or f"week {self.week}" in item['description'].lower())
            if (is_season and is_gtl and is_current_week):
                self.current_episode_duration = item['itunes:duration']
                episode_details = {
                    'title': item['title'],
                    'media_url': item['enclosure']['@url'],
                    'save_filename': f'guess_the_lines_{self.season}_week_{self.week}.mp3'
                }
                return episode_details
        return {
            'error': 'No Guess the Lines episode found for this week.',
            'last_ten_episodes': self.rss_obj['rss']['channel']['item'][:10]
        }
    
    def check_job_status(self, job_id):
        """Check the status of the current transcription job"""
        if not job_id:
            return None
        
        status_check_url = f"{self.whisper_server_url}/transcribe/{job_id}"
        r = requests.get(status_check_url)
        data = json.loads(r.text)
        
        self.job_status = data['status']
        
        if self.debug:
            print(f"Job status: {data['status']}")
        
        return data
    
    def transcribe_episode(self, episode_type):
        if episode_type == 'guess_the_lines':
            self.current_episode_name = "Guess the Lines with Cousin Sal"
            episode_details = self.__get_guess_the_lines()

        r = requests.post(
            self.whisper_server_url + "/transcribe",
            json = episode_details
        )
        
        data = r.text
        data = json.loads(r.text)

        job_id = data['job_id']
        return job_id

    
    def chunk_transcription(self, segments, chunk_size=50, overlap=10):
        chunks = []
        i = 0
        while i < len(segments):
            chunk_segment = segments[i:i + chunk_size]
            chunk_text = ' '.join(s['text'] for s in chunk_segment)
            chunks.append(chunk_text)
            i += (chunk_size - overlap)
        return chunks
