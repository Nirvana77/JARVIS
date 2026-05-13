import webbrowser
import urllib.parse
import libs.voice as voice

def run(query):
	voice.speak(f'Playing {query} on YouTube...')
	webbrowser.open(f'https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}')
