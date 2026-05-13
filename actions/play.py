import webbrowser
import libs.voice as voice

def run(query):
	voice.speak(f'Playing {query} on YouTube...')
	webbrowser.open(f'https://www.youtube.com/results?search_query={query}')