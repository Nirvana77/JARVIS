import libs.voice as voice
import wikipedia

def run(query):
	voice.speak(f'Searching for {query} on Wikipedia...')
	results = wikipedia.summary(query, sentences=2)
	voice.speak('According to Wikipedia...')
	voice.speak(results)