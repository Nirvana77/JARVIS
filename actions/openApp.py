
import libs.voice as voice
import webbrowser

def run(query):
	voice.speak(f'Opening {query}...')
	webbrowser.open(f'https://www.{query}.com')