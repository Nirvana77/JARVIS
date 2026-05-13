import libs.voice as voice
import wikipedia

def run(query):
	voice.speak(f'Searching for {query} on Wikipedia...')
	try:
		results = wikipedia.summary(query, sentences=2)
		voice.speak('According to Wikipedia...')
		voice.speak(results)
	except wikipedia.exceptions.DisambiguationError as e:
		voice.speak(f'That topic is ambiguous. Did you mean {e.options[0]}?')
	except wikipedia.exceptions.PageError:
		voice.speak(f'Sorry, I could not find a Wikipedia page for {query}.')
	except Exception:
		voice.speak('Sorry, I had trouble searching Wikipedia.')
