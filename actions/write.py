import os
import libs.voice as voice
from libs.command_helper import takeCommand

NOTES_FILE = os.path.expanduser('~/jarvis_notes.txt')

def run(query):
	voice.speak('What do you want me to write?')
	text = takeCommand()
	if text and text != 'None':
		with open(NOTES_FILE, 'a') as f:
			f.write(text + '\n')
		voice.speak('Done, I have written that down.')
	else:
		voice.speak('Sorry, I did not catch that.')
