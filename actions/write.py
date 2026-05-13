import libs.voice as voice
from libs.command_helper import takeCommand

def run():
	voice.speak('What do you want me to write?')
	text = takeCommand()
	voice.speak('Writing...')
	voice.write(text)