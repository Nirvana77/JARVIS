

import importlib
from dotenv import load_dotenv
import libs.voice as voice
import libs.openai_helper as openai
import libs.brain as brain
import libs.training as traing
import json
import threading
import time

import os
import speech_recognition as sr

language = "en"

intents = json.loads(open('intents.json').read())

standby = True
will_run = True

def takeCommand(pause_threshold=1):
	global language
	r = sr.Recognizer()
	r.pause_threshold = pause_threshold
	with sr.Microphone() as source:
		audio = r.listen(source)

	try:
		statement = r.recognize_google(audio, language=language)
		print(f'user said: {statement}\n')

	except Exception as e:
		print(e)
		return 'None'
	
	return statement

def runCommand(res, userIntent=None):
	tag = res.get('tag')
	response = res.get('response')
	action = res.get('action')
	global standby
	
	if action == 'start':
		if standby == True:
			voice.speak(response)
		else:
			voice.speak('I am already awake.')
		standby = False
		return
	elif action == 'exit':
		if standby == True:
			shutdown()
		else:
			standby = True
			voice.speak('I will enter standby mode´.')
		return
	elif action == 'shutdown':
		shutdown()
		return
	
	if standby == True:
		return
	
	if action == 'none':
		if response != "":
			voice.speak(response)
		return
	elif action == 'train':
		voice.speak('Training with the new data')
		threading.Thread(target=traing.traing_model).start()
		return
	elif response != "":
		voice.speak(response)
	
	
	if action == 'ask_chat_gpt':
		ask_chat_gpt()
	else:
		if action:
			try:
				# Import the module dynamically based on the action
				module = importlib.import_module(f'actions.{action}')
				# Call the run function inside the module with userIntent as argument
				module.run(userIntent)
			except ImportError:
				voice.speak('Sorry, I don\'t know how to do that yet.')
			except AttributeError:
				voice.speak('The action module does not have a run function.')
		else:
			# Handle the case where no action is specified
			voice.speak('Sorry, I don\'t know how to do that yet.')

def ask_chat_gpt():
	voice.speak('What is your question?')
	question = takeCommand()
	voice.speak('Searching...')
	answer = openai.ask_gpt3(question)
	voice.speak(answer)

def shutdown():
	global will_run
	voice.speak('Goodbye!')
	will_run = False
	my_thread.join()
	exit()

def my_thread_function():
	global will_run
	global lastTraining

	lastModif = os.path.getmtime('intents.json')

	while will_run:
		if lastModif != os.path.getmtime('intents.json'):
			lastModif = os.path.getmtime('intents.json')
			traing.traing_model()

		time.sleep(1)

my_thread = threading.Thread(target=my_thread_function)

def run():
	# start an other thread
	my_thread.start()

	while True:
		query = takeCommand().lower()

		if 'none' == query:
			continue
		else:
			ints = brain.predict_class(query)
			res = brain.get_response(ints, intents)
			runCommand(res)


def init():
	load_dotenv()
	global language
	language = os.getenv('language')
	global lastTraining

	# Check the creation date of the model
	# If it is older than 7 days, train the model again
	# If the model does not exist, train the model
	if not traing.model_exists():
		print('Training model...')
		traing.traing_model()
		print('Done')
	else:
		cration_date = os.path.getctime('JARVIS_model.keras')
		lastTraining = time.time() - cration_date
		if lastTraining > 604800: # 7 days in seconds
			print('Training model...')
			traing.traing_model()
			print('Done')

	voice.init()
	brain.init()
	
	#traing_model()

if __name__ == '__main__':
	init()