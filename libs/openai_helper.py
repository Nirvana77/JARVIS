import openai
from dotenv import load_dotenv
import os


def ask_gpt3(prompt):
	response = openai.Completion.create(
		engine="davinci",
		prompt=prompt,
		temperature=0.7,
		max_tokens=100,
		top_p=1.0,
		frequency_penalty=0.0,
		presence_penalty=0.0,
		stop=None
	)
	return response.choices[0].text.strip()

def init():
	load_dotenv()
	openai.api_key = os.getenv('api_key')