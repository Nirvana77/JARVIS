from openai import OpenAI
from dotenv import load_dotenv
import os

client = None

def ask_gpt3(prompt):
	response = client.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": prompt}],
		max_tokens=150,
	)
	return response.choices[0].message.content.strip()

def init():
	global client
	load_dotenv()
	client = OpenAI(api_key=os.getenv('api_key'))
