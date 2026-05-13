import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

words = None
classes = None
model = None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json) -> dict:
    if not intents_list:
        return {'response': "I'm not sure I understand. Could you rephrase?", 'tag': 'unknown', 'action': 'none'}

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return {'response': random.choice(intent['responses']), 'tag': tag, 'action': intent['action']}

    return {'response': "I'm not sure I understand. Could you rephrase?", 'tag': 'unknown', 'action': 'none'}

def init():
    global words, classes, model

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))

    model = load_model('JARVIS_model.keras')

if __name__ == '__main__':
    print('JARVIS is ready to chat!')
    intents = json.loads(open('intents.json').read())
    while True:
        message = input('')
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res) 