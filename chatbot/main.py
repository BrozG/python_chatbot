import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pickle
import webbrowser

# Load necessary files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(ints, intents_json):
    if len(ints) == 0:
        return "I haven't understood", None
    tag = classes[ints[0][0]]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result, tag
    return "I haven't understood", None

def perform_google_search(query):
    google_url = f"https://www.google.com/search?q={query}"
    webbrowser.open(google_url)

print("Chatbot is ready to talk! Type 'quit' to stop.")

while True:
    message = input("you>")
    if message.lower() == "quit":
        print("chatbot>Goodbye!")
        break
    ints = predict_class(message, model)
    res, tag = get_response(ints, intents)
    if tag is None:
        print("chatbot>I haven't understood")
    elif tag == "google_search":
        print("chatbot>Performing a Google Search.........")
        perform_google_search(message)
    else:
        print(f"chatbot>{res}")
    