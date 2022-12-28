# Importing libraries
import numpy as np
import nltk 
import string 
import random

f = open('.\BERT_texte.txt', encoding="utf8")
raw_doc = f.read()
raw_doc = raw_doc.lower() #Converting entire text to lowercase
nltk.download('punkt') #Using the Punkt tokenizer
nltk.download('wordnet') #Using the wordnet dictionary
nltk.download('omw-1.4')
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
sentence_tokens = nltk.sent_tokenize(raw_doc) #Converting into sentences tokenizer
word_tokens = nltk.word_tokenize(raw_doc) #Converting into words tokenizer

# Performing Text-PreProcessing Steps:
from nltk.corpus.reader.tagged import word_tokenize
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Define Greeting fucntions:
greet_inputs = ('hello', 'hi', 'whassup', 'how are you?', 'yo', 'hey', 'hey bro!')
greet_responses = ('hi', 'hey', 'hey there!', 'hey bro!')
def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_responses)

# Response Generation by the Bot:
from sklearn.feature_extraction.text import TfidfVectorizer #Converts a collection of raw documents into a matrix of TF-IDF features
from sklearn.metrics.pairwise import cosine_similarity #Measures the similarity between two vectors

def response(user_response):
  robot1_response = ''
  TfidfVec = TfidfVectorizer(tokenizer = LemNormalize)
  tfidf = TfidfVec.fit_transform(sentence_tokens)
  vals = cosine_similarity(tfidf[-1], tfidf)
  idx = vals.argsort()[0][-2] #Finfing the most similar 
  flat = vals.flatten()
  flat.sort()
  req_tfidf = flat[-2]
  if (req_tfidf == 0):
    robot1_response = robot1_response + 'Im sorry. Unable to understand you!'
    return robot1_response
  else:
    robot1_response = robot1_response + sentence_tokens[idx]
    return robot1_response

# Defining the ChatFlow:

flag = True
print('Hello! I am the learning bot. Start typing your text after greeting to talk to me. For ending convo type bye!')
while(flag == True):
  user_response = input()
  user_response = user_response.lower()
  if (user_response != 'bye'):
    if (user_response == 'thank you' or user_response == 'thanks'):
      flag = False
      print('Bot: You are welcome..')
    else:
      if (greet(user_response) != None):
        print('Bot'+greet(user_response))
      else: 
        sentence_tokens.append(user_response)
        word_tokens = word_tokens + nltk.word_tokenize(user_response)
        final_words = list(set(word_tokens))
        print('Bot:', end = '')
        print(response(user_response))
        sentence_tokens.remove(user_response)
  else:
      flag = False
      print('Bot: Goodbye!')