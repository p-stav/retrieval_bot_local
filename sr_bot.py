import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# sentiment scoring
# using the Google Cloud Language API -- from what I can tell, it is one of the best sentiment analyzers in the market
# NOTE: You have to run a few things in the client first to get this running.

#REPLACE the path above with the correct path -- this is on the google drive
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types




# open the file that we will use to match which response to reply with
f=open('/Users/paulstavropoulos/Desktop/response_dict.txt','r')
raw=f.read()
raw=raw.lower()

# tokenize to words and sentences
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# open the actual file that has the actual responses to reply with
# we will use an index (idx) to match up between respone_dict and responses
responses = open('/Users/paulstavropoulos/Desktop/responses.txt','r')
raw_responses = responses.read()
response_array = raw_responses.split('\n')


# responder -- finds nearest match with corpus
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)

    # perform tf-idf here
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    print vals
    # grab the pertinent index with the highest non-1 result for tf-idf value
    # this reveals the index of the closest matching text
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry, I can't understand you! I'm routing this to a ScaleRep director to answer!"
        return robo_response
    else:
        # use the index we found from sent_tokens and the response doc to grab
        # the right reply from the response array
        robo_response = robo_response+response_array[idx]
        return robo_response

def response_sentiment(user_response):
    document = types.Document(content=user_response,type=enums.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    return 'Sentiment: {}, {}\n\n'.format(sentiment.score, sentiment.magnitude)

# lemmatize functions
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# greeting inputs
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
  for word in sentence.split():
      if word.lower() in GREETING_INPUTS:
          return random.choice(GREETING_RESPONSES)




# actually kick off the program
flag=True
print("SCALEREP: Hello. This is a demo of ScaleRep. I will answer your queries. If you want to exit, type 'Bye'")
# initiate instance of Google client
client = language.LanguageServiceClient()

while(flag==True):
    user_response = raw_input()
    user_response=user_response.lower()

    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("SCALEREP: You are welcome...")
        else:
            if(greeting(user_response)!=None):
                print("SCALEREP: "+greeting(user_response))
            else:
                print("SCALEREP: ")
                print(response(user_response))
                print(response_sentiment(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("SCALEREP: Bye! take care..")
