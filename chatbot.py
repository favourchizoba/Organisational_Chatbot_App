import nltk
import random
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


data =pd.read_csv('Samsung Dialog.txt',sep = ':',header = None)

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0]== 'Sales Agent']

sales = sales[1].reset_index(drop = True)
cust = cust[1].reset_index(drop = True)


new_data = pd.DataFrame()
new_data['Question'] = cust
new_data['Answer'] = sales


# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)
new_data['tokenized Questions'] = new_data['Question'].apply(preprocess_text)


xtrain = new_data['tokenized Questions'].to_list()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)


#-----------------------------------STREAMLIT IMPLEMENTATION---------------------------------------


st.header('Project Background Information', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ.")

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>ORGANISATIONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Frances</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


user_hist = []
reply_hist = []

robot_image, space1,space2, chats = st.columns(4)
with robot_image:
     robot_image.image('pngwing.com (21).png', width = 400)

with chats:
    user_message = chats.text_input('Hello there you can ask your questions:')

    def responser(text):
        user_input__preprocess =preprocess_text(text)
        vect_user_input =tfidf_vectorizer.transform([user_input__preprocess])
        similarity_score = cosine_similarity(vect_user_input,corpus)
        argument_maximum = similarity_score.argmax()
        return (new_data['Answer'].iloc[argument_maximum])

        
    bot_greeting = ['Hello user,i am a creation of Frances.......Ask your question',
                'Hey human what do you want',
                'Welcome Pls what do you want',
            'How can i help you?',
            'Hi user how can i be of help' ]

    bot_farewell = ['Thanks for your usage.......',
                'bye see you soon ',
            'Take care',
                'Bye... See you later', ]

    human_greeting = ['hi','hello','hey','hello there', 'welcome']
    human_exists = ['Thanks','bye','quit','exist','close']

    import random
    random_greetings = random.choice(bot_greeting)
    random_farewell = random.choice(bot_farewell)

    if user_message.lower() in human_exists:
        chats.write(f"\nchatbot: {random_farewell}!")
        user_hist.append(user_message)
        reply_hist.append(random_farewell)

    elif user_message.lower() in human_greeting:
        chats.write(f"\nchatbot: {random_greetings}!")
        user_hist.append(user_message)
        reply_hist.append(random_greetings)

    elif user_message == '':
        chats.write('')

    else:
        response = responser(user_message)
        chats.write(f"\nchatbot:{response}!")
        user_hist.append(user_message)
        reply_hist.append(response)


# Clearing Chat History 
def clearHistory():
    with open('history.txt', 'w') as file:
        pass  

    with open('reply.txt', 'w') as file:
        pass



#to save the histroy of the user texts
import csv
with open('history.txt','a') as file:
    for item in user_hist:
        file.write(str(item) + '\n')


        #to save histroy of bot reply
with open('reply.txt','a') as file:
    for item in reply_hist:
        file.write(str(item) + '\n')



#import the file to display it in the frontend
with open('history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open ('reply.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({'User Input': data1, 'Bot Reply' : data2})



#history = pd.Series(data)
st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)
#st.sidebar.write(data2)


if st.button('Clear Chat History'):
    clearHistory()

# primarycolor = '#FF4B4B'
# backgroundcolor = '#FB9AD1'
# secondarybackgroundcolor = '#F6F5F2'
# textcolor = '#292E1A'
# frontfamily = 'Sans Serif'


