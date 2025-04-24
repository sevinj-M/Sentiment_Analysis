import pandas as pd
import numpy as np
from utils_nb import process_tweet
from sklearn.model_selection import train_test_split



'''
1. load data -> take artist(s), text, positivenes
2. create column 'label' -> 0/1
3. tokenize lyrics
4. train the model
5. lyrics -> vectors
6. X ve Y ler
7. train + test datasetleri
8. 
'''

#1
all_data = pd.read_csv("/Users/sevinjmahmudova/Desktop/archive/spotify_dataset.csv")


#2
data = all_data[['Artist(s)', 'song', 'text', 'Positiveness']].copy()

data['Labels'] = 0 
data.loc[data['Positiveness'] >= 50, 'Labels'] = 1 

positives = data.loc[data['Labels'] == 1][:225000]
negatives = data.loc[data['Labels'] == 0][:225000]

data = pd.concat([positives, negatives], ignore_index=True) 

#3
data['tokenized_lyrics'] = data['text'].apply(process_tweet)

#4 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


X = data['text']  
y = data['Labels']  

#5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#6
tfidf_vectorizer = TfidfVectorizer(max_features=1500)  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#7
model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

#8
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

#Ending: To predict wether your lyrics are positive or negative, use this function:

def predict_l(lyrics):
  new_guess = tfidf_vectorizer.transform(lyrics)
  new = model.predict(new_guess)

print(predict_l(["Friends told her she was better off at the bottom of a river Than in a bed with him He said 'Until you try both, you won't know what you like better Why don't we go for a swim?' Well, friends told her this and friends told her that But friends don't choose what echoes in your head When she got bored with all the idle chit-and-chat Kept thinking 'bout what he said I'll swim even when the water's cold That's the one thing that I know Even when the water's cold She remembers it fondly, she doesn't remember it all But what she does, she sees clearly She lost his number, and he never called But all she really lost was an earring The other's in a box with others she has lost I wonder if she still hears me I'll swim even when the water's cold That's the one thing that I know Even when the water's cold If you believe in love You know that sometimes it isn't Do you believe in love? Then save the bullshit questions Sometimes it is and sometimes it isn't Sometimes it's just how the light hits their eyes Do you believe in love?"]))
