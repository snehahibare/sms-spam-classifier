import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# keep required columns
df = df[['v1','v2']]
df.columns = ['label','text']

# convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

# vectorization
tfidf = TfidfVectorizer(stop_words='english')

X = tfidf.fit_transform(df['text'])
y = df['label']

# train model
model = MultinomialNB()
model.fit(X,y)

# save files
pickle.dump(tfidf, open("vectorizer.pkl","wb"))
pickle.dump(model, open("model.pkl","wb"))

print("Model trained successfully")
