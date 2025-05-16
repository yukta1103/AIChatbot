import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load data
with open("intents.json") as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Preprocess and prepare data
corpus = []
labels = []
all_tags = []

for intent in intents["intents"]:
    tag = intent["tag"]
    all_tags.append(tag)
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        corpus.append(" ".join(words))
        labels.append(tag)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Model
model = LogisticRegression()
model.fit(X, labels)

# Chat function
def chatbot_response(user_input):
    tokens = nltk.word_tokenize(user_input)
    words = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    X_input = vectorizer.transform([" ".join(words)])
    tag = model.predict(X_input)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Chat loop
print("Bot: Hello! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)

