import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = pickle.load(open('file.pkl', 'rb'))
vocabulary_to_load = pickle.load(open("cv.pkl", 'rb'))
loaded_vectorizer = CountVectorizer(ngram_range=(4, 4), vocabulary=vocabulary_to_load)

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

@app.route('/')
def home():
    return render_template('index.html', data ='')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
        
    Sequence = request.form['seq']
    words = getKmers(Sequence, size=6)
    seq = ' '.join(words)
    #print(seq)
    X1 = loaded_vectorizer.fit_transform([seq,]).toarray()
    prediction=model.predict(X1)
    
    return render_template('index.html', data = {'seq':Sequence,'op':'Class of Sequence is $ {}'.format(prediction)} )

if __name__ == "__main__":
    app.run(debug=True)
