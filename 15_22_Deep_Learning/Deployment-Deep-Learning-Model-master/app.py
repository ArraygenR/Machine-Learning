from __future__ import division, print_function
# coding=utf-8
import sys , keras
from keras import backend as K
import os
import glob , cv2
import re
import numpy as np
import skimage.transform
# Keras

from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Load your trained model


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path):

    model = keras.models.load_model('models/path_to_my_model.h5')
    model._make_predict_function()
    X = []
    img_file = cv2.imread(img_path)
    if img_file is not None:
        img_file = skimage.transform.resize(img_file,(60, 80, 3))
        img_arr = np.asarray(img_file)
        X.append(img_arr)
    X = np.asarray(X)
    
    y_pred = model.predict(X)
    Y_pred_classes1 = np.argmax(y_pred,axis=1) 
    l1 = list(Y_pred_classes1)
    map_characters =  {0:'Class 1',1:'Class 2'}
    K.clear_session()
    return map_characters[l1[0]]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

