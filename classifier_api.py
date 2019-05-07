import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from flask import Flask, request, abort

import settings
from classifier.simple_classifier import build_simple_classifier_model

app = Flask(__name__)

ALLOWED_FILE_EXTENSIONS = ['jpeg', 'jpg', 'png']

simple_classifier_model = build_simple_classifier_model()

simple_classifier_model.load_weights(os.path.join(settings.MODEL_WEIGHTS_DIR, 'simple_classifier.h5'))

graph = tf.get_default_graph()


@app.route('/dog_cat_classifier', methods=['post'])
def classify_dog_cat():
    if 'file' not in request.files:
        return abort(400, 'can not find file in request body')

    img_file = request.files['file']

    if not _is_file_allowed(img_file.filename):
        return abort(400, 'wrong file format')

    img: JpegImageFile = Image.open(img_file)
    img = img.convert(mode='RGB')
    resized_img = img.resize(size=(150, 150))

    to_classify_img = np.array(resized_img)

    with graph.as_default():
        pred_result = simple_classifier_model.predict(np.array([to_classify_img]), batch_size=1)

    dog_prob = float(pred_result[0][0])

    return json.dumps({'pred_prob': [1-dog_prob, dog_prob],
                       'pred_label': 'dog' if pred_result[0][0] > 0.5 else 'cat'})


def _is_file_allowed(file_name):
    file_extension = file_name.split('.')[-1]

    return file_extension in ALLOWED_FILE_EXTENSIONS