import numpy as np

from object_detection.config import Config
from object_detection.faster_rcnn.neural_network import build_faster_rcnn


def evaluate_faster_rcnn(config: Config, img: np.ndarray):
    model_rpn, model_classifier, model_all = build_faster_rcnn(config=config, class_num=3)

    model_rpn.load_weights(config.model_path, by_name=True)
    model_classifier.load_weights(config.model_path, by_name=True)

    # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
    predicted_rpn = model_rpn.predict(np.array([img]), batch_size=1)

    return predicted_rpn