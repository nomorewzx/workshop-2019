import os

import settings
from classifier.classifier_models import build_simple_classifier_model, build_vgg16_backbone_model


def load_classifier_model(is_load_vgg_backbone_classifier_model):
    if not is_load_vgg_backbone_classifier_model:
        classifier_model = build_simple_classifier_model()

        classifier_model.load_weights(os.path.join(settings.MODEL_WEIGHTS_DIR, 'simple_classifier.h5'))

        print('load simple classifier model')
    else:
        classifier_model = build_vgg16_backbone_model()

        classifier_model.load_weights(os.path.join(settings.MODEL_WEIGHTS_DIR, 'vgg16_backbone_classifier.h5'))

        print('load vgg16 backbone classifier model')

    return classifier_model
