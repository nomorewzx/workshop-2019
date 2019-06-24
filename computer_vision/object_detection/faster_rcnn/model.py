from keras.layers import Input
from keras.models import Model

from object_detection.config import Config
from object_detection.faster_rcnn.model_components import nn_base, rpn_layer, classifier_layer
from object_detection.losses import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
from keras.optimizers import Adam, SGD, RMSprop


def build_faster_rcnn_model(config: Config, class_num):
    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)  # 9
    rpn = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(shared_layers, roi_input, config.num_rois, nb_classes=class_num)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    faster_rcnn_model = Model([img_input, roi_input], rpn[:2] + classifier)

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[class_loss_cls, class_loss_regr(class_num - 1)],
                             metrics={'dense_class_{}'.format(class_num): 'accuracy'})
    faster_rcnn_model.compile(optimizer='sgd', loss='mae')

    return model_rpn, model_classifier, faster_rcnn_model
