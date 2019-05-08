"""
Dataset from kaggle: https://www.kaggle.com/c/dogs-vs-cats/data
Blog post: https://blog.keras.io/building-powerful-image-classification-model_weights-using-very-little-data.html
"""
from keras import applications
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


def build_simple_classifier_model(img_height=150, img_width=150):
    input_shape = (img_width, img_height, 3)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def build_vgg16_backbone_model(img_height=150, img_width=150):
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning

    vgg16_backbone_classifier_model = Sequential()
    for layer in base_model.layers:
        vgg16_backbone_classifier_model.add(layer)

    # add the model on top of the convolutional base
    vgg16_backbone_classifier_model.add(top_model)

    return vgg16_backbone_classifier_model


