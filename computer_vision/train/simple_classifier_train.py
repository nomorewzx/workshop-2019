import os

from keras.preprocessing.image import ImageDataGenerator

import settings
from classifier.classifier_models import build_simple_classifier_model

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

TRAIN_DATA_DIR = os.path.join(settings.DATA_DIR, 'dog_cat', 'train')

VALIDATION_DATA_DIR = os.path.join(settings.DATA_DIR, 'dog_cat', 'validation')

WEIGHTS_PATH = os.path.join(settings.MODEL_WEIGHTS_DIR, 'simple_classifier.h5')


def train_simple_classifier():
    simple_classifier_model = build_simple_classifier_model()

    simple_classifier_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    simple_classifier_model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

    if not os.path.exists(settings.MODEL_WEIGHTS_DIR):
        os.makedirs(settings.MODEL_WEIGHTS_DIR)

    simple_classifier_model.save_weights(WEIGHTS_PATH)  # always save your weights after training or during training


if __name__ == '__main__':
    train_simple_classifier()
