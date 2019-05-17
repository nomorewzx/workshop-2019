import os

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import settings
from classifier.classifier_models import build_vgg16_backbone_model


def train_classifier(img_width, img_height,
                     train_data_dir, validation_data_dir, model_save_path,
                     train_samples_count, validation_samples_count, epochs = 20, batch_size = 32,
                     is_fine_tune_vgg = False):

    classifier_model = build_vgg16_backbone_model(img_width=img_width, img_height=img_height)

    print('new model has {} layers'.format(len(classifier_model.layers)))

    if is_fine_tune_vgg:
        classifier_model.load_weights(model_save_path)
        print('load weights')
        for layer in classifier_model.layers[:15]:
            layer.trainable = False
    else:
        for layer in classifier_model.layers[:18]:
            layer.trainable = False

    classifier_model.summary()

    classifier_model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    classifier_model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples_count // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples_count // batch_size)

    classifier_model.save_weights(model_save_path)


if __name__ == '__main__':

    img_width, img_height = 150, 150

    train_dir = os.path.join(settings.DATA_DIR, 'train')
    validation_dir = os.path.join(settings.DATA_DIR, 'validation')
    train_samples_count = 2000
    validation_samples_count = 800

    epochs = 20
    batch_size = 32

    if not os.path.exists(settings.MODEL_WEIGHTS_DIR):
        os.makedirs(settings.MODEL_WEIGHTS_DIR)

    model_save_path = os.path.join(settings.MODEL_WEIGHTS_DIR, 'vgg16_backbone_classifier.h5')

    train_classifier(img_width=img_width, img_height=img_height,
                     train_data_dir=train_dir, validation_data_dir=validation_dir,
                     validation_samples_count=validation_samples_count, train_samples_count=train_samples_count,
                     epochs=epochs, batch_size=batch_size, model_save_path=model_save_path, is_fine_tune_vgg=False)
