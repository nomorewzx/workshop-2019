"""
Dataset from kaggle: https://www.kaggle.com/c/dogs-vs-cats/data
Blog post: https://blog.keras.io/building-powerful-image-classification-model_weights-using-very-little-data.html
"""
from keras import applications
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

# Assignment: 尝试改进Simple Classifier Model, 使得在20个Epoch后, 分类准确率达到87%.
    # 最开始的Simple Classifier模型, 有如下的结构:
    # =====================Block 1=======================
    # Conv2D, filters=32, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # =====================Block 2=======================
    # Conv2D, filters=32, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # ====================Block 3========================
    # Conv2D, filters=64, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # ====================Block 4=======================
    # Flatten
    # Dense 全连接层, 输出维度units = 64
    # Relu 激活函数
    # Dropout
    # Dense 全连接层, 输出维度units = 1
    # Sigmoid 激活函数 

    # Hint: 如果希望能够提升模型的分类准确率, 第一种应该尝试的方式, 是将模型变得更Deep. 使得模型有更多的层, 同时每一层的参数增多.
    # 另外需要尝试更改训练过程中的参数, 如learning rate等.
    # 可以尝试实现如下的网络结构: 
    # =====================Block 1=======================
    # Conv2D, filters=32, kernel_size = (3,3)
    # Conv2D, filters=32, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # =====================Block 2=======================
    # Conv2D, filters=64, kernel_size = (3,3)
    # Conv2D, filters=64, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # ====================Block 3========================
    # Conv2D, filters=128, kernel_size = (3,3)
    # Conv2D, filters=128, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # ====================Block 4========================
    # Conv2D, filters=256, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # ====================Block 5========================
    # Conv2D, filters=512, kernel_size = (3,3)
    # Relu 激活函数
    # Maxpooling, pool_size = (2,2)
    # ====================Block 6=======================
    # Flatten
    # Dense 全连接层, 输出维度units = 1024
    # Relu 激活函数
    # Dense 全连接层, 输出维度units = 1024
    # Relu 激活函数
    # Dropout
    # Dense 全连接层, 输出维度units = 1
    # Sigmoid 激活函数
def build_simple_classifier_model(img_height=150, img_width=150):
    input_shape = (img_width, img_height, 3)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 指定网络模型, 要求实现

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


