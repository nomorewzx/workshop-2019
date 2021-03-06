## Workshop: Computer Vision and Deep Learning ##

### 安装Python virtualenv ###
推荐使用Pipenv管理virtualenv与依赖包.

1. 确保本地已经安装 Python 3.6 以及 pip.

2. 安装Pipenv. 执行命令: `pip install --user pipenv`

3. 在`computer_vision`目录下安装Python virtualenv以及相应依赖. 
        
        cd workshop/computer_vision
        pipenv install
    
4. 激活Python virtualenv. 执行 `pipenv shell`

5. 验证依赖包已经安装成功. 
    
    - 执行 `python` 进入 Python 解释器, 此时应该显示:
        
            Python 3.6.4 (v3.6.4:d48ecebad5, Dec 18 2017, 21:07:28)
            [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
            Type "help", "copyright", "credits" or "license" for more information.
            >>>
            
    - 执行 `import keras`, 此时应该显示:
        
            >>> import keras
            Using TensorFlow backend.
            >>> 

可以参考 [Pipenv & 虚拟环境¶
](https://pythonguidecn.readthedocs.io/zh/latest/dev/virtualenvs.html) 帮助安装Pipenv.

### Part1 Classifier ###
1. 从Google Drive下载 [Dog Cat数据集](https://drive.google.com/file/d/1ZZGnziQLhmoiz5Uz5GG5qKClfA9pM9GV/view)
2. 将数据集解压并放在 `./data` 目录中, 确保目录结构如下
    
        computer_vision
            |--data
                |-- train
                    |-- cat
                        |-- cat.y1.jpg
                        |-- ...
                    |-- dog
                        |-- dog.x1.jpg
                        |-- ...
                |-- validation
                    |--cat
                        |--cat.y2.jpg
                        |--...
                    |--dog
                        |--dog.x2.jpg
                        |--...
3. 安装完成keras后, 下载[VGG16预训练模型](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5),并将模型保存在 `~/.keras/models/` 文件夹中.
4. Part1 Slide: [Part1: Classification & Keras](https://docs.google.com/presentation/d/1IR1m9fHaD5oh2otVGiKzCAVJj-ry-sUUjSwFUuiYRh8/edit)

### Part2 Object Detection: Faster-RCNN ###
0. 根据 `安装Python virtualenv` 中的方法安装新的依赖包.

1. 从腾讯云盘下载 [Object Detection数据集](https://share.weiyun.com/5hZeBzk), 该数据集来自[Google OpenImage](https://storage.googleapis.com/openimages/web/index.html) 数据集, 并且仅提取了其中的 `Person`, `Car` 和 `Mobile phone`三个class, 每个class各取1000张图片.

2. 将数据集解压缩并放在 `./data/object_detection` 目录中, 确保目录结构如下


        computer_version
            |--data
                |-- object_detection
                    |--train
                        |-- xxx.jpg
                    |--test
                        |-- xxx.jpg
                    |--annotation.txt
                    |--test_annotation.txt

3. 从腾讯云盘下载 [Faster RCNN的Trained Weights](https://share.weiyun.com/5eqK3jE), 并存放在`model_weight/faster_rcnn/`中. 目录结构如下:
    
        computer_vision
            |--model_weights
                |--faster_rcnn
                    |-- model_frcnn_vgg.h5

4. 查看训练集样例图片与标注. 执行以下命令来查看样例图片与相应的标注.

        python -m unittest object_detection.data_loader_test
   
   样例图片与标注如下所示:
   ![样例图片与标注](sample/annotation_sample.png)

5. [Part 1 录像: Faster RCNN中的RPN与训练数据生成](https://drive.google.com/file/d/18NXUiE4QmdemUgNqupXIWuO4SS3yaQUz/view?usp=sharing)
 
 > Faster RCNN的实现主要参考以下资料:  
 >1. [Faster R-CNN (object detection) implemented by Keras for custom data from Google’s Open Images Dataset V4](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a)  
 >2. [Faster R-CNN: Down the rabbit hole of modern object detection](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/)
 >3. [Faster_RCNN_for_Open_Images_Dataset_Keras](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras)

