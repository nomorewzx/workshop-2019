## Workshop: Computer Vision and Deep Learning ##

#### 1. 安装Python virtualenv ####
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


#### 2. 下载所需数据 ####
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
