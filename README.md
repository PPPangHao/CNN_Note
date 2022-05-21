# CNN_Note
机器学习-从0开始构建自己的深度学习网络

博客地址：[机器学习-从0开始构建自己的深度学习网络](https://blog.csdn.net/qq_52222102/article/details/124890717?spm=1001.2014.3001.5502)

#前言
本文仅作为经验分享以及学习记录，如有问题，可以在评论区和我讨论。

理论知识暂且不讲，待有时间了我就会慢慢分享理论知识，目前就整点干活，直接上代码，怎么从零开始构建自己的神经网络。

#第一步 软件安装——anaconda，pycharm
正所谓工欲善其事必先利其器，既然要构建深度学习网络，那么我们就要安装我们的“工具”,这里我选择的是anaconda，和pycharm。具体的安装方法，请参考：

1.  anaconda安装方法

2.  pycharm安装方法

#第二步 工具安装
多数神经网络代码，是.ipynb格式文件，所以我选择的运行工具是，anaconda里面的，Jupyterlab\Jupyter Notebook
这两个一样，没差什么功能，就是Jupyterlab美观一些。

安装方法：
![image](https://img-blog.csdnimg.cn/835c91fdc11b4229a8712d172991f2b1.png)


 切换自己的环境，然后点击Jupyterlab\Jupyter Notebook的install。等待之后就好了。

关于如何更改Jupyterlab\Jupyter Notebook的起始文档位置，切换Jupyterlab\Jupyter Notebook启动浏览器
添加快捷方式等。都可以在网上查到，有时间我也会整理一下，发布出来。

 启动之后浏览器会跳转到这个页面，这个就是Jupyterlab:
![image](https://img-blog.csdnimg.cn/1fa369c41844448e97196098c67566c8.png)

#第三步 Jupyterlab的使用
这一部分简单介绍一下，Jupyterlab的使用方法。

##1.新建文件：点击加号，选择notebook即可。
![image](https://img-blog.csdnimg.cn/0740c58436724bd59b967584c55e76e1.png)



## 2.简单功能：

上面红框依次是：保存、增加cell、剪切、复制、粘贴、运行、暂停、重置。

右侧红框依次是：复制当前cell并粘贴到本cell下方、cell上移、cell下移、上方加cell、下方加cell、删除cell。
![image](https://img-blog.csdnimg.cn/3961c0c568d14e9dad34498b7119033a.png)



##  3.快捷键：

 Ctrl+Enter：运行此cell，不跳转到下一个cell
 Shi+Enter：运行此cell，跳转到下一个cell
选中cell前面的 [ ]:

X 删除选中的cell
M ，然后运行cell，将当前cell变成文本（个人理解，有误望指出）
# 第四步 正式开始
这里我们直接开始，直接上代码，通过代码，一方面有助于我梳理本次学习思路，二是我觉得这样更直接明了一些，毕竟动手才有趣。

##1. 文件展示
![image](https://img-blog.csdnimg.cn/5dc394e3852047b9a5eb5c99fba7d8d8.png)
![image](https://img-blog.csdnimg.cn/0715b546eeab42019793c14cc924f4ba.png)
![image](https://img-blog.csdnimg.cn/4f455dc2d56742929d83f5cf87a24629.png)





 每个cats、dogs文件夹下都是 猫or狗 图片

##2. 代码（一）——图片读取
```python
import os
# import zipfile
dog_dir = os.path.join('F:/tmp/cat_dog/training/dogs')
cat_dir = os.path.join('F:/tmp/cat_dog/training/cats')

print('total training dog images:', len(os.listdir(dog_dir)))
print('total training cat images:', len(os.listdir(cat_dir)))    # 计算路径下有多少文件

dog_files = os.listdir(dog_dir)            # 获取路径下所有文件
print(dog_files[:10])                        # 列出前十个文件名

cat_files = os.listdir(cat_dir)
print(cat_files[:10])
```
cell运行结果：
```
total training dog images: 8000
total training cat images: 8000
['0.jpg', '1.jpg', '10.jpg', '100.jpg', '1000.jpg', '1001.jpg', '1002.jpg', '1004.jpg', '1005.jpg', '1006.jpg']
['0.jpg', '1.jpg', '10.jpg', '100.jpg', '1000.jpg', '1001.jpg', '1002.jpg', '1003.jpg', '1004.jpg', '1005.jpg']
 ```
可以看到当前目录下共8000个文件，与结果符合。
![image](https://img-blog.csdnimg.cn/22d1ff001b9d4ad5b98da0c7b4446e24.png)

## 3. 代码（二）——图片输出
```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2            # 输出几张图片

next_dog = [os.path.join(dog_dir, fname)
                for fname in dog_files[pic_index-2:pic_index]]
next_cat = [os.path.join(cat_dir, fname)
                for fname in cat_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_dog+next_cat):            #输出图片
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
  ```
cell运行结果： 猫狗分别两张
![image](https://img-blog.csdnimg.cn/65e3dbe0c72c48e48be5ac52e34f391e.png)
![image](https://img-blog.csdnimg.cn/675bd069abe0469abba2f0fe5fc85340.png)





##  4. 代码（三）——数据预处理以及网络搭建
```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "./rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()                  # 神经网络结构可视化

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            # loss 损失函数：交叉熵         # optimizer 学习步长            # metrics 性能：准确度
history = model.fit_generator(train_generator, epochs=5, validation_data = validation_generator, verbose = 1)
                                # 学习步长             #                # verbose 记录日志
# model.save("catdog.h5")
   # 模型保存为 catdog.h5
```
 cell运行结果： 可以看到训练样本准确度和测试样本准确的分别为70%和76%（可以就该epoch，来增加训练伦数，以提高准确度）
网络结构：
![image](https://img-blog.csdnimg.cn/9f189ff5794845588f5cb1eaba7b1acd.png)



 训练过程：
```
Epoch 1/5
499/500 [============================>.] - ETA: 0s - loss: 0.7014 - acc: 0.5471
D:\PPPanghao\anaconda3\envs\py37\lib\site-packages\PIL\TiffImagePlugin.py:793: UserWarning: Truncated File Read
  warnings.warn(str(msg))
500/500 [==============================] - 300s 600ms/step - loss: 0.7014 - acc: 0.5473 - val_loss: 0.6831 - val_acc: 0.5045
Epoch 2/5
500/500 [==============================] - 221s 441ms/step - loss: 0.6510 - acc: 0.6269 - val_loss: 0.6043 - val_acc: 0.6790
Epoch 3/5
500/500 [==============================] - 222s 444ms/step - loss: 0.6168 - acc: 0.6694 - val_loss: 0.5558 - val_acc: 0.7080
Epoch 4/5
500/500 [==============================] - 223s 446ms/step - loss: 0.5964 - acc: 0.6859 - val_loss: 0.5875 - val_acc: 0.6630
Epoch 5/5
500/500 [==============================] - 222s 443ms/step - loss: 0.5751 - acc: 0.7024 - val_loss: 0.4868 - val_acc: 0.7660
```

##  5. 代码（四）——验证分类结果
先将图片预处理：

```python
from PIL import Image

#  待处理图片路径
image = Image.open("./cd/cat.jpg")
#  resize图片大小，入口参数为一个tuple，新的图片的大小
img_size = image.resize((150,150))
image = img_size
#  处理图片后存储路径，以及存储格式
img_size.save('./cd/catt.jpg', 'jpeg')

```
测试图片：

![image](https://img-blog.csdnimg.cn/c8c7ec549a6f4da9857161d2c8f4486c.jpeg)


```python
import numpy as np
from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

image = mpimg.imread("./cd/catt.jpg")
# print(image.shape)
image = image.reshape(1,150,150,3)/255
# image=image/255
c = model.predict(image,batch_size=1)
print(c)

```
cell运行结果：可以看到是猫的概率为70%，狗的概率为30%，所以结果为猫
```
  [[0.69656205 0.30343798]]
```
# 总结
至此本博客，从0开始搭建神经网络就结束了，有什么问题欢迎和我讨论。
