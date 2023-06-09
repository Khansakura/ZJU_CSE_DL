#!/usr/bin/env python
# coding: utf-8

# # 作业第三周  CNN网络练习
1.针对fashion_mnist数据集，设计一个CNN网络，训练fashion_mnist的分类操作，将准确率提高到92%以上！
请尝试通过优化网络层数与超参数、正则化等措施，提高准确率。
请与第二周MLP模型相比较，在文末总结说明不同模型在收敛速度与分类准确率的表现。
注意：输入数据需要有Channel维度，需要reshape为(28,28,1)
# In[29]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,512)


# In[30]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

#获取数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#查看数据集
print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
print(test_images.shape) # (10000, 28, 28)
print(test_labels.shape) # (10000,)
print(np.unique(train_labels)) # [0 1 2 3 4 5 6 7 8 9]


# In[31]:


# 通过搜索得到1-9代表的标签意义，定义相应的标签名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 可视化部分数据集
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[32]:


#处理到0-1之间，并增加Channel维度
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)
#处理后数据形式
print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
print(test_images.shape) # (10000, 28, 28)
print(test_labels.shape) # (10000,)


# In[36]:


#建立CNN模型并查看模型结构
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10)
])
model.summary()


# In[37]:


#模型编译
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[38]:


history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))


# In[39]:


# 可视化训练过程中的损失和准确率变化
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[46]:


# 在测试集上评估模型的性能
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# 可以看出:
# CNN模型在收敛速度和分类准确率上都优于MLP模型.因为CNN模型可以利用卷积层提取图像的局部特征，并通过池化层降低参数量和过拟合风险,同时也可以通过dropout层减少过拟合风险。
# 而MLP模型则将图像展平为一维向量，忽略了图像的空间信息，并且参数量较多，容易导致过拟合，因此训练集表现很好但最终验证集准确率较低。
