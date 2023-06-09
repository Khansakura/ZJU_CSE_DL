#!/usr/bin/env python
# coding: utf-8

# # 作业第二周  MLP模型练习fashion_mnist分类操作
仿照课件完成fashion_mnist的分类操作：
1.练习keras内置数据集fashion_mnist的读取与操作。
Fashion-MNIST克隆了MNIST的所有外在特征： 60000张训练图像和对应Label； 10000张测试图像和对应Label； 10个类别；
(train_images, train_labels),(test_images,test_labels)= tf.keras.datasets.fashion_mnist.load_data()

2.设计一个简单多层感知机网络，训练fashion_mnist的分类操作。
(打印loss变化曲线曲线，显示测试集最后的预测准确率、混淆矩阵、典型误判图像等)
# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,512)


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# 查看数据集形状
print("train_data:",train_images.shape) # (60000, 28, 28)
print("train_label:",train_labels.shape) # (60000,)
print("test_data:",test_images.shape) # (10000, 28, 28)
print("test_label",test_labels.shape) # (10000,)

#查看数据集具体图像
def plot_images_labels_prediction(images,labels, prediction,idx,num=10):
    fig = plt.gcf
    if num>25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(images[idx],cmap='binary')
        title = "label="+str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediciton[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()


# In[3]:


plot_images_labels_prediction(train_images,train_labels,[],0,10)


# In[4]:


#归一化图像数据，到[0,1]
train_images = train_images/255.0
test_images = test_images/255.0


# In[5]:


#建立一个keras网络模型
# 设计多层感知机网络
model = keras.Sequential([
    # Flatten层，将输入层的二维图像数据展平为一维向量
    keras.layers.Flatten(input_shape=(28, 28)),
    # Dense层，添加一个隐藏层，使用ReLU激活函数，有128个神经元
    keras.layers.Dense(128, activation='relu'),
    # 添加一个输出层，使用softmax激活函数，有10个神经元，对应10个类别
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())


# In[6]:


# 使用训练集训练模型，指定训练轮数为25，batch_size为32
history = model.fit(train_images, train_labels, validation_split=0.1, epochs=20, batch_size=32)

# 使用测试集评估模型的性能
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


# In[7]:


import itertools
'''
# 绘制损失变化曲线
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
'''
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()
    
show_train_history(history,'accuracy','val_accuracy')
show_train_history(history,'loss','val_loss')
# 根据label定义类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 获取模型在测试集上的预测结果
predictions = model.predict(test_images)

# 绘制混淆矩阵函数
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    给定一个混淆矩阵，绘制一个可视化的混淆矩阵。
    参数：
        cm: 一个方形的混淆矩阵（numpy array）
        target_names: 混淆矩阵中每个类别的名称（list或tuple）
        title: 图表的标题（string）
        cmap: 颜色映射（colormap），默认为None表示使用matplotlib默认颜色映射。
        normalize: 是否将混淆矩阵归一化（boolean）
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# 使用sklearn库来计算混淆矩阵，并调用上面定义的函数来绘制混淆矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, np.argmax(predictions,axis=1))
plot_confusion_matrix(cm,
                      class_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=False)

# 定义一个函数来绘制典型误判图像
def plot_misclassified_images(images,
                              labels,
                              predictions,
                              class_names,
                              max_num=25):
    """
    给定一组图像、标签、预测结果和类别名称，绘制一些典型误判图像。
    参数：
        images: 图像数据（numpy array）
        labels: 真实标签（numpy array）
        predictions: 预测结果（numpy array）
        class_names: 类别名称（list或tuple）
        max_num: 最多显示多少张误判图像（int），默认为25。
    """
    
    # 找出所有预测错误的索引
    error_indices = np.where(labels != predictions)[0]
    
    # 如果错误数量超过最大数量，则随机选择一些错误索引
    if len(error_indices) > max_num:
        error_indices = np.random.choice(error_indices, size=max_num)
    
    # 计算每行每列显示多少张图像
    num_cols = 5
    num_rows = int(np.ceil(len(error_indices) / num_cols))
    
    # 创建一个画布，并按行列顺序显示误判图像及其真实标签和预测标签
    plt.figure(figsize=(15, num_rows * 3))
    
    for i in range(len(error_indices)):
        index = error_indices[i]
        
        image = images[index]
        label = labels[index]
        prediction = predictions[index]
        
        true_name = class_names[label]
        pred_name = class_names[prediction]
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        plt.title(f'True: {true_name}')


# In[ ]:





# In[ ]:




