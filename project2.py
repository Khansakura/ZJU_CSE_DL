#!/usr/bin/env python
# coding: utf-8

# # 期末综合练习（2）

# ### 对keras内置路透社文本数据集进行内容分类（要求模型至少包含有两要素：CNN 、 RNN、 注意力机制）
# 即：Conv + RNN 或RNN + 注意力机制 或 CNN + 注意力机制
# 路透社数据集：<BR>
# 路透社数据集包含许多短新闻及其对应的主题，由路透社在1986年发布。包括46个不同的主题，其中某些主题的样本更多，但是训练集中的每个主题都有至少10个样本。<BR>
# 与IMDB数据集一样，路透社数据集也内置到了Keras库中，并且已经经过了预处理。<BR>
# #### 提示：
# 由于文本较长，先用CNN卷积上采样到较短长度，再用RNN处理是一个避免梯度消失的方案。<BR>
#     (由于卷积核为一维，卷积核大小要相应增大到5或7，stride增加到3或5)。<BR>
# 引入注意力机制是另一种克服遗忘的方案。<BR>
# 采用pytorch框架的同学，也利用keras读取数据集内容后进行训练
# #### 要求：
# 利用callback将最佳模型保存到文件(注意：在"save"目录下建立以自己学号命名的子目录，然后在该子目录下保存文件)，
# 最后对最佳模型进行指标评估，展示混淆矩阵
# #### 数据读取方法：
# (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000, test_split=0.2)
# 
# #### 考核办法：
# 1）程序功能完成度<BR>
# 2）计算得到的准确率为指标，准确率达到0.7为及格成绩起点，0.8以上优秀<BR>
# score = model.evaluate(x_test, y_test)
# 

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,1024)


# In[5]:


# 导入所需的库
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix


# In[6]:


#读取数据集，限制词汇表大小为10000，测试集比例为0.2
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000, test_split=0.2)
# 查看数据集的形状和类别数量
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(np.unique(y_train).shape[0])


# In[8]:


# 对输出类别进行one-hot编码
num_classes = np.unique(y_train).shape[0]
print(num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[16]:


# 建立带有注意力机制的卷积层类
class Conv_Attention(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Conv_Attention, self).__init__()
        # 定义卷积层
        self.conv = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')
        # 定义注意力层
        self.attention = layers.Dense(1, activation='tanh')
        self.flatten = layers.Flatten()
        self.softmax = layers.Activation('softmax')
        self.repeat = layers.RepeatVector(filters)
        self.permute = layers.Permute([2, 1])
        # 定义乘法层
        self.multiply = layers.Multiply()
        # 定义求和层
        self.sum = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))
        # 保存自定义参数
        self.filters = filters
        self.kernel_size = kernel_size

    def call(self, inputs):
        # 使用卷积层对输入进行特征提取
        conv = self.conv(inputs)
        # 使用注意力机制计算每个特征的权重
        attention =  self.attention(conv)
        attention =  self.flatten(attention)
        attention =  self.softmax(attention)
        attention =  self.repeat(attention)
        attention =  self.permute(attention)
        # 将卷积层的输出和注意力权重相乘，得到加权平均的特征向量
        feature =  self.multiply([conv, attention])
        feature =  self.sum(feature)
        # 返回特征向量
        return feature

    def compute_output_shape(self, input_shape):
        # 返回输出的形状，即(batch_size, filters)
        return (input_shape[0], self.conv.filters)

    def get_config(self):
        # 返回包含类配置的字典
        config = super(Conv_Attention, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'conv': self.conv,
            'attention': self.attention,
            'flatten': self.flatten,
            'softmax': self.softmax,
            'repeat': self.repeat,
            'permute': self.permute,
            'multiply': self.multiply,
            'sum': self.sum
        })
        return config


# In[345]:


# 创建模型，使用CNN和双向LSTM的组合，同时带有注意力机制
model = keras.models.Sequential()
# 将整数序列转换为词嵌入
model.add(layers.Embedding(10000, 64, input_length=maxlen))
model.add(layers.Dropout(0.75))
# 使用卷积层对词嵌入进行特征提取，并使用注意力机制得到特征向量
model.add(Conv_Attention(64, 7))
# 使用双向LSTM层对特征向量进行编码，并使用注意力机制得到上下文向量
model.add(layers.Reshape((-1, 1), input_shape=(32,)))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001))))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.001))))
model.add(layers.Dropout(0.15))
# 使用全连接层进行分类
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))
# 查看模型的结构
model.summary()


# In[346]:


# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[347]:


import os
save_dir = "save/3200105710"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# In[348]:


# 定义回调函数，保存最佳模型到文件，且包含早停功能
callbacks = [
    keras.callbacks.ModelCheckpoint("save/3200105710/model.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    keras.callbacks.EarlyStopping(patience=7)
]


# In[349]:


# 训练模型
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=callbacks)


# In[350]:


import matplotlib.pyplot as plt

# 显示loss曲线和accuracy曲线
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[351]:


# 进行指标评估并显示结果
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# In[352]:


# 展示混淆矩阵
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)


# In[353]:


# 绘制混淆矩阵函数
import itertools
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          text=True):
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
    if text == True:
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
    plt.show()


# In[354]:


#定义类别名及混淆矩阵热力图表示
plot_confusion_matrix(cm,
                      target_names = None,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=False,
                      text=False)


# #### 总结说明
# 此处说明关于模型设计与模型训练（参数设置、训练和调优过程）的心得与总结

# 模型设计过程：
# 模型需要集合CNN与LSTM模型，同时引入注意力机制，采用简单的序列模型和已有的keras模型结构难以达到目标。因此选择建立自己的卷积模型类以引入注意力机制。
# 开始时本来设计采用函数以Model形式建立模型，但采用函数设计模型结构，不同层之间的数据结构的传输难以实现，采用（input）的实例化则会发生错误，搜索发现是可能是TensorFlow和keras混用产生的问题。但调整后依然存在，可能与编程实现的算法过程有问题有关，最终重新设计模型结构才得以成功。
# 模型设计的调整主要在训练调优过程中实现，包括但不限于添加LSTM层数量，添加Dropout层等实现过程。
# 训练调优过程：
# 初始的模型结果一直相对不佳，主要表现为：过拟合严重，同时存在训练后期验证集loss可能反而下降的现象。
# 由于路透社数据是已经预处理的数据，且难以像图像ImageGenerator一样进行数据增强，因此模型调优主要集中于优化模型结构方面。
# 查阅资料后发现，造成问题的主要原因是模型过于复杂，LSTM和CNN层的参数过多，因此对参数进行了调整，模型准确率有所上升，能达到0.6左右，但仍然存在震荡的问题，进一步调整batch_size，模型验证集loss震荡现象有所缓解，但过拟合现象依然很严重。
# 对模型进行进一步分析，在CNN层和LSTM层加入dropout，调整参数希望解决过拟合问题，一定程度上缓解了过拟合问题，但模型的准确率依然只能达到0.65左右，不能满足需求，在LSTM和Dense层加入了L1-L2正则化调整后也存在该问题。
# 对模型进一步分析发现，模型参数量最大的部分来自于第一层Embedding层的嵌入，因此应该在该层后添加Dropout防止过拟合能有效降低模型的参数量。在该层后添加Dropout函数后果然模型的过拟合问题得到进一步缓解，模型最终准确率能达到0.7以上，符合要求。

# In[ ]:




