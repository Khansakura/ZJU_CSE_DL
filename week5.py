#!/usr/bin/env python
# coding: utf-8

# # 作业第5周   循环神经网络练习

# 1.仿照课件关于IMDb数据集的分类训练，在课件示例基础上改用GRU、优化网络层数与其它参数，尽力提升分类准确率。<BR>
# 要求：使用callback方法保存最佳模型；训练完成后将保存在文件的模型读入，用于评估与预测测试集样本。<BR>
# 请每个人在"save"目录下建立以自己学号命名的子目录，然后在该子目录下保存文件

# In[2]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,1024)


# In[3]:


#创建保存目录及文件夹
import os
#定义路径并进行查询操作，防止重复创建
save_dir = "save/3200105710"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# In[4]:


import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


# In[5]:


max_features = 25000
maxlen = 380
(x_train, y_train), (x_val, y_val) = tensorflow.keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train),"Training sequences")
print(len(x_val),"Validation sequences")


# In[6]:


x_train[0][:10]


# In[7]:


len(x_train[0])


# In[8]:


y_train[0:10]


# In[9]:


word_index = tensorflow.keras.datasets.imdb.get_word_index(path='imdb_word_index.json')


# In[10]:


word_index


# In[11]:


index_to_word = {v:k for k, v in word_index.items()}


# In[12]:


index_to_word


# In[13]:


' '.join([index_to_word[x] for x in x_train[0]])


# In[14]:


x_train_pad = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val_pad = sequence.pad_sequences(x_val, maxlen=maxlen)


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN, LSTM
from tensorflow.keras.optimizers import SGD, Adam


# In[16]:


#LSTM建立循环神经网络
'''
model = Sequential()

model.add(Embedding(output_dim=64,
                   input_dim=max_features,
                   input_length=maxlen))

model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=128))
model.add(Dropout(0.5))

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
'''


# In[17]:


#SimpleRNN建立循环神经网络
model = Sequential()

model.add(Embedding(output_dim=64, input_dim=max_features, input_length=maxlen))

model.add(SimpleRNN(units=32, return_sequences=True)) 
model.add(Dropout(0.4)) 

model.add(Flatten())

model.add(Dense(units=64,activation='relu')) 
model.add(Dropout(0.4))

model.add(Dense(units=1,activation='sigmoid'))


# In[18]:


model.summary()


# In[19]:


opt = Adam(lr=1e-4, decay=1e-3)
model.compile(loss='binary_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])


# In[40]:


#训练模型，用callback保存
callbacks = [
    tensorflow.keras.callbacks.EarlyStopping(patience=3), # 提前停止训练如果验证损失不再下降
    tensorflow.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "3200105710_model.h5"), save_best_only=True) # 保存最佳模型权重
]


# In[41]:


train_history = model.fit(x_train_pad, y_train, batch_size=32,
                         epochs=10, verbose=1,
                          validation_split=0.2, callbacks=callbacks)


# In[42]:


import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train HIstory')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[43]:


show_train_history(train_history,'accuracy','val_accuracy')


# In[44]:


show_train_history(train_history,'loss','val_loss')


# In[20]:


model.load_weights("save/3200105710/3200105710_model.h5")
scores = model.evaluate(x_val_pad, y_val, verbose=1)
scores[1]


# In[ ]:




