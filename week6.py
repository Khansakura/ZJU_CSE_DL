#!/usr/bin/env python
# coding: utf-8

# # 作业第6周  歌词生成练习
1.仿照课件关于歌词生成例子，在课件示例基础上将LSTM网络改为GRU且多层堆砌，优化网络层数及其它参数，尽力提升效果。

# # 导入库

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
# GPU.show()
GPU.alloc(0,1024)


# In[2]:


import tensorflow as tf
from tensorflow import keras
import random
import zipfile
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN,LSTM


# In[3]:


with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')


# In[4]:


print(corpus_chars[:40])


# In[5]:


#使用set()函数将数据集中重复的字符删掉，然后放入列表中。
idx_to_char = list(set(corpus_chars))
len(idx_to_char)


# In[6]:


#将字符映射到索引
char_to_idx = {char:i for i, char in enumerate(idx_to_char)}


# In[7]:


vocab_size = len(char_to_idx)
vocab_size


# In[8]:


#将字符转化成索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
len(corpus_indices)


# In[9]:


sample = corpus_indices[1000:1020]
print('indices:', sample)
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))


# In[10]:


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
 
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# In[11]:


my_seq = list(range(30))
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
     print('X: ', X, '\nY:', Y, '\n')


# In[12]:


#LSTM实现的模型结构
'''
num_hiddens = 256
batch_size =160
num_steps = 35
model = Sequential()
model.add(keras.Input(batch_input_shape=(batch_size,num_steps))) 
model.add(Embedding(output_dim=256,
 input_dim=vocab_size, 
 input_length=num_steps))
model.add(LSTM(units=num_hiddens,
 return_sequences=True, 
 stateful=True))
model.add(Dense(units=vocab_size,activation='softmax' ))
model.summary()
'''


# In[61]:


#GRU实现的模型结构
'''
经过测试，使用两层SimpleRNN层(128隐藏层)，不使用Dropout的情况下，相同训练epoch情况下效果不如单层SimpleRNN层（128隐藏层）
尝试过单层无堆叠的SimpleRNN（256隐藏层），发现效果有所提升，推测128隐藏层神经元数量不够，难以完全拟合模型的变化，在此基础上添加堆叠
采用两层SimpleRNN（256层），添加Dropout防止过拟合，增加训练epoch，发现效果有所提升，一首歌曲总计约100-200词左右，因此
perplexity值大约表示下一个词有1.02个可以选，准确度较高

'''

num_hiddens = 256
batch_size =160
num_steps = 35

model = Sequential()

model.add(keras.Input(batch_input_shape=(batch_size,num_steps))) 

model.add(Embedding(output_dim=256,
 input_dim=vocab_size, 
 input_length=num_steps))

model.add(SimpleRNN(units=num_hiddens,
 return_sequences=True, 
 stateful=True))

model.add(Dropout(0.3))

model.add(SimpleRNN(units=num_hiddens,
 return_sequences=True, 
 stateful=True))

model.add(Dense(units=vocab_size,activation='softmax' ))

model.summary()


# In[62]:


from tensorflow.keras.utils import plot_model
plot_model(model=model,show_shapes=True)


# In[63]:


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[64]:


def predict_rnn_keras(prefix, num_chars):
    # 使用model的成员函数来初始化隐藏状态
    model.reset_states()
    output = [char_to_idx[prefix[0]]] #上一次输出
 
    for t in range(num_chars + len(prefix) - 1):
        X = (np.array([output[-1]]).repeat(batch_size)).reshape((batch_size, 1)) #timestep=1
        Y = model(X) # 前向计算
 
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]]) #引导前缀不使用预测结果
        else:
            output.append(sample(np.array(Y[0,0,:]))) #批量大小， 时间步数, 词典大小
            ##output.append(int(np.array(tf.argmax(Y,axis=-1))[0,0])) #批量大小， 时间步数, 词典大小
 
    return ''.join([idx_to_char[i] for i in output])


# In[65]:


predict_rnn_keras('分开', 10)
#因为模型参数为随机值，所以预测结果也是随机的。


# In[66]:


# 计算裁剪后的梯度示意代码，本例并不使用该函数！！！
def grad_clipping(grads,theta):
    norm = np.array([0])
    for i in range(len(grads)):
        norm+=tf.math.reduce_sum(grads[i] ** 2)
    norm = np.sqrt(norm).item()
 
    if norm <= theta:
        return grads
 
    new_gradient=[]
    for grad in grads:
        new_gradient.append(grad * theta / norm)
    return new_gradient


# In[67]:


opt=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=0.1)
model.compile(loss='sparse_categorical_crossentropy', 
    optimizer=opt, 
    metrics=['accuracy'])


# In[68]:


def train_and_predict_rnn_keras(num_epochs, batch_size, pred_period, pred_len, prefixes):
    for epoch in range(num_epochs):
        l_sum, n = 0.0, 0
        model.reset_states()
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps)
 
        for X, Y in data_iter:
            y_pred = model.train_on_batch(X,Y) #y_pred[0]为loss， y_pred[1]为accuracy
            loss=y_pred[0]
            l_sum += loss
            n += 1
            
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f' % (
                epoch + 1, math.exp(l_sum / n)))
            for prefix in prefixes:
                print('>>', predict_rnn_keras(prefix, pred_len))
                
num_epochs = 2500
train_and_predict_rnn_keras(num_epochs, batch_size, 100, 50, ['想要', '我们'])


# In[ ]:




