#!/usr/bin/env python
# coding: utf-8

# # 附加练习作业（选做，折平时分）  seq2seq翻译改进练习
为了提升效果，本示例程序将英文原按字符分割改进为按单词分割。
请在理解seq2seq代码基础上，完成代码改造工作，提高性能。要求：
1）文本分割处理：注意简单空格分割会导致标点符号没有分离，标点符号有意义应作为单独token。非标点符号删除（isn't的引号如何处理可斟酌）
2）模型结构：改为GRU单元，多层堆砌，参数优化：神经元数量 等优化  
3）训练过程保存最佳模型用于预测  
# # 导入库

# In[163]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,1024)


# In[156]:


import tensorflow as tf
import re
from tensorflow import keras
from tensorflow.keras.layers import Input,LSTM,Dense,GRU,Embedding
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np


# In[115]:


N_UNITS = 256
BATCH_SIZE = 64
EPOCH = 50
NUM_SAMPLES = 21000
embedding_dim = 256


# # 数据处理

# In[116]:


data_path = 'data/cmn.txt'


# In[117]:


df = pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,:2,]


# In[118]:


df.tail()


# In[119]:


df.columns=['inputs','targets']
df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')
df.head()


# In[120]:


# 导入所需的库
import re
import string

# 预处理英文句子
def preprocess_sentence(w):
    """处理英文句子
        1,符号左右加空格;
        2,将非字母和非标点符号的字符替换为空格;
        3,空格去重;
    """
    # 1.符号左右加空格
    # 使用正则表达式，将标点符号前后添加空格
    w = re.sub(r"([?.!,])", r" \1 ", w)
    # 2.将非字母和非标点符号的字符替换为空格
    # 使用正则表达式，将非字母和非标点符号的字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)
    # 3.空格去重
    # 使用正则表达式，将多个空格替换为一个空格
    w = re.sub(r"\s+", " ", w)
    # 返回
    return w

input_texts=[ preprocess_sentence(t) for t in input_texts]


# In[121]:


# 4,加载数据;使用tokenizer做词嵌入;分割数据为train,vaild,test数据集
def max_length(tensor):
    """找到数据集中padding的最大值"""
    return max(len(t) for t in tensor)

def tokenize(lang, china = False):
    """将数据集做padding"""
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level = china)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# ## 向量化

# In[122]:


target_tensor, targ_lang_tokenizer = tokenize(target_texts, china=True)
max_length_targ= max_length(target_tensor)
max_length_targ


# In[123]:


#PADDING字符占用一个码，需要+1
vocab_inp_size = len(inp_lang_tokenizer.word_index)+1
vocab_inp_size


# In[124]:


vocab_tar_size = len(targ_lang_tokenizer.word_index)+1
vocab_tar_size


# In[125]:


max_length_inp =  max_length(input_tensor)
max_length_inp


# In[126]:


# 6,测试数据转化结果 
def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print ("%d ----> %s" % (t, lang.index_word[t]))
            
print("Input Language; index to word mapping")
convert(inp_lang_tokenizer, input_tensor[-1])


# In[127]:


print("Target Language; index to word mapping")
convert(targ_lang_tokenizer, target_tensor[-1])


# In[128]:


from sklearn.model_selection import train_test_split
# 5,拆分训练集和验证集
# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=1000)

# 打印数据集长度 - Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


# In[129]:


#decoder的label tensor
output_tensor_train=np.zeros_like(target_tensor_train)
output_tensor_train[:,:-1]=target_tensor_train[:,1:]

output_tensor_val=np.zeros_like(target_tensor_val)
output_tensor_val[:,:-1]=target_tensor_val[:,1:]


# In[130]:


input_tensor.shape, target_tensor.shape


# # 创建模型（作业要点）

# In[131]:


'''
def create_model(n_input,n_output,n_units):
    #训练阶段
    #encoder
    encoder_input = Input(shape = (None, ))
    embeddin = keras.layers.Embedding(n_input, embedding_dim)
    #encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    #n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    _,encoder_h,encoder_c = encoder(embeddin(encoder_input))
    encoder_state = [encoder_h,encoder_c]
    #保留下来encoder的末状态作为decoder的初始状态
    
    #decoder
    decoder_input = Input(shape = (None, ))
    embeddout = keras.layers.Embedding(n_output, embedding_dim)
    #decoder的输入维度为中文字符数
    decoder = LSTM(n_units,return_sequences=True, return_state=True)
    #训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _, _ = decoder(embeddout(decoder_input),initial_state=encoder_state)
    #在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_dense = Dense(n_output,activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    #输出序列经过全连接层得到结果
    
    #生成的训练模型
    model = Model([encoder_input,decoder_input],decoder_output)
    #第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出
    
    #推理阶段，用于预测过程
    #推断模型—encoder
    encoder_infer = Model(encoder_input,encoder_state)
    
    #推断模型-decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))    
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]#上个时刻的状态h,c   
    
    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(embeddout(decoder_input),initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]#当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)
    
    return model, encoder_infer, decoder_infer
'''


# In[165]:


# 定义create_model函数，接受n_input, n_output, n_units作为参数
def create_model(n_input, n_output, n_units):
    # 训练阶段
    # encoder
    encoder_input = Input(shape=(None,))
    embedding = keras.layers.Embedding(n_input, embedding_dim)
    # encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = GRU(n_units, return_state=True)
    # n_units为GRU单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h
    _, encoder_h = encoder(embedding(encoder_input))
    encoder_state = [encoder_h]
    # 保留下来encoder的末状态作为decoder的初始状态

    # decoder
    decoder_input = Input(shape=(None,))
    embedding = keras.layers.Embedding(n_output, embedding_dim)
    # decoder的输入维度为中文字符数
    decoder = GRU(n_units, return_sequences=True, return_state=True)
    # 训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _ = decoder(embedding(decoder_input), initial_state=encoder_state)
    # 在训练阶段只需要用到decoder的输出序列，不需要用最终状态h
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    # 输出序列经过全连接层得到结果

    # 生成的训练模型
    model = Model([encoder_input, decoder_input], decoder_output)
    # 第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出

    # 推理阶段，用于预测过程
    # 推断模型-encoder
    encoder_infer = Model(encoder_input, encoder_state)

    # 推断模型-decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h]#上个时刻的状态h

    decoder_infer_output, decoder_infer_state_h = decoder(embedding(decoder_input), initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h]#当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)
    
    return model, encoder_infer, decoder_infer


# In[166]:


model_train, encoder_infer, decoder_infer = create_model(vocab_inp_size, vocab_tar_size, N_UNITS)


# In[167]:


plot_model(model=encoder_infer,show_shapes=True)


# In[168]:


#input_tensor, target_tensor

model_train.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


# # 模型训练

# In[141]:


#创建保存目录及文件夹
import os
#定义路径并进行查询操作，防止重复创建
save_dir = "save/3200105710"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# In[142]:


#训练模型，用callback保存
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5), # 提前停止训练如果验证损失不再下降
    tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "model.h5"), save_best_only=True) # 保存最佳模型权重
]


# In[143]:


model_train.fit([input_tensor_train, target_tensor_train],output_tensor_train,batch_size=BATCH_SIZE,epochs=EPOCH,
                validation_data=([input_tensor_val, target_tensor_val],output_tensor_val),
               callbacks=callbacks)


# # 预测序列

# In[169]:


# 加载最佳模型
model_train.load_weights('save/3200105710/model.h5')


# In[173]:


def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    state = encoder_inference.predict(source)
    #第一个字符'\t',为起始标志
    predict_seq = np.zeros((1,1))
    predict_seq[0,0]=1   ##target_dict['\t']

    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h隐状态，以及上一次的预测字符predict_seq
        yhat,h= decoder_inference.predict([predict_seq]+[state])
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        char = targ_lang_tokenizer.index_word[char_index]
        output += char + ' '
        state = [h]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1))
        predict_seq[0,0]=char_index
        if char == '\n':#预测到了终止符则停下来
            break
    return output


# In[174]:


def convertinp(tensor):
    s=''
    for t in tensor:
        if t != 0:
            s += inp_lang_tokenizer.index_word[t]+' '
    return s


# In[175]:


for i in range(20):
    test = input_tensor_val[i:i+1,:]#i:i+1保持数组是三维
    out = predict_chinese(test,encoder_infer,decoder_infer,max_length_targ,vocab_tar_size)

    print(convertinp(input_tensor_val[i]))
    print(out)


# In[ ]:





# In[ ]:




