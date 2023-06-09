#!/usr/bin/env python
# coding: utf-8

# # 作业第四周  CNN（2）网络练习

# 1.仿照课件关于deepdream的程序，在data目录选择一张背景图片(zju1.jpg或zju2.jpg或zju3.jpg或zju4.jpg或者用代码下载一张网络图片保存在save/目录)，
# 选取一个ImageNet预训练网络，通过选择以及组合不同的特征层，训练出一张自己满意的deepdream图片。<BR>
# 
# 

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,1024)


# In[2]:


#观察四张图片
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
img1 = mpimg.imread('data/zju1.jpg') # 读取图片1
img2 = mpimg.imread('data/zju2.jpg') # 读取图片2
img3 = mpimg.imread('data/zju3.jpg') # 读取图片3
img4 = mpimg.imread('data/zju4.jpg') # 读取图片4

#imshow()对图像进行处理，画出图像，show()进行图像显示
plt.subplot(221)
plt.imshow(img1)
plt.title('图像1')
plt.axis('off')
#子图2
plt.subplot(222)
plt.imshow(img2)
plt.title('图像2')
plt.axis('off')
#子图3
plt.subplot(223)
plt.imshow(img3)
plt.title('图像3')
plt.axis('off')
#子图4
plt.subplot(224)
plt.imshow(img4)
plt.title('图像4')
plt.axis('off')


# In[37]:


#导入相关库及基本图像处理函数
import tensorflow as tf 
import numpy as np
import IPython.display as display
import PIL.Image
from tensorflow.keras.preprocessing import image

#图像标准化
def normalize_image(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)
#图像显示
def show_image(img):
    display.display(PIL.Image.fromarray(np.array(img)))
#图像保存
def save_image(img,file_name):
    PIL.Image.fromarray(np.array(img)).save(file_name)
#图像读取
def read_image(file_name, max_dim=None):
    img=PIL.Image.open(file_name)
    if max_dim:
        img.thumbnail((max_dim,max_dim))
    return np.array(img)


# In[4]:


#选择zju2进行操作，读取并显示
img_file = 'data/zju2.jpg'
orig_img = read_image(img_file, 500)
show_image(orig_img)


# In[5]:


#加载预训练模型
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
base_model.summary()


# In[8]:


#最大限度激活这些层的指定层
layer_names='conv2d_38'
layers = base_model.get_layer(layer_names).output


# In[9]:


#创建特征提取模型
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
dream_model.summary()


# In[40]:


#定义损失函数
def calc_loss(img, model):
    channels = [20,80,149]
    img = tf.expand_dims(img, axis=0)
    layer_activations = model(img)
    losses = []
    for cn in channels:
        act = layer_activations[:,:,:,cn]
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)


# In[41]:


def render_deepdream(model, img, steps=100, step_size=0.01, verbose=1):
    for n in tf.range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img, model)
        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
        if(verbose == 1):
            if ((n+1)%10 == 0):
                print("Step{}/{}, loss{}".format(n+1,steps,loss))
    return img


# In[42]:


img = tf.keras.applications.inception_v3.preprocess_input(orig_img)
img = tf.convert_to_tensor(img)


# In[43]:


import time
start = time.time()
print("开始做梦")
dream_img= render_deepdream(dream_model,img,steps=200,step_size=0.01)
end = time.time()
end-start
print("梦醒时分")
dream_img = normalize_image(dream_img)
show_image(dream_img)


# In[ ]:




