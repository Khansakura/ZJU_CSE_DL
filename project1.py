#!/usr/bin/env python
# coding: utf-8

# # 期末综合练习（1）

# ### 设计一个CNN网络，训练自定义图像数据集的分类操作。 通过优化网络结构与超参数、正则化、数据增强等各种手段，尽可能提高准确率。（要求自建模型，形式不限，不能使用预定义模型）
# Dataset Context：<BR>
# This is image data of Natural Scenes around the world.<BR>
# Dataset Content：<BR>
# This Data contains around 25k images of size 96x96 distributed under 6 categories.<BR>
# {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2,
# 'mountain' -> 3,  'sea' -> 4, 'street' -> 5 }
# <BR>
#     
# #### 要求：
# 1)利用callback将最佳模型保存到文件(注意：在"save"目录下建立以自己学号命名的子目录，然后在该子目录下保存文件)。显示loss曲线和accuracy曲线。<BR>
# 2)读取最佳模型进行指标评估并显示结果，展示混淆矩阵。<BR>
# 3)尝试展示典型图片的热力图<BR>
# 
# #### 考核办法：
# 1）程序功能完成度<BR>
# 2）score = model.evaluate(testset)<BR>
# 计算得到的准确率为指标，达到0.8为及格成绩起点，0.9优秀<BR>
#### 数据组织形式：seg_train为训练集，seg_test为测试集，各category子目录下存有jpg文件的图片
data/project/
    ├── seg_test
    │   └── seg_test
    │       ├── buildings
    │       ├── forest
    │       ├── glacier
    │       ├── mountain
    │       ├── sea
    │       └── street
    └── seg_train
        └── seg_train
            ├── buildings
            ├── forest
            ├── glacier
            ├── mountain
            ├── sea
            └── street

#### 数据读取方法：
  IMG_SIZE=96
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(...)
  trainset = datagen.flow_from_directory(trainpath, (IMG_SIZE, IMG_SIZE), batch_size=32)
  testset = datagen.flow_from_directory(testpath, (IMG_SIZE, IMG_SIZE), batch_size=32)
  
model.fit(trainset, validation_data=testset, ...)  #不区分验证集与测试集<BR>

关于数据增强，ImageDataGenerator自带很多增强方式，十分方便，请参考相关文档
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,2048)


# In[2]:


#创建保存目录及文件夹
import os
#定义路径并进行查询操作，防止重复创建
save_dir = "save/3200105710"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# In[3]:


#导入部分必要库
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


#定义图像大小
IMG_SIZE = 96 
#定义类别个数
NUM_CLASSES = 6 
#定义验证集及测试集路径
trainpath = "data/project/seg_train/seg_train"
testpath = "data/project/seg_test/seg_test"
#数据增强
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,#归一化
    width_shift_range=0.1, # 随机水平平移
    height_shift_range=0.1, # 随机垂直平移
    zoom_range=0.1, # 随机缩放
    horizontal_flip=True, # 随机水平翻转
    validation_split=0.2 # 划分验证集
)
#获取训练集和验证集
trainset = datagen.flow_from_directory(trainpath, (IMG_SIZE, IMG_SIZE), batch_size=64, subset="training")
valset = datagen.flow_from_directory(trainpath, (IMG_SIZE, IMG_SIZE), batch_size=64, subset="validation")


# In[5]:


#定义CNN模型结构

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2,2)),
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])


# In[6]:


#观察模型结构
model.summary()


# In[7]:


#编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])


# In[8]:


#训练模型，用callback保存
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5), # 提前停止训练如果验证损失不再下降
    tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "model.h5"), save_best_only=True) # 保存最佳模型权重
]
history = model.fit(trainset, validation_data=valset, epochs=40, callbacks=callbacks)


# In[9]:


#绘制loss和accuracy曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()


# In[10]:


#测试集评估
model.load_weights("save/3200105710/model.h5")
testset = datagen.flow_from_directory(testpath, (IMG_SIZE, IMG_SIZE), batch_size=32)
test_loss, test_acc = model.evaluate(testset)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


# In[66]:


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


# In[71]:


y_true = testset.classes # 真实标签是一个整数数组，表示每张图片所属的类别编号（0-5）
y_pred = model.predict(testset) # 预测标签是一个概率数组，表示每张图片属于每个类别的概率（0-1）
y_pred = np.argmax(y_pred, axis=1) # 将概率数组转换为整数数组，表示每张图片预测的类别编号（0-5）
cm = tf.math.confusion_matrix(y_true, y_pred) # 计算混淆矩阵


# In[72]:


#将tf结构的混淆矩阵转化为np格式
cm=cm.numpy()


# In[73]:


cm


# In[74]:


#定义类别名及混淆矩阵热力图表示
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
plot_confusion_matrix(cm,
                      class_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=False,
                      text=True)


# In[16]:


model.evaluate(testset)


# In[48]:


#查看文件名，选择一张图片作为后续的热力图处理
import os

# 要获取文件名的文件夹路径
folder_path = "data/project/seg_train/seg_train/sea"

# 使用os.listdir()函数获取文件夹下的所有文件名
file_names = os.listdir(folder_path)

# 打印所有文件名
for file_name in file_names:
    print(file_name)


# In[59]:


# 尝试展示典型图片的热力图
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

img_path='data/project/seg_train/seg_train/sea/431.jpg'


# 选择一个测试图片，这里假设是第0张图片
img = img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
# 将图片转换为4维张量
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# 预测图片的类别
pred = model.predict(img_tensor)
pred = np.argmax(pred, axis=1)
# 显示图片
plt.imshow(img)
plt.show()

# 获取模型的最后一个卷积层的输出
last_conv_layer = model.get_layer('conv2d_2')
last_conv_model = Model(inputs=model.input, outputs=last_conv_layer.output)

# 获取模型的分类层的权重
classifier_weights = model.layers[-1].get_weights()[0]

# 计算最后一个卷积层输出的加权和，权重为分类层对应类别的权重
conv_output = last_conv_model.predict(img_tensor)
# 将conv_output转换为float类型
conv_output = conv_output.astype(float)
cam = np.dot(conv_output, classifier_weights[:, pred])
# 将cam转换为一维数组
cam = np.squeeze(cam)


# In[75]:


# 将热力图归一化并缩放到原始图片的大小
cam = (cam - cam.min()) / (cam.max() - cam.min())
print(cam)


# In[77]:


#绘制图像对应最后一个卷积层的热力图
plot_confusion_matrix(cam,
                      ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'],
                      title='Confusion matrix',
                      cmap=None,
                      normalize=False,
                      text=False)


# In[79]:


#为cam增加一个维度，便于与图像叠加
cam = np.expand_dims(cam, axis=-1)
cam = tf.image.resize(cam, (96, 96)).numpy()

# 将热力图叠加到原始图片上，使用jet颜色映射
plt.imshow(img, alpha=0.5)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.axis('off')
plt.show()


# ### 总结与心得：
# 
# 大作业一中首次在没有老师提供的基础代码结构的情况下进行任务求解和代码编写工作，难度增大不少，但总体结构和数据处理方式依旧与前面的处理方式相似。在对模型结构的操作上，应用了callback进行模型的保存，同时搜索资料发现callback可以实现早停的功能，为了减少模型训练的时间，便于进行参数优化调节的工作，加入了patience早停的部分。
# 
# 模型调参工作进行了数十次，但调节参数模型训练结果的准确性一直相对不是很高，在复杂模型下，训练集准确率可以很高，达到0.92-0.94之间，但很难解决过拟合的问题，搜索资料并尝试了dropout，正则化，数据增强后，过拟合问题并没有得到很好的解决。因此考虑牺牲一部分的准确度，解决过拟合问题，最终训练曲线的trainloss、trainacc和valloss、valacc也基本重合，过拟合问题得到很好的解决。有意思的是在寻找资料的过程中，发现BatchNormalization层似乎可以有效的降低过拟合的程度，尝试之后确实对模型的最终表现起到了正向优化作用，因此也在模型中进行采用。
# 
# 同时在最后呈现整体预测的混淆矩阵和混淆矩阵对应的热力图之后，对应于典型图片的热力图的任务要求，参考了老师上课所讲的Deepdream章节的相关原理和Deepdream当周的作业，选择了一张sea图像进行热力图处理。最终也得到了典型图片的热力图叠加结果，实现了卷积神经网络的可视化工作。
# 
# 总体来讲，本次实验我收获良多，有机会自己从基本的数据处理到模型构建到训练、调参、以及最终的评估和可视化的整体流程更加熟悉并能够用代码进行实现，也对老师课上所讲的知识点有了更深入的理解。

# In[ ]:




