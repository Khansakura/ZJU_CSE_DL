#!/usr/bin/env python
# coding: utf-8

# # 作业第7周：CGAN练习

# 1.仿照课件示例的GAN生成网络代码，实现Fashion_mnist数据集的条件生成对抗网络CGAN。<BR>
#     （对于感觉困难同学，可以降级选择完成GAN练习，计80%）

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,1024)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import glob
from IPython import display
import tensorflow as tf


# In[3]:


fasion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()


# In[4]:


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

#train_labels = tf.one_hot(train_labels, depth=10)
#train_labels = tf.cast(train_labels, tf.float32)


# In[5]:


buffer_size = 60000
batch_size = 256


# In[6]:


#train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(buffer_size).batch(batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)


# In[7]:


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    
    return model


# In[8]:


generator = make_generator_model()
generator.summary()


# In[9]:


noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()


# In[10]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model


# In[11]:


discriminator = make_discriminator_model()
discriminator.summary()


# In[12]:


decision = discriminator(generated_image)
print(decision)


# In[13]:


epochs = 50
noise_dim = 100
num_examples_to_generate = 16


# In[14]:


seed = tf.random.normal([num_examples_to_generate, noise_dim])

#labels = [i%10 for i in range(num_examples_to_generate)]

#labels = tf.one_hot(labels, depth=10)

#labels = tf.cast(labels, tf.float32)

#seed=tf.concat([seed,labels], 1)

print(seed.shape)


# In[15]:


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[16]:


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# In[17]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# In[18]:


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# In[19]:


@tf.function
def train_step(images):
    #images = data_batch[0]
    #labels = data_batch[1]
    #batch_size = images.shape[0]
    
    #labels_input = tf.concat([labels, labels, tf.zeros([batch_size, 8], dtype='float32')], 1)
    #labels_input = tf.reshape(labels_input, [batch_size, 28, 1])
    #labels_input = labels_input * tf.ones([batch_size, 28, 28], dtype='float32')
    #labels_input = tf.reshape(labels_input, [batch_size, 28, 28, 1])
    
    noise = tf.random.normal([batch_size, noise_dim])
    #noise_input = tf.concat([noise, labels], 1)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generated_images = generator(noise_input, training=True)
        generated_images = generator(noise, training=True)
        
        #img_input = tf.concat([images, labels_input], 3)
        #gen_input = tf.concat([generated_images, labels_input], 3)
        
        #real_output = discriminator(img_input, training=True)
        #fake_output = discriminator(gen_input, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator =disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss
        


# In[28]:


def train(dataset, epochs):
    for epoch in range(epochs):
        start=time.time()
        
        for i,image_batch in enumerate(dataset):
            g,d = train_step(image_batch)
            print("batch %d, gen_loss %f, disc_loss %f"%(i, g.numpy(), d.numpy()))
            
        display.clear_output(wait=True)
        generate_images(generator, seed)
        
        if (epoch + 1) % 5 == 0:
            generator.save(f'save/3200105710/gan_fashion-mnist_tf_{epoch+1}.h5')
        
        print("Time for epoch{} is {} sec".format(epoch+1, time.time() - start))


# In[29]:


def generate_images(model, test_input):
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4,4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()


# In[30]:


get_ipython().run_cell_magic('time', '', 'train(train_dataset, epochs)\n')


# In[33]:


model = tf.keras.models.load_model('save/3200105710/gan_fashion-mnist_tf_50.h5')
test_input = tf.random.normal([16, 100])
generate_images(model, test_input)


# In[ ]:




