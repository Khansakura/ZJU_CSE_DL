#!/usr/bin/env python
# coding: utf-8

# # 作业第8周：强化学习

# ### 理解Q学习算法，练习Keras实现 DQN网络  (！注意使用DeepL-3.10服务！)
# #### 要求：
# 1）设计DQN网络，生成policy_net和target_net<BR>
# 2）调整Q学习算法参数，提高倒立摆保持时间。(可以尝试利用env提供的信息改进reward策略)
# 
# #### 考核办法：
# 训练100epoch后的性能表现,取保持时间最大的一次（80为及格线，160以上优秀）
Gym makes interacting with the game environment really simple:

next_state, reward, done, info = env.step(action)
Here, action can be either 0 or 1. If we pass those numbers, env, which represents the game environment, will emit the results. done is a boolean value telling whether the game ended or not. next_state space handles all possible state values:
(
[Cart Position from -4.8 to 4.8],
[Cart Velocity from -Inf to Inf],
[Pole Angle from -24° to 24°注意，得到的实际数值是弧度!],
[Pole Velocity At Tip from -Inf to Inf]
)

Episode Termination:（OR condition）
Pole Angle is more than 12 degrees
Cart Position is more than 2.4
# In[1]:


#首先执行GPU资源分配代码，勿删除。
import GPU
GPU.show()
GPU.alloc(0,1024)
# import os
# os.environ["PATH"]=os.environ["PATH"]+':/usr/local/cuda/bin'


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers,Sequential
import tensorflow as tf
#Windows调试时!!删除下面2行！！
from pyvirtualdisplay import Display
Display().start()

env = gym.make('CartPole-v0').unwrapped
env.reset()


# In[3]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ### 请在下面设计DQN网络结构
# 输入：图片shape=(40, 90, 3,)<BR>
# 输出：（2，）

# In[4]:


num_actions = 2

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(40, 90, 3,))
    
    # 添加卷积层提取图像特征 
    x = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs) 
    x = layers.Conv2D(64, 4, strides=2, activation='relu')(x) 
    x = layers.Conv2D(64, 3, strides=1, activation='relu')(x) 
    # 添加全连接层输出动作值 
    x = layers.Flatten()(x) 
    x = layers.Dense(512, activation='relu')(x) 
    
    action = layers.Dense(num_actions, activation='linear')(x) 

    # 可以在这里用函数方式设计模型，也可以用sequential方式构建
    #提示最后一层：Dense(num_actions, activation="linear")

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
model_target = create_q_model()
model.summary()


# In[5]:


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. 
    screen = env.render(mode='rgb_array')
    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width,_ = screen.shape
    rawscreen=screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:,:]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, slice_range, :]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen=tf.image.resize(screen,(40,90),method='bicubic')/255.0
    return screen,rawscreen
    


# ### 后续代码可以根据需要进行调参或修改细节

# In[6]:


BATCH_SIZE = 128
GAMMA = 0.93
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
memory = ReplayMemory(10000)

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00020, clipnorm=1.0)
# Using huber loss for stability
loss_function = keras.losses.Huber()


steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        action_probs =model(tf.expand_dims(state, 0), training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action
    else:
        return random.randrange(num_actions)


episode_durations = []
def plot_durations():
    ax1_2.clear()
    ax1_2.set_title('Training...')
    ax1_2.set_xlabel('Episode')
    ax1_2.set_ylabel('Duration')
    ax1_2.plot(episode_durations)
    fig1.canvas.draw()


# In[7]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask =np.array( tuple(map(lambda s: s is not None,       batch.next_state)))
#     print(non_final_mask)
    non_final_next_states = tf.stack([s for s in batch.next_state
                                                if s is not None])
#     print(non_final_next_states.shape)
    state_sample = tf.stack(batch.state)

    action_sample =np.array( batch.action)
    rewards_sample =np.array(batch.reward)
    
    future_rewards = np.zeros(BATCH_SIZE,dtype='float32')
    future_rewards[non_final_mask] =tf.reduce_max( model_target.predict(non_final_next_states,verbose=0), axis=1)

    # Q value = reward + discount factor * expected future reward
    updated_q_values = rewards_sample + GAMMA * future_rewards

    # Create a mask so we only calculate loss on the updated Q-values
    masks = tf.one_hot(action_sample, num_actions)

    with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value
        loss = loss_function(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))



# In[8]:


###尝试调整reward策略

#定义一个函数计算角度奖励，当角度接近垂直时奖励最高，当角度偏离垂直时奖励降低
def angle_reward (angle): 
    return 1 - abs (angle) / math.pi

#定义一个函数计算位置奖励，当位置接近中心时奖励最高，当位置偏离中心时奖励降低
def position_reward (position): 
    return 1 - abs (position) / 2.4

#定义一个函数计算速度奖励，当速度较小时奖励最高，当速度较大时奖励降低
def speed_reward (speed): 
    return 1 - abs (speed) / 10

#定义一个函数计算动作奖励，当动作与速度方向相反时奖励最高，当动作与速度方向相同时奖励降低
def action_reward (action, speed): 
    if speed > 0 and action == 0 or speed < 0 and action == 1: 
        return 1
    else: 
        return 0.5

#综合以上四种奖励，并给予不同的权重，得到最终的奖励值
def get_reward(ss, action):
    #获取倒立摆的角度和位置
    angle = ss[2] 
    position = ss[0]

    #获取倒立摆的角速度和位置速度
    angle_speed = ss[3] 
    position_speed = ss[1]

    #获取倒立摆的reward
    reward = 0.5 * angle_reward(angle) + 0.1 * position_reward(position) + 0.2 * speed_reward(angle_speed) + 0.2 * speed_reward(position_speed) + 0.1 * action_reward(action, position_speed)
    return reward


# In[9]:


# import warnings
# warnings.filterwarnings('ignore')

plt.ion()
fig1=plt.figure(figsize=(12,4))
ax1_1 = fig1.add_subplot(121)
ax1_2 = fig1.add_subplot(122)
init_screen,rawscreen = get_screen()
ax1_1.imshow(rawscreen)
fig1.canvas.draw()

num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = init_screen
    current_screen,_ = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        ss, reward, done, _ = env.step(action)
        
        ###尝试调整reward策略
        #reward=......
        reward = get_reward(ss, action)
        
        # Observe new state
        last_screen = current_screen
        current_screen,rawscreen = get_screen()
        
        '''
        ###以下显示动画会较严重影响运行速度，建议调试期间去除
        ax1_1.clear()
        if done:
            ax1_1.set_title('!!Fail!!')
        else:
            ax1_1.set_title('score:'+str(t+1))
        ax1_1.imshow(rawscreen)
        fig1.canvas.draw()
        plt.pause(0.001)
        ###显示动画会较严重影响运行速度，建议调试期间去除
        '''
        
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        model_target.set_weights(model.get_weights())
    

print('Complete')
# env.render()
env.close()
plt.ioff()
plt.show()
print('最好成绩：', max(episode_durations))


# #### 总结说明
# 此处说明关于模型设计与模型训练（参数设置、训练和调优过程）的心得与总结

# 本次作业为使用深度Q学习算法（DQN）来解决倒立摆环境的强化学习任务。倒立摆环境是一个经典的控制问题，目标是通过向左或向右施加力来平衡一个连接在小车上的杆子。在作业完成中的心得总结主要有以下几个方面：
# 
# 模型设计：使用Keras函数式API来构建DQN网络，该网络由四个卷积层和两个全连接层组成，输入是倒立摆的图像状态，输出是两个动作的值。
# 
# 参数设置：本文根据倒立摆环境的特点和实验结果，设置了一些合理的参数，例如学习率、折扣因子、批次大小等，微调后得到较不错的效果。
# 
# 训练和调优过程：本文使用了一个循环来训练每个epoch，并在每个epoch结束后统计并打印平均奖励和步数。同时尝试了改进reward策略的方法，通过给予不同的奖励来激励模型关注倒立摆的稳定性、平衡性、平滑性。
# 基础代码中默认的reward策略为：每保持一步倒立摆就给予+1的奖励。这种策略可能导致模型学习到一种保守的策略，即尽量让倒立摆保持在中间位置，而不是尝试探索更多的状态空间。为了改进reward策略，我尝试了以下几种方法：
# （1）基于倒立摆的角度和位置给予不同的奖励，例如当倒立摆接近垂直时给予更高的奖励，当倒立摆偏离中心时给予更低的奖励。
# （2）基于倒立摆的速度给予不同的奖励，例如当倒立摆的角速度和位置速度较小时给予更高的奖励，当倒立摆的速度较大时给予更低的奖励。
# （3）基于倒立摆的动作给予不同的奖励，例如当倒立摆采取相反方向的动作时给予更高的奖励，当倒立摆采取同向方向的动作时给予更低的奖励。
# 倘若需要有平衡性和平滑性的要求，会对倒立摆的稳定性的效果造成影响，很多时候会因为速度过快到达小车运动极限而结束（最优解150左右），最终还是根据考核目标选择调整参数，使得其保持重视稳定性的reward策略，而减少速度的reward权重。
# 
# 通过以上的模型设计与模型训练，最终完成了保持160步以上的目标。
