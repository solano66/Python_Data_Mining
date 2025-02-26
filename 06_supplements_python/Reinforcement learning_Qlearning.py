'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS (資訊與決策科學研究所暨智能控制與決策研究室), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### A. Reinforcement Learning Introduction 強化學習簡介
# 代理人/智能體agent
# 環境environment(動態且不確定dynamic & uncertain)
# 狀態state S_t 狀態轉移函數state transition probability Prob(S_t+1=s_t+1|S_t=s_t, A_t=a_t) 條件機率密度函數
# 行動action A_t 政策函數 policy function Prob(A_t=a_t|S_t=s_t) 條件機率密度函數
# 報酬reward(也包括懲罰) R_t

# 不同的行動產生不同的報酬
# 報酬需折現成為報償return U_t = R_t + gamma R_t+1 + gamma^2 R_t+2 + .....
# 行動的報酬與狀態有關

#### Frozen Lake Game 凍湖遊戲
# S: initial state 初始狀態
# F: frozen lake 凍湖
# H: hole 洞
# G: the goal 目標
# Red square: indicates the current position of the player 當前位置

import gym # !conda install conda-forge::gym==0.19.0 --y (python 3.6); !conda install conda-forge::gymnasium --y
print(gym.__version__) # 0.19.0
# https://www.gymlibrary.dev/index.html

MAX_ITERATIONS = 10

# "FrozenLake-v0" deprecated after gym 0.19.0 (included)
# DeprecatedEnv: Env FrozenLake-v0 not found (valid versions include ['FrozenLake-v1'])
env = gym.make("FrozenLake-v1")
# Attention to argument is_slippery (False means deterministic policy) 注意！is_slippery參數，預設是True，表隨機政策；False表確定性政策。Try 'is_slippery=False' and happily to get 1.00 'prob' & reward.
[(name, type(getattr(env, name))) for name in dir(env)]
# [(name, type(getattr(env, name))) for name in dir(env) if name is not '_np_random']

print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)

#### Play the game please (can try several times.) 凍湖遊戲試玩
env.reset() # 還原佈局
env.render() # Print the grid world (!conda install cogsci::pygame --y)
# Try pip install 'gymnasium[classic-control]' if your using zsh / macOS
# But gotten AssertionError: Something went wrong with pygame. This should never happen.
for i in range(MAX_ITERATIONS):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(
       random_action)
    print() # Line break
    env.render() # Print the grid world
    if i == 9:
        print() # Line break
        print('Max iterations has been reached')
        break
    if done:
        print() # Line break
        print('Finished')
        break

#### Stochastic vs Deterministic (under winning sequence) 隨機性與確定性(必勝序列)
# import gym
 
actions = {
    'Left': 0,
    'Down': 1,
    'Right': 2, 
    'Up': 3
}
 
print('---- winning sequence ------ ')
winning_sequence = (2 * ['Right']) + (3 * ['Down']) + ['Right']
print(winning_sequence)
 
env = gym.make("FrozenLake-v1") # Attention to argument is_slippery (False means deterministic policy) 注意！is_slippery參數，預設是True，表隨機政策；False表確定性政策。Try 'is_slippery=False' and happily to get 1.00 'prob' & reward.
env.reset() # Start from here and reset everytime please
env.render() # Print the grid world

for a in winning_sequence: # only six steps 只有六個步驟
    new_state, reward, done, info = env.step(actions[a])
    # AssertionError: Cannot call env.step() before calling reset()
    print() # Line break
    env.render() # Print the grid world
    print("Reward: {:.2f}".format(reward))
    print(info) # {'prob': 0.3333333333333333} or {'prob': 1.0} when is_slippery=False
    if done:
        break  
 
print()

env = gym.make("FrozenLake-v1", is_slippery=False) # Attention to argument is_slippery (False means deterministic policy) 注意！is_slippery參數，預設是True，表隨機政策；False表確定性政策。Try 'is_slippery=False' and happily to get 1.00 'prob' & reward.

#### 較大的FrozonLake
env = gym.make("FrozenLake8x8-v1")
env.reset()
env.render()

env = gym.make('FrozenLake-v1', map_name='8x8')
env.reset()
env.render()

#### 客製化FrozonLake圖
custom_map = [
    'SFFHF',
    'HFHFF',
    'HFFFH',
    'HHHFH',
    'HFFFG'
]
 
env = gym.make('FrozenLake-v1', desc=custom_map)
env.reset()
env.render()

#### CartPole Game at first glance 台車桿遊戲
import gym
env = gym.make('CartPole-v0') # env provides states and rewards

state = env.reset()
import time
for i in range(100):
    env.render()
    print(state) # [position of cart, velocity of cart, angle of the pole, angular velocity of the pole]
    
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(
       random_action)
    
    env.render() # Print the grid world
    time.sleep(0.3)
    if done:
        print('Finished')
        break


#### B. Q Learning Q學習
#### Q表的估計
# 二維列聯表, 橫列為狀態, 縱行為行動, 表中數字即為價值, i.e. 輸入狀態與行動, 傳回價值(Q值)
# RL的解法有三種: policy-based, value-based (Q-learning_OFF-policy, Temporal difference learning_ON-policy, Monte Carlo methods_ON-policy 價值基礎方法又有三種：離線Q學習、在線時間差學習、在線蒙地卡羅法), model-based
# A3C(Asynchronous Advantage Actor-Critic, A3C), A2C(Advantage Actor-Critic)融合了policy-based與value-based兩者(可視為政策基礎法的改良)

#### 動態規劃中的貝爾曼方程式(Bellman Equation)
# 遞迴方程式, 用於訓練過程中Q表的更新
# $Q(s, a) = r + gamma * max wrt a' Q(s', a')$

# 1. num_episode 外圈次數為情境模擬次數
# 2. num_steps 內圈次數是每次情境下, 訓練的步驟數
# 3. 根據當前的環境狀態與Q值選擇行動(可加入隨機性即epsilon greedy)
# 4. 獲得下一步的狀態和報酬
# 5. 更新Q表: Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])

#### Frozen Lake and Q-Learning 凍湖遊戲與Q學習
# import lib
import gym # pip install gym in the Anaconda Prompt
import numpy as np

# Load the environment
env = gym.make('FrozenLake-v1')

env.render()
env.reward_range # (0, 1)

# Implement Q-Table learning algorithm
# Initialize table with all zeros 以0值初始化Q表
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters and discount rate 設定學習參數與折現率
lr = .8 # learning rate
y = .95 # discount rate
num_episodes = 2000

# Create lists to contain total rewards and steps per episode 建立存放每回合總報酬與步數的串列
#jList = []
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False # Done initially set be False (not done yet!)
    j = 0
    # The Q-Table learning algorithm
    while j <= 99: # Iterate 100 times 迭代100次
        j+=1
        # Choose an action by greedily (with noise) picking from Q table 貪婪策略
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        # Get new state and reward from environment
        s1, r, d, _ = env.step(a) # Accepts an action and returns a tuple (observation, reward, done, info).
        # Update Q-Table with new knowledge 更新Q表
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1 # Update state
        if d == True: # If Done is True, then break the while loop
            break
    # jList.append(j)
    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
import pandas as pd
print(pd.DataFrame(Q, columns = ["Left", "Down", "Right", "Up"]))

#### C. Deep Q Learning 深度Q學習DQN
# 前面Bellman Equation的方法有一個問題，那就是真實世界的狀態可能無窮多，如此Q-table的建構產生問題
# 透過神經網路實作Q-table (row: states; column: actions; entry: values)，輸入狀態後產出不同行動的Q值

#### 經驗的重播(RL: Learning from experiences; SVL: Learning from examples)
# 記憶的經驗儲存(s, a, r, s')四元組

#### 剝削與探索之間的權衡取捨(Tradeoff between exploitation and exploration)
# 探索: 行動選取時加入一定的隨機性
# 剝削: 隨著訓練加深，逐漸降低隨機性

#### 算法：
# 1. 初始化記憶D
# 2. 以隨機權重初始化行動價值網路Q
# 3. 對每個情境(M個)
#   4. 時間從t到T
#       5. epsilon的機率隨機產生行動，否則a_t = argmax_a Q(s,a)
#       6. 執行行動a_t，並觀察報酬r_t+1與狀態s_t+1
#       7. 將轉移經驗<s_t, a_t, r_t+1, s_t+1>存入記憶體D中
#       8. 從D: <s_j, a_j, r_j, s'_j>中進行小批量抽樣
#       8. Q^_j = r_j如果情境已終止; 否則，Q^_j = r_j + gamma * max_a' Q(s'_j, a')
#       9. 以損失函數(Q^_j-Q(s_j,a_j))^2進行梯度陡降計算
#   10. endfor
# 11 endfor

#### Frozen Lake
# import lib
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

# load env
env = gym.make('FrozenLake-v1')

# The Q-Network Approach
tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions 選擇行動的前向式神經網路
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# Training

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode 建立存放每回合總報酬與步數的串列
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation 重設環竟，以取得第一個觀測值
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

# Some statistics on network performance
plt.plot(rList)

plt.plot(jList)

#### CartPole 再訪台車桿遊戲(自行參考)
# import lib
import gym # A toolkit for developing and comparing reinforcement learning algorithms. https://gym.openai.com/
import tensorflow as tf
import numpy as np

# Create the Cart-Pole game environment
env = gym.make('CartPole-v0')
env.reset()
env.render()

# Q-network
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, 
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

# Experience replay
from collections import deque
class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

# hyperparameters
train_episodes = 1000          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)

# Populate the experience memory
# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())

memory = Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state


# Training
# Now train with experiences
saver = tf.train.Saver()
rewards_list = []
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1

            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
            if explore_p > np.random.rand():
                # Make a random action
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps

                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))

                # Add experience to memory
                memory.add((state, action, reward, next_state))

                # Start new episode
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1

            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Train network
            target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

            # Set target_Qs to 0 for states where episode ends
            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            target_Qs[episode_ends] = (0, 0)

            targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                feed_dict={mainQN.inputs_: states,
                                           mainQN.targetQs_: targets,
                                           mainQN.actions_: actions})

    saver.save(sess, "checkpoints/cartpole.ckpt")

# Testing
test_episodes = 10
test_max_steps = 400
env.reset()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render() 

            # Get action from Q-network
            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                t += 1

env.close()

#### References:
# 1. Gym Tutorial: The Frozen Lake: https://reinforcementlearning4.fun/2019/06/16/gym-tutorial-frozen-lake/
# 2. https://blog.csdn.net/Young_Gy/article/details/73485518


