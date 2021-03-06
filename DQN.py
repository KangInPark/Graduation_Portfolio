import gym
import torch
import random
import collections
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from moviepy.editor import *

gamma = 0.98
lr = 0.001
epsilon_max = 0.1
epsilon_min = 0.01
epsilon_weight = 0.0001
def save_frames(frames, path='./', filename='dqn.mp4'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, fps=60)

class Experience_replay():
    def __init__(self):
        self.length = 10000
        self.dq = collections.deque()
        self.batch_size = 32

    def insert(self, data):
        self.dq.append(data)
        if(len(self.dq) > self.length):
            self.dq.popleft()
    
    def sample(self,n):
        return random.sample(self.dq, min(len(self.dq),n))

class DQN(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, n_output)     

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        return x

    def action(self, s, epsilon):
        rand = random.random()
        if rand < epsilon:
            return random.randint(0,1)
        else:
            return self.forward(s).argmax().item()

def train(DQN, target, buffer, optimizer):
    T = 10
    for t in range(T):
        data = buffer.sample(buffer.batch_size)
        sl, al, rl, spl, dl = [], [], [], [], []
        for item in data:
            s, a, r, sp, d = item
            sl.append(s)
            al.append([a])
            rl.append([r])
            spl.append(sp)
            dl.append([d])
        s, action, reward, s_prime, done = torch.FloatTensor(sl), torch.tensor(al), torch.FloatTensor(rl), torch.FloatTensor(spl), torch.FloatTensor(dl)
        q = DQN(s)
        q_a = q.gather(1,action)
        max_q_prime = target(s_prime).max(1)[0].unsqueeze(1)
        td = r + gamma * max_q_prime * done
        loss = torch.nn.functional.smooth_l1_loss(td, q_a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def RL(share, n_epi, game, n_input, n_output, n_play, hyper): 
    global gamma
    global lr
    global epsilon_max
    global epsilon_min
    global epsilon_weight
    gamma, lr, epsilon_max, epsilon_min, epsilon_weight = hyper
    env = gym.make(game)
    dqn = DQN(n_input, n_output)
    target = DQN(n_input, n_output)
    target.load_state_dict(dqn.state_dict())
    buffer = Experience_replay()
    optimizer = torch.optim.Adam(dqn.parameters(), lr = lr)
    interval = n_play
    sc = 0.0    
    frame = []
    for n in range(n_epi+1):
        epsilon = max(epsilon_min, epsilon_max - epsilon_weight*n)
        s = env.reset()
        done = False
        epi_reward = 0
        while not done:
            if n%n_play ==0:
                frame.append(env.render(mode="rgb_array"))
            action = dqn.action(torch.FloatTensor(s), epsilon)
            s_prime, reward, done, tmp = env.step(action)
            sc += reward
            epi_reward += reward
            if game == 'MountainCar-v0':
                if done:
                    reward = 200 + epi_reward
                else:
                    reward = abs(s_prime[0] - s[0])
            buffer.insert((s,action,reward,s_prime,not(done)))
            s = s_prime
            if done:
                if n%n_play ==0:
                    save_frames(frame)
                    frame=[]
                    tmp = str(int(n/n_play))      
                    VideoFileClip('dqn.mp4').write_gif(os.getcwd()+'\data\dqn'+ tmp + '.gif', loop = 1)
                    os.remove(os.getcwd() + '\dqn.mp4')
                break
        train(dqn, target, buffer, optimizer)
        if n%5 ==0 and n !=0:
            target.load_state_dict(dqn.state_dict())
        if n%interval ==0:
            if n!=0:
                print("{} : score:{}".format(n,sc/interval))
                share['r3'] = sc/interval
                sc = 0.0
            else:
                sc = 0.0
            share['dqn'] = 1
            while share['wait'] and n!= n_epi:
                continue
            share['dqn'] = 0
        if n == n_epi:
            if n%interval ==0:
                share['s'] = 2
            else:
                share['s'] = 0
            share['dqn'] = 1
    env.close()