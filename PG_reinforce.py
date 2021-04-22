import gym
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from moviepy.editor import *

gamma = 0.98
lr = 0.001

def save_frames(frames, path='./', filename='pgre.mp4'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, fps=60)

class PGRE(torch.nn.Module):
    def __init__(self, n_input, n_output):
        self.buffer = []
        super(PGRE, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, 256)
        self.linear2 = torch.nn.Linear(256, n_output)
        self.optimizer = torch.optim.Adam(self.parameters(), lr= lr)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return x
    
    def save(self, data):
        self.buffer.append(data)
    
    def train(self):
        Return = 0
        self.optimizer.zero_grad()
        for reward, value in self.buffer[::-1]:
            Return = gamma*Return + reward
            loss = -torch.log(value) * Return
            loss.backward()
        self.optimizer.step()
        self.buffer= []


def RL(share, n_epi, game, n_input, n_output, n_play, hyper): 
    global gamma
    global lr
    env = gym.make(game)
    PG = PGRE(n_input, n_output)
    interval = n_play
    sc = 0.0
    frame = []
    gamma, lr = hyper
    for n in range(n_epi):
        ob = env.reset()
        done = False
        while not done:
            if n%n_play == 0:
                frame.append(env.render(mode="rgb_array"))
            ob = torch.FloatTensor(ob)
            pi = PG(ob)
            A = torch.distributions.Categorical(pi)
            action = A.sample().item()
            ob, reward, done, tmp = env.step(action)
            PG.save((reward, pi[action]))
            sc += reward
        
        if n%n_play ==0:
            save_frames(frame)
            frame=[]
            tmp = str(int(n/n_play))
            VideoFileClip('pgre.mp4').write_gif(os.getcwd()+'\data\pgre' + tmp + '.gif', loop = 1)
            os.remove(os.getcwd() + '\pgre.mp4')
        PG.train()
        
        if n%interval ==0:
            if n!=0:
                print("{} : score:{}".format(n,sc/interval))
                share['r1'] = sc/interval
                sc = 0.0
            share['pgre'] = 1
            while share['wait']:
                continue
            share['pgre'] = 0
    env.close()