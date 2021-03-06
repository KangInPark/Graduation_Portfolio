import gym
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from moviepy.editor import *

gamma = 0.98
lr = 0.001
lamb = 0.95
epsilon = 0.1

def save_frames(frames, path='./', filename='ppo.mp4'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, fps=60)

class PPO(torch.nn.Module):
    def __init__(self,n_input, n_output):
        self.K = 3
        self.buffer = []
        super(PPO, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, 256)
        self.pi = torch.nn.Linear(256, n_output)
        self.val = torch.nn.Linear(256, 1)      
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def pi_forward(self, x, softmax_dim = 0):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        pi = self.pi(x)
        pi = torch.nn.functional.softmax(pi, dim=softmax_dim)
        return pi

    def val_forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        val = self.val(x)
        return val

    def save(self, data):
        self.buffer.append(data)

    def train(self):
        sl, al, rl, spl, pl, dl = [], [], [], [], [], []
        for item in self.buffer:
            s, a, r, sp, p, d = item
            sl.append(s)
            al.append([a])
            rl.append([r])
            spl.append(sp)
            pl.append([p])
            dl.append([not(d)])

        s, action, reward, s_prime, prob_a, done = torch.FloatTensor(sl), torch.tensor(al), torch.FloatTensor(rl), torch.FloatTensor(spl), torch.FloatTensor(pl), torch.FloatTensor(dl)

        self.buffer = []

        for epoch in range(self.K):
            td = reward + gamma * self.val_forward(s_prime) * done
            delta = td - self.val_forward(s)
            delta = delta.detach().numpy()

            Al = []
            A = 0.0
            for value in delta[::-1]:
                A = value[0] + gamma * lamb * A
                Al.append([A])
            Al.reverse()
            A = torch.FloatTensor(Al)

            pi = self.pi_forward(s, softmax_dim=1)
            pi_a = pi.gather(1,action)
            r = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            loss = -torch.min(r*A, torch.clamp(r,1-epsilon,1+epsilon)*A) + torch.nn.functional.smooth_l1_loss(td.detach(), self.val_forward(s))
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def RL(share, n_epi, game, n_input, n_output, n_play, hyper):
    global gamma
    global lr
    global lamb
    global epsilon
    gamma, lr, lamb, epsilon = hyper
    env = gym.make(game)
    
    ppo = PPO(n_input, n_output)
    T = 20
    sc = 0.0
    interval = n_play
    frame = []
    for n in range(n_epi+1):
        s = env.reset()
        done = False
        epi_reward = 0
        while not done:
            for step in range(T):
                if n%n_play==0:
                    frame.append(env.render(mode="rgb_array"))
                pi = ppo.pi_forward(torch.FloatTensor(s))
                A = torch.distributions.Categorical(pi)
                action = A.sample().item()
                s_prime, reward, done, tmp = env.step(action)
                sc += reward
                epi_reward += reward
                if game == 'MountainCar-v0':
                    if done:
                        reward = 200 + epi_reward
                    else:
                        reward = abs(s_prime[0] - s[0])
                ppo.save((s,action,reward,s_prime,pi[action].item(),done))
                s = s_prime
                if done:
                    if n%n_play ==0:
                        save_frames(frame)
                        frame=[]
                        tmp = str(int(n/n_play))
                        VideoFileClip('ppo.mp4').write_gif(os.getcwd()+'\data\ppo'+ tmp +'.gif', loop = 1)
                        os.remove(os.getcwd() + '\ppo.mp4')
                    break
            ppo.train()

        if n%interval ==0:
            if n!=0:
                print("{} : score:{}".format(n,sc/interval))
                share['r4'] = sc/interval
                sc = 0.0
            else:
                sc = 0.0
            share['ppo'] = 1
            while share['wait'] and n!= n_epi:
                continue
            share['ppo'] = 0
        if n == n_epi:
            if n%interval ==0:
                share['s'] = 2
            else:
                share['s'] = 0
            share['ppo'] = 1
    env.close()