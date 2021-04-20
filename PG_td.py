import gym
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from moviepy.editor import *

def save_frames(frames, path='./', filename='pgtd.mp4'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, fps=60)

class PGTD(torch.nn.Module):
    def __init__(self, n_input, n_output):
        self.gamma = 0.98
        self.buffer = []
        super(PGTD, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, 256)
        self.pi = torch.nn.Linear(256, n_output)
        self.val = torch.nn.Linear(256, 1)      
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

    def pi_forward(self, x, softdim = 0):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        pi = self.pi(x)
        pi = torch.nn.functional.softmax(pi, dim=softdim)
        return pi

    def val_forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        val = self.val(x)
        return val

    def save(self, data):
        self.buffer.append(data)

    def train(self):
        sl, al, rl, spl, dl = [], [], [], [], []
        for item in self.buffer:
            s, a, r, sp, d = item
            sl.append(s)
            al.append([a])
            rl.append([r/100.0])
            spl.append(sp)
            dl.append([not(d)])
        s, action, reward, s_prime, done = torch.FloatTensor(sl), torch.tensor(al), torch.FloatTensor(rl), torch.FloatTensor(spl), torch.FloatTensor(dl)
        val = self.val_forward(s)
        val_prime = self.val_forward(s_prime)
        td = reward + self.gamma * val_prime * done
        delta =  td - val
        pi = self.pi_forward(s, softdim=1)
        pi_a = pi.gather(1,action)
        loss = -torch.log(pi_a)*delta.detach()  + torch.nn.functional.smooth_l1_loss(td.detach(), val)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.buffer = []

def RL(share, n_epi, game, n_input, n_output, n_play): 
    """
    PG = 0 : reinforce
    PG = 1 : Q actor-critic with TD(0)
    PG = 2 : Advantage actor-critic with TD
    """
    env = gym.make(game)
    
    """
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
    """
    PG = PGTD(n_input, n_output)
    sc = 0.0
    epoch = 10
    interval = n_play
    frame = []

    for n in range(n_epi):
        s = env.reset()
        done = False
        while not done:
            if n%n_play == 0:
                frame.append(env.render(mode="rgb_array"))
            for step in range(epoch):
                pi = PG.pi_forward(torch.FloatTensor(s))
                A = torch.distributions.Categorical(pi)
                action = A.sample().item()
                s_prime, reward, done, tmp = env.step(action)
                PG.save((s,action,reward,s_prime,done))
                s = s_prime
                sc += reward
                if done:
                    if n%n_play ==0:
                        save_frames(frame)
                        frame=[]
                        if os.path.isfile(os.getcwd()+'\pgtd.gif'):
                            os.remove(os.getcwd() + '\pgtd.gif')
                        VideoFileClip('pgtd.mp4').write_gif('pgtd.gif')
                        os.remove(os.getcwd() + '\pgtd.mp4')
                    break
            PG.train()
        if n%interval ==0:
            if n!=0:
                print("{} : score:{}".format(n,sc/interval))
                sc = 0.0
            share['pgtd'] = 1
            while share['wait']:
                continue
            share['pgtd'] = 0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi")
    RL(int(n_epi))