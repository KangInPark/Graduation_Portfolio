import gym
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os

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
        self.gamma = 0.98
        super(PGRE, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, 256)
        self.linear2 = torch.nn.Linear(256, n_output)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

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
            Return = self.gamma*Return + reward
            loss = -torch.log(value) * Return
            loss.backward()
        self.optimizer.step()
        self.buffer= []


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
    PG = PGRE(n_input, n_output)
    interval = n_play
    sc = 0.0
    frame = []

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
            if os.path.isfile(os.getcwd()+'\pgre.gif'):
                os.remove(os.getcwd() + '\pgre.gif')
            os.system("ffmpeg -i "+ os.getcwd() + "\pgre.mp4 " + os.getcwd() + "\pgre.gif")
            os.remove(os.getcwd() + '\pgre.mp4')
        PG.train()
        
        if n%interval ==0:
            if n!=0:
                print("{} : score:{}".format(n,sc/interval))
                sc = 0.0
            share['pgre'] = 1
            while share['wait']:
                continue
            share['pgre'] = 0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi")
    RL(int(n_epi))