import gym
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os

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
        self.gamma = 0.98
        self.lamb = 0.95
        self.epsilon = 0.1
        self.K = 3
        self.buffer = []
        super(PPO, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, 256)
        self.pi = torch.nn.Linear(256, n_output)
        self.val = torch.nn.Linear(256, 1)      
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

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
            td = reward + self.gamma * self.val_forward(s_prime) * done
            delta = td - self.val_forward(s)
            delta = delta.detach().numpy()

            Al = []
            A = 0.0
            for value in delta[::-1]:
                A = value[0] + self.gamma * self.lamb * A
                Al.append([A])
            Al.reverse()
            A = torch.FloatTensor(Al)

            pi = self.pi_forward(s, softmax_dim=1)
            pi_a = pi.gather(1,action)
            r = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            loss = -torch.min(r*A, torch.clamp(r,1-self.epsilon,1+self.epsilon)*A) + torch.nn.functional.smooth_l1_loss(td.detach(), self.val_forward(s))
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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
    ppo = PPO(n_input, n_output)
    T = 20
    sc = 0.0
    interval = n_play
    frame = []

    for n in range(n_epi):
        s = env.reset()
        done = False
        while not done:
            if n%n_play==0:
                frame.append(env.render(mode="rgb_array"))
            for step in range(T):
                pi = ppo.pi_forward(torch.FloatTensor(s))
                A = torch.distributions.Categorical(pi)
                action = A.sample().item()
                s_prime, reward, done, tmp = env.step(action)
                ppo.save((s,action,reward,s_prime,pi[action].item(),done))
                s = s_prime
                sc += reward
                if done:
                    if n%n_play ==0:
                        save_frames(frame)
                        frame=[]
                        if os.path.isfile(os.getcwd()+'\ppo.gif'):
                            os.remove(os.getcwd() + '\ppo.gif')
                        os.system("ffmpeg -i "+ os.getcwd() + "\ppo.mp4 " + os.getcwd() + "\ppo.gif")
                        os.remove(os.getcwd() + '\ppo.mp4')
                    break
            ppo.train()

        if n%interval ==0:
            if n!=0:
                print("{} : score:{}".format(n,sc/interval))
                sc = 0.0
            share['ppo'] = 1
            while share['wait']:
                continue
            share['ppo'] = 0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi?")
    RL(int(n_epi))