import gym
import torch
import numpy as np
import random
import collections

class Model():
    def __init__(self):
        self.buffer = collections.deque(maxlen=10000)
        self.tran = {}
        self.reward = {}
        self.done = {}

    def step(self, s, a):
        s = tuple(s.tolist())
        if (s,a) not in self.tran:
            return torch.numpy([0.0, 0.0, 0.0, 0.0]), 0.0, False
        else:
            return self.tran[(s,a)], self.reward[(s,a)], self.done[(s,a)]

    def learn(self, buffer):
        for item in buffer:
            self.buffer.append(item)
            s, a, r, sp, d = item
            s = tuple(s.tolist())
            self.tran[(s,a)] = sp
            self.reward[(s,a)] = r
            self.done[(s,a)] = d

class Dyna(torch.nn.Module):
    def __init__(self):
        self.gamma = 0.98
        self.buffer = []
        super(Dyna, self).__init__()
        self.linear1 = torch.nn.Linear(4, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

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

        self.optimizer.zero_grad()
        q = self.forward(s)
        q_a = q.gather(1,action)
        max_q_prime = self.forward(s_prime).max(1)[0].unsqueeze(1)
        td = reward + self.gamma * max_q_prime *done
        loss = torch.nn.functional.smooth_l1_loss(td.detach(), q_a)
        loss.backward()
        self.optimizer.step()
        self.buffer= []

def RL(n_epi): 
    env = gym.make('CartPole-v1')
    
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
    dyna = Dyna()
    model = Model()

    sc = 0.0
    interval = 50
    nself = 10
    epoch = 20
    for n in range(n_epi):
        epsilon = max(0.01, 0.1 - 0.01 * n/100)
        s = env.reset()
        edone = False
        while not edone:
            for step in range(epoch):
                action = dyna.action(torch.FloatTensor(s), epsilon)
                s_prime, reward, done, tmp = env.step(action)
                edone = done
                dyna.save((s, action, reward, s_prime, done))
                sc += 1
                s = s_prime
                if edone:
                    break
            model.learn(dyna.buffer)
            dyna.train()
            if len(model.buffer)>5000:
                for m in range(nself):
                    data = random.sample(model.buffer, min(epoch,len(model.buffer)))
                    for item in data:
                        s, a, _, _, _ = item
                        s_prime, reward, done = model.step(s,a)
                        dyna.save((s,a,reward,s_prime,done))
                    dyna.train()

        if n%interval ==0 and n!=0:
            print("{} : score:{}".format(n,sc/interval))
            sc = 0.0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi")
    RL(int(n_epi))