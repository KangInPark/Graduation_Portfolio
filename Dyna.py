import gym
import torch
import random
import collections

class Model(torch.nn.Module):
    def __init__(self):
        self.buffer = collections.deque(maxlen=10000)
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(5,256)
        self.linear2 = torch.nn.Linear(256,256)
        self.s_prime = torch.nn.Linear(256,4)
        self.reward = torch.nn.Linear(256,1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

    def s_forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        s_prime = self.s_prime(x)
        return s_prime
    
    def reward_forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        reward = self.reward(x)
        return reward

    def step(self, s, a):
        input = torch.cat((s,a), dim = 1)
        reward = self.reward_forward(input)
        s_prime = self.s_forward(input)
        return s_prime, reward

    def learn(self, buffer):
        sl, al, rl, spl, dl = [], [], [], [], []
        for item in buffer:
            self.buffer.append(item)
            s, a, r, sp, d = item
            sl.append(s)
            al.append([a])
            rl.append([r])
            spl.append(sp)
            dl.append([not(d)])
        s, action, reward, s_prime, done = torch.FloatTensor(sl), torch.tensor(al), torch.FloatTensor(rl), torch.FloatTensor(spl), torch.FloatTensor(dl)

        self.optimizer.zero_grad()
        predict_s_prime, predict_reward = self.step(s,action)
        
        s_loss = torch.nn.functional.smooth_l1_loss(s_prime, predict_s_prime)
        r_loss = torch.nn.functional.smooth_l1_loss(reward, predict_reward)

        loss = torch.mean(s_loss + r_loss)
        loss.backward()
        self.optimizer.step()

class Dyna(torch.nn.Module):
    def __init__(self):
        self.gamma = 0.98
        self.buffer = []
        super(Dyna, self).__init__()
        self.linear1 = torch.nn.Linear(4, 128)
        self.linear2 = torch.nn.Linear(128, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
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
            rl.append([r])
            spl.append(sp)
            dl.append([not(d)])
        s, action, reward, s_prime, done = torch.FloatTensor(sl), torch.tensor(al), torch.FloatTensor(rl), torch.FloatTensor(spl), torch.FloatTensor(dl)

        self.optimizer.zero_grad()
        q = self.forward(s)
        q_a = q.gather(1,action)
        max_q_prime = self.forward(s_prime).max(1)[0].unsqueeze(1)
        td = reward + self.gamma * max_q_prime
        loss = torch.nn.functional.smooth_l1_loss(td, q_a)
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
        done = False
        while not done:
            for step in range(epoch):
                action = dyna.action(torch.FloatTensor(s), epsilon)
                s_prime, reward, done, tmp = env.step(action)
                dyna.save((s, action, reward, s_prime, done))
                sc += 1
                s = s_prime
                if done:
                    break
            model.learn(dyna.buffer)
            dyna.train()
            for m in range(nself):
                '''
                if m == 0:
                    test = random.sample(model.buffer, 1)
                    s, action, reward, s_prime, _ = test[0]
                    t_s, t_r = model.step(torch.FloatTensor(s).reshape(1,-1),torch.tensor(action).reshape(1,-1))
                    print(t_s[0].detach().numpy()-s_prime, reward- t_r.item())
                '''
                data = random.sample(model.buffer, min(epoch,len(model.buffer)))
                for item in data:
                    s, a, _, _, _ = item
                    s_prime, reward = model.step(torch.FloatTensor(s).reshape(1,-1),torch.tensor(a).reshape(1,-1))
                    dyna.save((s,a,reward.item(),s_prime[0].detach().numpy(),0))
                dyna.train()
        if n%interval ==0 and n!=0:
            print("{} : score:{}".format(n,sc/interval))
            sc = 0.0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi")
    RL(int(n_epi))