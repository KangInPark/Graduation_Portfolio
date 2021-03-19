import gym
import torch
import random
import collections

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

class Model(torch.nn.Module):
    def __init__(self):
        self.gamma = 0.98
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 2)     

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
        td = r + DQN.gamma * max_q_prime * done
        loss = torch.nn.functional.smooth_l1_loss(td, q_a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def RL(n_epi): 
    """
    PG = 0 : reinforce
    PG = 1 : Q actor-critic with TD(0)
    PG = 2 : Advantage actor-critic with TD
    """
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
    DQN = Model()
    target = Model()
    target.load_state_dict(DQN.state_dict())
    buffer = Experience_replay()
    optimizer = torch.optim.Adam(DQN.parameters(), lr = 0.001)

    sc = 0.0
    interval = 50

    for n in range(n_epi):
        epsilon = max(0.01, 0.1 - 0.01 * n/100)
        s = env.reset()
        done = False
        while not done:
            action = DQN.action(torch.FloatTensor(s), epsilon)
            s_prime, reward, done, tmp = env.step(action)
            buffer.insert((s,action,reward,s_prime,not(done)))
            s = s_prime
            sc += 1
            if done:
                break
        train(DQN, target, buffer, optimizer)
        if n%5 ==0 and n !=0:
            target.load_state_dict(DQN.state_dict())
        if n%interval ==0 and n!=0:
            print("{} : score:{}".format(n,sc/interval))
            sc = 0.0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi?")
    RL(int(n_epi))