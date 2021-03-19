import gym
import torch
class Model(torch.nn.Module):
    def __init__(self):
        self.buffer = []
        self.gamma = 0.98
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 128)
        self.linear2 = torch.nn.Linear(128, 2)
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
    PG = Model()

    sc = 0.0
    interval = 50
    for n in range(n_epi):
        ob = env.reset()
        done = False
        while not done:
            ob = torch.FloatTensor(ob)
            pi = PG(ob)
            A = torch.distributions.Categorical(pi)
            action = A.sample().item()
            ob, reward, done, tmp = env.step(action)
            PG.save((reward, pi[action]))
            sc += 1
        PG.train()
        
        if n%interval ==0 and n!=0:
            print("{} : score:{}".format(n,sc/interval))
            sc = 0.0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi")
    RL(int(n_epi))