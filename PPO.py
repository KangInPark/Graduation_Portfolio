import gym
import torch
class Model(torch.nn.Module):
    def __init__(self):
        self.gamma = 0.98
        self.lamb = 0.95
        self.epsilon = 0.1
        self.K = 3
        self.buffer = []
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 128)
        self.pi = torch.nn.Linear(128, 2)
        self.val = torch.nn.Linear(128, 1)      
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
    PPO = Model()
    T = 20
    sc = 0.0
    interval = 50

    for n in range(n_epi):
        s = env.reset()
        done = False
        while not done:
            for step in range(T):
                pi = PPO.pi_forward(torch.FloatTensor(s))
                A = torch.distributions.Categorical(pi)
                action = A.sample().item()
                s_prime, reward, done, tmp = env.step(action)
                PPO.save((s,action,reward,s_prime,pi[action].item(),done))
                s = s_prime
                sc += 1
                if done:
                    break
            PPO.train()

        if n%interval ==0 and n!=0:
            print("{} : score:{}".format(n,sc/interval))
            sc = 0.0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi?")
    RL(int(n_epi))