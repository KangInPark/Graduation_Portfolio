import gym
import torch
class Model(torch.nn.Module):
    def __init__(self):
        self.gamma = 0.98
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 128)
        self.pi = torch.nn.Linear(128, 2)
        self.val = torch.nn.Linear(128, 1)      
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)

    def pi_forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        pi = self.pi(x)
        pi = torch.nn.functional.softmax(pi, dim=0)
        return pi

    def val_forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        val = self.val(x)
        return val

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
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
    PG = Model()

    sc = 0.0
    interval = 50
    for n in range(n_epi):
        s = env.reset()
        done = False
        while not done:
            pi = PG.pi_forward(torch.FloatTensor(s))
            A = torch.distributions.Categorical(pi)
            action = A.sample()
            s_prime, reward, done, tmp = env.step(action.item())
            val = PG.val_forward(torch.FloatTensor(s))
            val_prime = PG.val_forward(torch.FloatTensor(s_prime))
            td = reward + PG.gamma * val_prime
            delta =  td - val
            loss = -torch.log(pi[action])*delta  + torch.nn.functional.smooth_l1_loss(td, val)
            PG.train(loss)
            s = s_prime
            sc += 1

        if n%interval ==0 and n!=0:
            print("{} : score:{}".format(n,sc/interval))
            sc = 0.0
    env.close()
        
if __name__ == '__main__':
    n_epi = input("n_epi")
    RL(int(n_epi))