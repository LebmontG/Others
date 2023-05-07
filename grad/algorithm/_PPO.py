import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_sigma = nn.Linear(hidden_dim, action_dim)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc_mu.weight)
        nn.init.normal_(self.fc_sigma.weight)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # action_prob = F.softmax(self.fc3(x), dim=-1)
        mu=torch.tanh(self.fc_mu(x))
        sigma=F.softplus(self.fc_sigma(x))+0.0001
        return mu,sigma

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc3.weight)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPO(object):
    def __init__(self, state_dim, action_dim,
                  hidden_dim=12, lr_actor=0.0003, lr_critic=0.00003,
                  gamma=0.99, clip_ratio=0.01, beta=0.005):
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.optimizer_actor = optim.SGD(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.SGD(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.beta = beta
        return
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # Compute the returns
        returns = torch.zeros_like(rewards)
        next_value = 0
        for t in reversed(range(rewards.size(0))):
            returns[t] = rewards[t] + self.gamma * next_value * masks[t]
            next_value = values[t]
        # Compute the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    def update(self, states: torch.Tensor, actions: torch.Tensor, log_probs_old: torch.Tensor,
               returns: torch.Tensor, advantages: torch.Tensor) -> tuple:
        # Compute actor loss
        mean, std = self.actor(states)
        dist = torch.distributions.normal.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        ratios = torch.exp(log_probs - log_probs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.beta * dist.entropy().mean()
        # Compute critic loss
        value_pred = self.critic(states).squeeze()
        critic_loss = F.mse_loss(value_pred, returns)
        # Update actor and critic networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        # print(actor_loss.item(), critic_loss.item())
        return actor_loss.item(), critic_loss.item()
    def train(self, env,episode,rf=None,batch_size=100):
        steps=[];upd_i=0
        states, actions, rewards, log_probs, masks = [], [], [], [], []
        for _ in range(episode):
            state = env.reset();done=False;i=0
            # if rf.style=="AWSL":rf.reset()
            while not done:
                if upd_i==batch_size:
                    state_tensor = torch.tensor(states).float().to(self.device)
                    action_tensor = torch.tensor(actions).float().to(self.device)
                    log_prob_tensor = torch.stack(log_probs).float().to(self.device)
                    reward_tensor = torch.tensor(rewards).float().to(self.device).unsqueeze(1)
                    mask_tensor = torch.tensor(masks).float().to(self.device).unsqueeze(1)
                    value_tensor = self.critic(state_tensor).detach()
                    advantages, returns = self.compute_advantages(reward_tensor, value_tensor, mask_tensor)
                    actor_losses, critic_losses = [], []
                    for _ in range(10):
                        actor_loss, critic_loss = self.update(state_tensor, action_tensor, log_prob_tensor, returns, advantages)
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
                    states, actions, rewards, log_probs, masks = [], [], [], [], []
                    upd_i=0
                else:upd_i+=1
                with torch.no_grad():
                    mu,sigma = self.actor(torch.tensor(state).float().to(self.device))
                    # action = torch.multinomial(action_prob, 1).item()
                    dist=torch.distributions.normal.Normal(mu,sigma)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).to(self.device)
                    action=np.clip(action.cpu().detach().numpy(),-1,1)
                next_state, reward, done, _ = env.step(action)
                if rf!=None:
                    if rf.style=="AWSL":
                        rf.step(state,reward)
                        reward*=rf.W(state)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                masks.append(not done)
                state = next_state
                if i>=200:print("done!!!!");break
                else:i+=1
            steps.append(i)
        return steps

if __name__ =="__main__":
    scl=6
    agent=PPO(4,1)
    s,a=np.random.rand(4),np.random.rand(1)
    #a=agent.predict(s)
    print(a)