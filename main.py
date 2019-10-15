from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
alpha = 0.9999
gamma = .9
epsilon = 0.0001
print("Gym Taxi version 3 -- my solutions")
print("Alpha=",alpha)
print("gamma=",gamma)
print("epsilon=",epsilon)

print("----Sarsa Algorithm ")
agent = Agent(solution=1,alpha=alpha,gamma=gamma,eps=epsilon)
avg_rewards, best_avg_reward = interact(env, agent)

print("----SarsaMax or Q-learning Algorithm ")
agent = Agent(solution=2,alpha=alpha,gamma=gamma,eps=epsilon)
avg_rewards, best_avg_reward = interact(env, agent)

print("----Extended Sarsa Algorithm ")
agent = Agent(solution=1,alpha=alpha,gamma=gamma,eps=epsilon)
avg_rewards, best_avg_reward = interact(env, agent)
