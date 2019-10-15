import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6,solution=1,alpha=0.02,gamma=1.0, eps=0.01):
        """ Initialize agent.0

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.solution = solution
        self.gamma=gamma
        self.alpha =alpha
        self.eps = eps
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.
        self.
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps:  # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:  # otherwise, select an action randomly
            return random.choice( np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        if self.solution ==1 :
            #first solution
            self.Q[state][action] = self.Q[state][action] +\
               self.alpha * (reward + self.gamma *  np.max(self.Q[next_state])  - self.Q[state][action])

        if self.solution== 2:
            # second solution
            next_action = self.select_action( next_state)
            self.Q[state][action] = self.Q[state][action] + \
                                   self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

        if self.solution==3:
            #third solution
            policy_s = np.ones(self.nA) * (self.eps / self.nA)
            policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
            Qss = np.dot(self.Q[next_state], policy_s)
            self.Q[state][action] = self.Q[state][action] + \
                                       self.alpha * (reward + self.gamma * Qss- self.Q[state][action])

