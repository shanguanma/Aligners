#!/usr/bin/env python3

# modified from https://github.com/Eilene/HMM-python/blob/master/HMM.py
# reference : https://www.cnblogs.com/gongyanzh/p/12878375.html
# 2020-4-3
"""
1. implement forward and backward algorithm on the HMM model
1a. The input is  state transition matrix, observation maxtrix, initial state probability vector,
    observation sequence.
2. implement veterbi algorithm(find the most probable state sequence under the current observation sequence)
"""

import numpy as np

class HMM:
    def __init__(
        self,
        transition_matrix,
        observation_matrix,
        initial_vector,
        observation_vector,
    ):
        self.transition_matrix = transition_matrix # N x T , N is number of states, T is time axis
        self.observation_matrix = observation_matrix # N x K
        self.initial_vector = initial_vector # N
        self.observation_vector = observation_vector # T
    ##### veterbi algorithm
    def hmm_veterbi(self):
        time_axis = len(self.observation_vector) # T ,
        states = np.shape(self.transition_matrix)[0] # N
        
        ##### step1 init
        detal = [[0] * states for _ in range(time_axis)] # it is list,  elements of the list is time_axis, 
                                                         # per element is also list, it length is states
        psi = [[0] * states for _ in range(time_axis)] # the biggest probability path

        # first time
        for i in range(states):
            detal[0][i] = self.initial_vector[i] * self.observation_matrix[i][self.observation_vector[0]]
            psi[0][i] = 0
        
        ##### step2 iter
        for t in range(1, time_axis):
            for i in range(states):
                temp, maxindex = 0, 0
                for j in range(states):
                    res = detal[t - 1][j] * self.transition_matrix[j][i]
                    if res > temp:
                        temp = res
                        maxindex = j
                detal[t][i] = temp * self.observation_matrix[i][self.observation_vector[t]]
                psi[t][i] = maxindex
        #### step3 the last state of getting max probality
        p = max(detal[-1]) # it is a value
        for i in range(states):
            if detal[-1][i] == p:
                i_T = i
        #### step4 backtrack
        path = [0] * time_axis
        i_t = i_T
        for t in reversed(range(time_axis - 1)):
            i_t = psi[t+1][i_t]
            path[t] = i_t
        path[-1] = i_T
        return detal, psi, path


    #### hmm forward backward algorithm, it is probability compute problem, in other words, 
    #### it wants to get probability of observation sequences under the hmm model and observation sequences 
    ### reference: https://www.cnblogs.com/gongyanzh/p/12880387.html
    def hmm_forward(self,):
        time_axis = len(self.observation_vector)
        states = np.shape(self.transition_matrix)[0]
        # step1 init
        alpha = [[0] * time_axis for _ in range(states)]  # N x T
        for i in range(states):
            alpha[i][0] = self.initial_vector[i] * self.observation_matrix[i][self.observation_vector[0]]

        # step2 compute alpha, it is difficut.
        for t in range(1, time_axis):
            for i in range(states):
                temp = 0
                for j in range(states):
                    temp += alpha[j][t-1] * self.transition_matrix[j][i]
                alpha[i][t] = temp * self.observation_matrix[i][self.observation_vector[t]]

        # step3 compute alpha(T)
        prob = 0
        for i in range(states):
            prob += alpha[i][-1]
        return prob, alpha
    def hmm_backward(self,):
        time_axis = len(self.observation_vector)
        states = np.shape(self.transition_matrix)[0]
        

        # step1 init, the last time case ,t = T
        beta = [[0] * time_axis for _ in range(states)] # N x T
        for i in range(states):
            beta[i][-1] = 1
        
        # step2 compute beta(t) the medium case, t = 1~ T-1
        for t in reversed(range(time_axis -1)):
            for i in range(states):
                for j in range(states):
                    beta[i][t] += self.transition_matrix[i][j] * self.observation_matrix[i][self.observation_vector[t+1]] * beta[i][t+1]

        # step3 compute beta(0) the first time case, t=0
        prob=0
        for i in range(states):
            prob += self.initial_vector[i] * self.observation_matrix[i][self.observation_vector[0]] * beta[i][0]
        return prob, beta

    def hmm_gamma(self,alpha, beta):
        time_axis = len(self.observation_vector)
        states = np.shape(self.transition_matrix)[0]
        gamma = [[0] * time_axis for _ in range(states)] # N x T
        for t in range(time_axis):
            for i in range(states):
                gamma[i][t] += alpha[i][t] * beta[i][t] / sum(alpha[j][t] * beta[j][t] for j in range(states))
        return gamma


if __name__ == "__main__":
    transition_matrix=np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]]) # 
    observation_matrix=np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    initial_vector=np.array([0.2,0.4,0.4])
    observation_vector = np.array([0,1,0,1])
    hmm = HMM(
        transition_matrix = transition_matrix,
        observation_matrix = observation_matrix,
        initial_vector = initial_vector,
        observation_vector = observation_vector,       
    )
    detal, psi, path = hmm.hmm_veterbi()
    print(f"hmm_veterbi detal is {detal}")
    print(f"hmm_veterbi psi is {psi}")
    print(f"hmm_veterbi path is {path}")
    prob_forward, alpha = hmm.hmm_forward()
    print(f"hmm_forward prob is {prob_forward}")
    print(f"hmm_forward alpha is {alpha}")
    prob_backward, beta = hmm.hmm_backward()
    print(f"hmm_backward prob is {prob_backward}")
    print(f"hmm_backward beta is {beta}")
    gamma = hmm.hmm_gamma(alpha,beta)
    print(f"hmm_gmma gamma is {gamma}")

