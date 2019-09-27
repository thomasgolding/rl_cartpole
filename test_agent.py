import pytest
import numpy as np
import pandas as pd 
import gym
from agent import Agent

n_state = 17
n_action = 5
epsilon = 0.05
alpha = 0.01

@pytest.fixture
def agent_req():
    action_space = gym.spaces.Discrete(n_action)   
    agent = Agent(action_space = action_space, state_dim = n_state, 
        epsilon=epsilon, alpha = alpha)
    return agent


def test_construct_stateaction(agent_req):
    ndim = agent_req.state_dim
    ndim_action = agent_req.action_space.n
    assert ndim == n_state
    assert ndim_action == n_action
    
    dum_state = np.zeros((ndim))
    for a in range(ndim_action):
        sa0 = agent_req.construct_stateaction(state=dum_state,action=a)
        assert sa0[n_state + a] == 1


def test_decide_next_action_q(agent_req):
    state = np.ones(n_state)

    experiments = [(False, False, 3.0), (True, False, 0.01)]
    for (deterministic, old, thresh) in experiments:
        a = []
        for _ in range(50):
            a.append(agent_req.decide_next_action_q(state, deterministic = deterministic, old=old))
    
        s = pd.Series(a)
        s = s.value_counts(normalize=True,  sort=True)

        errmsg = "problem: epsilon, deterministic = {}, old = {}"
        errmsg = errmsg.format(deterministic, old)
        assert s.iloc[0] > (1 - thresh*epsilon), errmsg

    

def test_compute_q(agent_req):
    state = np.ones(n_state)
    q1 = agent_req.compute_q(state, 0, old=False)['q']
    q2 = agent_req.compute_q(state, 0, old=True)['q']
    assert np.abs(q1-q2) > 1.e-6


def test_update_q(agent_req):
    s1 = np.random.normal(size = n_state)
    s2 = np.random.normal(size = n_state)
    a1 = np.random.randint(0, high = n_action)

         
    res=[]
    nn=30
    for i in range(nn):
        [loss, max_diff] = agent_req.update_q(state = s1, previous_action = a1, next_state = s2, reward=1)
        if i == nn-2:
            res.append([loss, max_diff])
        if i == nn-1:
            res.append([loss, max_diff])   

    assert res[0][0] > res[1][0], "Loss function not decreasing"
    assert res[0][1] > res[1][1], "Corrections not decreasing"   





