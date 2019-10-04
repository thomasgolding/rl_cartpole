import pytest
from agent_tabq import AgentTabQ
import numpy as np
import pandas as pd

nstate = 17
naction = 3
discount = 0.9

@pytest.fixture
def agent():
    a = AgentTabQ(nstate = nstate, naction = naction, discount = discount)
    return a

def test_decide_action_random(agent):
    act = [agent.decide_action_random() for _ in range(100)]
    act = np.array(act)
    unact = np.unique(act)
    assert unact.shape[0] == naction

def test_decide_action_deterministic(agent):
    agent.q[0,naction-1]=1.0
    assert agent.decide_action_deterministic(0) == naction -1

def test_decide_action_explore(agent):
    state = 1
    agent.q[1,:] = np.array([0.1,0.2, 0.1])
    act = [agent.decide_action_explore(state = state, timestep = 60) \
           for _ in range(200)]
    s = pd.Series(act)
    freq = s.value_counts(normalize=True)
    assert freq[1] > freq[0]
    assert freq[1] > freq[2]


def test_update_q(agent):
    state = 1
    action = 1
    reward = 1
    next_state=1
    
    q = []

    for timestep in range(10):
        agent.update_q(state = state, 
                            action = action,  
                            reward = reward, 
                            next_state = next_state, 
                            timestep = timestep)
        q.append(agent.q[state,action])
    
    reldiff1 = (q[2]-q[1])/q[1]
    reldiff2 = (q[-2]-q[-1])/q[-1]
    assert reldiff1 > reldiff2


#def 