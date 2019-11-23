import pytest
import numpy as np
from agent_dqn import AgentDQN

state_dim = 17
action_dim = 3

@pytest.fixture
def agent():
    a = AgentDQN(state_dim =
     state_dim, action_dim = action_dim)
    return a


def test_update_q(agent):
    s = np.random.normal(size=(1, state_dim))
    action = 0
    reward = 1.

    loss = []
    for it in range(10):
        l = agent.test_update_q(state = s, action = action, 
                            reward = reward, next_state = s)
        loss.append(l)
    
    reldiff1 = (loss[2] - loss[1])/loss[1]
    reldiff2 = (loss[-1] - loss[-2])/loss[-2]


    assert reldiff1 > reldiff2