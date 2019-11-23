import numpy as np
from agent_dqn import AgentDQN
import gym
from make_gif import make_rl_gif

RECORD = 25
game_id  = 'Acrobot-v1'



def play_game(a, envir, record = False):
    s = envir.reset()
    score = 0
    done = False
    if record:
        image = [envir.render(mode = 'rgb_array')]
    while not done:
        s = np.expand_dims(s, axis = 0)
        action = a.decide_action_deterministic(s)
        [s, r, done, _] = envir.step(action)
        score = score + r
        if record:
            image.append(envir.render(mode = 'rgb_array'))

    envir.close()

    if record:
        return image
    else:
        return score

def n_timesteps(a, envir):
    s = envir.reset()
    for it in range(210):
        s = np.expand_dims(s, axis = 0)
        action = a.decide_action_deterministic(s)
        [s,_,d,_] = envir.step(action)
        if d:
            break
    
    return it


def exp_timestep(a, envir):
    n = 50
    nt = np.zeros(n)
    for i in range(n):
        nt[i] = n_timesteps(a, envir)
    return  nt.mean()

def train_episode(env, ag, verb=False):
    s = env.reset()
    done = False
    losslist = []
    nt = 0
    while not done:
        a = ag.decide_action_explore(s)
        [s2, r, done, info] = env.step(a)
        #if np.abs(s2[0]) > 1.0:
        #    done = True
        if done:
            r = 10.
        ag.save_timestep_to_memory(state=s, action = a, reward = r, next_state = s2, done=done)
        s = s2.copy()
        nt += 1
        ag.update_qmodel()
        #if verb:
        #    print("action = {}, loss = {}".format(a, l[0]))
        #if (l[0]>1.e4):
        #    return ag
    #return ag, l[0]



env = gym.make(game_id)
agent = AgentDQN(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0], 
                 neurons=[8,16,8], 
                 online_training=True, 
                 discount=0.9, 
                 nupdate_target_qnet=1,
                 memorysize=10000,
                 learning_rate=0.0005,
                 tau = 0.01,
                 ddqn = False)

def run_experiment(nn = 20):
    for ep in range(nn):
        train_episode(env = env, ag = agent)
        exp_nt = exp_timestep(a = agent, envir=env)
        print('ep={}, nt={}'.format(ep, exp_nt))
        if exp_nt > 400.:
            agent.qmodel.save_weights('weights2.hdf5')
            print('agent coverged. weights saved.')
            break





def run_saved_model():
    agent.qmodel.load_weights('weights.hdf5')
    print(exp_timestep(agent, env))

    


