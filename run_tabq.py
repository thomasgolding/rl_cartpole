import numpy as np
from agent_tabq import AgentTabQ
from cartpole_env import EnvCartPole
from make_gif import make_rl_gif

RECORD = 25


def play_game(a, record = False):
    envir = EnvCartPole()
    s = envir.reset()
    score = 0
    done = False
    if record:
        image = [envir.env.render(mode = 'rgb_array')]
    while not done:
        action = a.decide_action_deterministic(s)
        [s, r, done, _] = envir.step(action)
        score = score + r
        if record:
            image.append(envir.env.render(mode = 'rgb_array'))

    envir.env.close()

    if record:
        return image
    else:
        return score

def n_timesteps(a):
    envir = EnvCartPole()
    s = envir.reset()
    for it in range(210):
        action = a.decide_action_deterministic(s)
        [s,_,d,_] = envir.step(action)
        if d:
            break
    
    return it


def exp_timestep(a):
    n = 50
    nt = np.zeros(n)
    for i in range(n):
        nt[i] = n_timesteps(a)
    return  nt.mean()

def run_experiment(ntime = 100):
    env = EnvCartPole()
    discount = 1.0

    qdict = {}
    qdict['info'] = []
    agent = AgentTabQ(nstate = env.nstate, naction = env.naction, discount=discount)

    rawim = []
    epim = []
    
    for t in range(ntime):
        s = env.reset()
        done = False
        while not done:
            a = agent.decide_action_explore(s, timestep=t)
            [s2, r, done, info] = env.step(a)
            agent.update_q(state=s, action=a, reward=r, next_state=s2, timestep = t)
            s = s2

        if np.mod(t, 10) == 0:
            exp_n = exp_timestep(agent)
            istring = "episode = {}, exp_ts = {}".format(t, exp_n)
            print(istring)
            qdict[t] = agent.q.copy()
            qdict['info'].append(istring)

        if np.mod(t, RECORD) == 0:
            xim = play_game(agent, record=True)
            xep =[t for _ in range(len(xim))]

            rawim += xim
            epim += xep

    

    return (agent, qdict, rawim, epim)



if __name__ == '__main__':
    [a, q, rawim, epim] = run_experiment(ntime = 355)
    make_rl_gif('rl_cartpole.gif', rawim, epim)

    



            

