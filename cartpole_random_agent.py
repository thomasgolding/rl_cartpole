import gym
import imageio
import argparse
import numpy as np
from agent import Agent


# parser = argparse.ArgumentParser(description='Run agent episode')
# parser.add_argument('-filename')
# parser.add_argument('-not_random', type=int)
# args = vars(parser.parse_args())


do_random = True
# if args['not_random'] == 1:
#     do_random=False

#filename = args['filename']
filename = 'dump.gif'


env = gym.make('CartPole-v0')
action_space = env.action_space
state_dim = env.reset().shape[0]
agent = Agent(action_space = action_space, state_dim = state_dim, \
    qtype='linear', discount = 0.9, epsilon=0.3, alpha = 0.05)

## make one episode
rgb_episode = []
episode = []
action = []
n_episode=100

jcount=0

for i in range(n_episode):
    done = False
    state = env.reset()
    while not done:

        #action = agent.decide_next_action_random()    
        action = agent.decide_next_action_q(state)
        if not do_random:
            action = 0
        [next_state, reward, done, info] = env.step(action)
        
        ## do update ogf Q-model
        [loss, reldiff] = agent.update_q(state = state, previous_action=action, next_state=next_state, reward = reward)
        state = next_state.copy()
        rgb_episode.append(env.render(mode = 'rgb_array'))
        
        ## print message:
        if np.mod(jcount, 10) == 0:
            #msg = 'ep = {:4d}, jcount = {:6d}, loss = {:7.3f}, reldiff = {:7.3f}'.format(i, jcount, loss[0], reldiff[0])
            print(reldiff)
        
        jcount += 1
    
    env.close()

## make gif
imageio.mimsave(filename, rgb_episode)





#image.io




