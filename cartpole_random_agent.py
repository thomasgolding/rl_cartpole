import gym
import imageio
import argparse
import pickle
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


env = gym.make('CartPole-v1')
action_space = env.action_space
state_dim = env.reset().shape[0]
agent = Agent(action_space = action_space, state_dim = state_dim, \
    qtype='linear', discount = 0.9, epsilon=0.3, alpha = 0.05)

## make one episode
rgb_episode = []
episode = []
action = []
n_episode=300      

jcount=0

for i in range(n_episode):
    print(i)
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
        
        if np.mod(i, 20) == 0:
            rgb_episode.append(env.render(mode = 'rgb_array'))
            episode.append(i)

        ## print message:
        if np.mod(jcount, 10) == 0:
            #msg = 'ep = {}, jcount = {}, loss = {}, reldiff = {}'.format(i, jcount, loss[0], reldiff[0])
            print(loss)
        
        jcount += 1
    
env.close()

## make gif

data = {
    'rgb_images': rgb_episode,
    'episode': episode,
    'agent': agent,
    'env': env
    }
with open('res.pickle', 'wb') as f:
    pickle.dump(data, f)



## imageio.mimsave(filename, rgb_episode)






#image.io




