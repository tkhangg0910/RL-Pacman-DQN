import numpy as np  
import gymnasium as gym
from keras.models import load_model
from keras.metrics import MeanSquaredError
import cv2
import ale_py
from collections import deque
import argparse
import tensorflow as tf
import numpy as np
from agent_helper import Agent 

# Adding arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="Model's destination", required=True)
args = vars(ap.parse_args())
model = args['model']

# Make an environment
env = gym.make('ALE/MsPacman-v5', render_mode="human")
state, info = env.reset()
# Load model
state_size = env.observation_space.shape
action_size = env.action_space.n
my_agent = Agent(state_size, action_size, batch_size=32)
my_agent.load_checkpoint(model)
my_agent.main_nn.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
n_steps = 10000
total_reward = 0

state = my_agent.preprocess(state)
stack_frame = deque([state]*4, maxlen=4)
state = np.stack(stack_frame, axis=0).astype(np.float32)
state = tf.transpose(state, [1,2,0])
state = np.expand_dims(state, axis=0)
for t in range(n_steps):
    q_values = my_agent.main_nn.predict(state, verbose=0)
    print(q_values)
    max_q_values = np.argmax(q_values[0])  
    print(f"Step: {t+1}, Action: {max_q_values}")
    # Do action and receive a result
    next_state, reward, terminal, truncated, info = env.step(max_q_values)
    env.render()
    next_state = my_agent.preprocess(next_state)
    stack_frame.append(next_state)
    next_state= np.stack(stack_frame, axis=0).astype(np.float32)
    state = tf.transpose(next_state, [1,2,0])
    state = np.expand_dims(state, axis=0)
    total_reward += float(reward)    
    # Exit if the game has ended
    if terminal or truncated:
        print(f"Total reward: {total_reward}")
        break

env.close()
