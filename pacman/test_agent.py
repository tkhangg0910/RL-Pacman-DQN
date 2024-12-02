import numpy as np  
import gymnasium as gym
from keras.models import load_model
from keras.metrics import MeanSquaredError
import cv2
import ale_py
import argparse

def preprocess( state):
    # Convert From RGB to GrayScale
    cropped_frame = state[:173, :]
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(gray_frame, (120, 120), interpolation=cv2.INTER_AREA)
    return im.reshape(120, 120, 1) / 255 

# Adding arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="Model's destination", required=True)
args = vars(ap.parse_args())
# Parse arguments
model = args['model']

# Make an environment
env = gym.make('ALE/MsPacman-v5', render_mode="human")
state, info = env.reset()
state = preprocess(state)
# Load model
my_agent = load_model(model, custom_objects={'mse': MeanSquaredError()})
n_steps = 10000
total_reward = 0

for t in range(n_steps):
    env.render()
    state = state.reshape(1, 120, 120, 1)
    q_values = my_agent.predict(state, verbose=0)
    max_q_values = np.argmax(q_values[0])  
    print(f"Step: {t+1}, Action: {max_q_values}")
    # Do action and receive a result
    next_state, reward, terminal, truncated, info = env.step(max_q_values)
    total_reward += float(reward)
    
    # Update the state for the next iteration
    state = preprocess(next_state)
    
    # Exit if the game has ended
    if terminal or truncated:
        print(f"Total reward: {total_reward}")
        break

env.close()
