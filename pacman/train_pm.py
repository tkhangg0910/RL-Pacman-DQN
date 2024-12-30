import gymnasium as gym
import ale_py
import csv
import argparse
from agent_helper import Agent 
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
import numpy as np
# GPU CONFIG
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set a memory limit (e.g., 2GB out of 4GB available)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]  
        )
        print("GPU memory limited to 3GB.")
        # # Enable dynamic memory growth for each GPU
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # print("Dynamic memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(e)

# Adding arguments
ap= argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=str, default=None, help="Path to load weights from checkpoint")
ap.add_argument("-v", "--verbose", type=int, default=0, help="Set verbosity level (default: 0)")
ap.add_argument("-e", "--num_episode", type=int, required=True, help="Number of training episodes")
ap.add_argument("-s", "--num_step", type=int, required=True, help="Number of steps per episode")

# Parse arguments
args = ap.parse_args()

# Extract the values
checkpoint = args.load
num_episodes = args.num_episode
num_steps = args.num_step
verbose = args.verbose

# Make an environment
env = gym.make('ALE/MsPacman-v5')
obs, info =env.reset()

# Initialize param
state_size = env.observation_space.shape
action_size = env.action_space.n
myAgent = Agent(state_size=state_size, action_size=action_size, batch_size=32)
ep_start = 0
if checkpoint is not None:
    myAgent.load_checkpoint(checkpoint)
    ep_start = int(checkpoint.split("_")[-1].split(".")[0])
total_time_step = 0
log_file = f'training_logs/training_log_{myAgent.batch_size}_e{num_episodes}_s{num_steps}.csv'

myAgent.log_csv(log_file=log_file, ep="Episode", step="ep_step",rw="ep_rewards")
myAgent.populate_replay_buffer(env, stack_size=4)
# Start an episode
for ep in range(ep_start,num_episodes):
    print(f"Start episode: {ep}")
    ep_rewards = 0.0
    obs,_=env.reset()
    obs=myAgent.preprocess(obs)
    stacked_frames = deque([obs for _ in range(4)], maxlen=4)
    state = tf.transpose(np.stack(stacked_frames, axis=0).astype(np.float32),[1,2,0])
    ep_step= 0 
    # Start a step
    for step in range(num_steps):
        total_time_step+=1
        ep_step+=1
        # Make decision
        action = myAgent.make_decison(state)
        next_frame, reward, terminal,_,_= env.step(action)
        # Save experience
        next_frame = myAgent.preprocess(next_frame)
        stacked_frames.append(next_frame)
        next_state = tf.transpose(np.stack(stacked_frames, axis=0).astype(np.float32),[1,2,0])
        myAgent.save_replay_exp(state, action, reward, next_state, terminal)
        if len(myAgent.replay_buffer) > myAgent.batch_size:
            myAgent.train_nn()
        state = next_state
        ep_rewards += float(reward)
        # Update Target NN weight
        if total_time_step% myAgent.update_targetnn_rate == 0:
            myAgent.update_target_network()
        # Check If terminal state
        if terminal:
            # print(f"Terminal state reached at step {step+1}  = ", ep_rewards)
            break
            
        if verbose == 1:
            print(f"Episode: {ep}, step: {step}, reward: {reward}")
        
    # Logging
    myAgent.log_csv(log_file=log_file, ep=ep, step=ep_step,rw=ep_rewards)

    # Epsilon Decay
    if myAgent.epsilon > myAgent.epsilon_min:
        myAgent.epsilon *=myAgent.epsilon_decay
    if ep %10 == 0:
        print(f"End episode: {ep} with episode's reward: {ep_rewards}")
        myAgent.save_checkpoint(filename=f"checkpoint_{ep}")

# Save Agent
myAgent.save_checkpoint(filename=f"checkpoint_{num_episodes}")

# preprocessed_obs = myAgent.preprocess(obs)
# print(preprocessed_obs.shape)

# plt.imshow(obs)
# plt.imshow(preprocessed_obs, cmap='gray')  
# plt.axis('off')  
# plt.show()
