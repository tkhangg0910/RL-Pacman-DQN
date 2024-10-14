import gymnasium as gym
import ale_py
import csv
import argparse
from agent_helper import Agent 
import matplotlib.pyplot as plt

# Adding arguments
ap= argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", type=int, default=0, help="Set verbosity level (default: 0)")
ap.add_argument("-e", "--num_episode", type=int, required=True, help="Number of training episodes")
ap.add_argument("-s", "--num_step", type=int, required=True, help="Number of steps per episode")

# Parse arguments
args = ap.parse_args()

# Extract the values
num_episodes = args.num_episode
num_steps = args.num_step
verbose = args.verbose

# Make an environment
env = gym.make('ALE/MsPacman-v5')
obs, info =env.reset()

# Initialize param
state_size = env.observation_space.shape
action_size = env.action_space.n
myAgent = Agent(state_size=state_size, action_size=action_size)
total_time_step = 0
log_file = f'pacman/training_logs/training_log_{myAgent.batch_size}_e{num_episodes}_s{num_steps}.csv'

myAgent.log_csv(log_file=log_file, ep="Episode", step="ep_step",rw="ep_rewards")

# Start an episode
for ep in range(num_episodes):
    print(f"Start episode: {ep}")
    ep_rewards = 0.0
    obs,_=env.reset()
    ep_step= 0 
    # Start a step
    for step in range(num_steps):
        total_time_step+=1
        ep_step+=1
        # Update Target NN weight
        if total_time_step% myAgent.update_targetnn_rate == 0:
            myAgent.target_nn.set_weights(myAgent.main_nn.get_weights())
        # Make decision
        action = myAgent.make_decison(obs)
        next_state, reward, terminal,_,_= env.step(action)
        # Save experience
        myAgent.save_replay_exp(obs, action, reward, next_state, terminal)

        state = next_state
        ep_rewards += float(reward)
        # Check If terminal state
        if terminal:
            # print(f"Terminal state reached at step {step+1}  = ", ep_rewards)
            break
        # Train MainNN if it's enough batch
        if len(myAgent.replay_buffer) > myAgent.batch_size:
            myAgent.train_nn()
            
        if verbose == 1:
            print(f"Episode: {ep}, step: {step}, reward: {reward}")
        
    # Logging
    myAgent.log_csv(log_file=log_file, ep=ep, step=ep_step,rw=ep_rewards)

    # Epsilon Decay
    if myAgent.epsilon > myAgent.epsilon_min:
        myAgent.epsilon *=myAgent.epsilon_decay
    if ep %10 == 0:
        print(f"End episode: {ep} with episode's reward: {ep_rewards}")

# Save Agent
myAgent.main_nn.save(f"pacman/models/train_agent_{myAgent.batch_size}_e{num_episodes}_s{num_steps}.keras")

# preprocessed_obs = myAgent.preprocess(obs)
# plt.imshow(obs)
# plt.imshow(preprocessed_obs, cmap='gray')  
# plt.axis('off')  
# plt.show()