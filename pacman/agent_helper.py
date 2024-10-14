from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random
import cv2
import csv

class Agent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, 
                 epsilon_min =0.01, epsilon_decay=0.995, alpha=0.001,update_targetnn_rate = 10, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)
        
        # Agent's Hyperparams
        self.batch_size=batch_size
        self.gamma =gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.update_targetnn_rate = update_targetnn_rate  
        
        # Neural Network for DQN
        self.main_nn = self.create_neuralnet()
        self.target_nn = self.create_neuralnet()
        self.target_nn.set_weights(self.main_nn.get_weights())
        
    def preprocess(self, state):
        # Convert From RGB to GrayScale
        gray_frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Crope unecessary frame
        cropped_frame = gray_frame[:173, :]
        # Resize img
        im = cv2.resize(cropped_frame, (106, 100), interpolation=cv2.INTER_AREA)  
        return im

    def create_neuralnet(self):
        # Create Neural Network with 6 layers
        model = Sequential([
            Flatten(input_shape=(100, 106)),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(self.action_size, activation="linear")  
        ])

        model.compile(loss="mse", optimizer=Adam(learning_rate= self.alpha))
        return model
    
    def save_replay_exp(self, current_state, action,reward, next_state, terminal):
        # Save experience to replay buffer for future training
        self.replay_buffer.append((self.preprocess(current_state), action,reward, self.preprocess(next_state), terminal))
    
    def get_experience(self):
        # Take #batch_size sample from replay buffer by random sampling
        batch_exp =random.sample(self.replay_buffer, self.batch_size)

        # Split each element of batch_exp to an array for training's purpose
        batch_state = np.array([batch[0] for batch in batch_exp])
        batch_action = np.array([batch[1] for batch in batch_exp])
        batch_reward = np.array([batch[2] for batch in batch_exp])
        batch_next_state = np.array([batch[3] for batch in batch_exp])
        batch_terminal = np.array([batch[4] for batch in batch_exp])

        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal

    def train_nn(self):
        # Get a training batch
        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = self.get_experience()
        # Get am initail Q values
        q_values = self.main_nn.predict(batch_state, verbose = 0)
        
        # Get max Q(s', a')
        next_state_q_values = self.target_nn.predict(batch_next_state, verbose = 0)
        max_next_state_q_values = np.amax(next_state_q_values, axis=1)
        
        for i in range(self.batch_size):
            # Get ideal Q value by formula rj + gamma*max Q(s', a')
            q_values[i][batch_action[i]] = batch_reward[i] if batch_terminal[i] else batch_reward[i]+self.gamma*max_next_state_q_values[i]
        
        # Train a main neural network
        self.main_nn.fit(batch_state, q_values, verbose = 0)
        
    def make_decison(self, state):
        # Make decision using epsilon-greedy algorithm
        if random.uniform(0,1) < self.epsilon:
            # Making choice by random
            return random.choice(range(self.action_size))
        else:
            # Making choice by max Q-value
            preprocesstate = self.preprocess(state).reshape(1, 100, 106)
            q_values = self.main_nn.predict(preprocesstate, verbose = 0)
            return np.argmax(q_values[0])
    
    def log_csv(self, log_file, ep, step, rw):
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep, step, rw])

        