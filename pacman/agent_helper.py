from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, ReLU, MaxPool2D, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import keras
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
        cropped_frame = state[:173, :]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(gray_frame, (120, 120), interpolation=cv2.INTER_AREA)
        return im.reshape(120, 120, 1) / 255 

    def create_neuralnet(self):
        # Create Neural Network
        def model(img_shape):
            input = Input(shape=img_shape)
            # Conv Layer 1
            x = Conv2D(filters=64, kernel_size=(5,5), strides = 2)(input)
            x = BatchNormalization(axis = 3)(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(3, 3),strides=(2,2))(x)
            
            # Conv Layer 2
            x = Conv2D(filters=128, kernel_size=(3,3), strides = 2)(x)
            x = BatchNormalization(axis = 3)(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(3, 3),strides=(2,2))(x)
            
            # Conv Layer 3
            x = Conv2D(filters=256, kernel_size=(3,3), strides = 2)(x)
            x = BatchNormalization(axis = 3)(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(3, 3),padding="same")(x)
            
            # FC layer 
            x = Flatten()(x)
            x = Dense(512, activation="relu")(x)
            x = Dense(128, activation="relu")(x)
            outputs = Dense(units = self.action_size, activation="linear")(x)
            
            model = keras.Model(inputs=input, outputs=outputs)
            return model
        conv_model = model((120, 120, 1))
        conv_model.compile(loss="mse", optimizer=Adam(learning_rate= self.alpha))
        return conv_model
    
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
            # Get ideal Q value using the formula: rj + gamma * max Q(s', a')
            if batch_terminal[i]:
                q_values[i][batch_action[i]] = batch_reward[i]
            else:
                q_values[i][batch_action[i]] = batch_reward[i] + self.gamma * max_next_state_q_values[i]
        
        # Train a main neural network
        self.main_nn.fit(batch_state, q_values, verbose = 0)
        
    def make_decison(self, state):
        # Make decision using epsilon-greedy algorithm
        if random.uniform(0,1) < self.epsilon:
            # Making choice by random
            return random.choice(range(self.action_size))
        else:
            # Making choice by max Q-value
            preprocessed_state = self.preprocess(state).reshape(1, 120, 120, 1)
            q_values = self.main_nn.predict(preprocessed_state, verbose = 0)
            return np.argmax(q_values[0])
    
    def log_csv(self, log_file, ep, step, rw):
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep, step, rw])
    def update_target_network(self):
        self.target_nn.set_weights(self.main_nn.get_weights())
        
