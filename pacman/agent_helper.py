from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, ReLU, MaxPool2D, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import keras
import numpy as np
from collections import deque
import os
from tensorflow.keras.models import load_model
import random
from tensorflow.keras.losses import MeanSquaredError
import cv2
import pickle
import tensorflow as tf
import csv

class Agent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, optimizer=Adam,
                 epsilon_min =0.1, epsilon_decay=0.995, alpha=0.0001,update_targetnn_rate = 100, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        
        self.optimizer = optimizer
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
        im = cv2.resize(cropped_frame, (100, 100))
        gray_frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return gray_frame / 255 

    def create_neuralnet(self):
        # Create Neural Network
        def model(img_shape):
            input = Input(shape=img_shape)
            
            # Conv Layer 1
            x = Conv2D(filters=64, kernel_size=(5,5), strides=2)(input)
            x = BatchNormalization(axis=-1)(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)  # Adjusted pooling size
            
            # Conv Layer 2
                
            x = Conv2D(filters=128, kernel_size=(3,3), strides=2)(x)
            x = BatchNormalization(axis=-1)(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)  # Adjusted pooling size
            
            # Conv Layer 3
            x = Conv2D(filters=256, kernel_size=(3,3), strides=2,padding="same")(x)
            x = BatchNormalization(axis=-1)(x)
            x = ReLU()(x)
            # FC Layer 
            x = Flatten()(x)
            x = Dense(256, activation="relu")(x)
            x = Dense(128, activation="relu")(x)
            x = Dense(64, activation="relu")(x)
            outputs = Dense(units=self.action_size, activation="linear")(x)
            
            model = keras.Model(inputs=input, outputs=outputs)
            return model
        conv_model = model((100, 100,4))
        conv_model.compile(loss="mse", optimizer=self.optimizer(learning_rate= self.alpha))
        return conv_model
    
    def save_replay_exp(self, current_state, action,reward, next_state, terminal):
        # Save experience to replay buffer for future training
        if current_state.shape != (100, 100, 4):
            print("Fail "+str(current_state.shape))
        self.replay_buffer.append((current_state, action,reward, next_state, terminal))
    
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
            print(state.shape)
            q_values = self.main_nn.predict(tf.expand_dims(state, axis=0), verbose = 0)
            return np.argmax(np.squeeze(q_values))
    
    def log_csv(self, log_file, ep, step, rw):
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep, step, rw])
    def update_target_network(self):
        self.target_nn.set_weights(self.main_nn.get_weights())
        
    def populate_replay_buffer(self, env, stack_size=4):
        epsilon = self.epsilon
        while len(self.replay_buffer) < 50000:
            state,_ = env.reset()
            
            state = self.preprocess(state)
            done = False
            stack = deque([state]*stack_size, maxlen=stack_size)
            state = np.stack(stack, axis=0).astype(np.float32)
            state = tf.transpose(state, [1,2,0])
            while not done:
                if random.uniform(0,1) < epsilon:
                    action = random.choice(range(self.action_size))
                else:
                    q_values = self.main_nn.predict(np.expand_dims(state, axis=0))
                    action = np.argmax(q_values[0])
                next_state, reward, done, _ ,_= env.step(action)
                next_state = self.preprocess(next_state)
                stack.append(next_state)
                next_state = np.stack(stack, axis=0).astype(np.float32)
                next_state = tf.transpose(next_state, [1,2,0])
                if state.shape != (100, 100, 4):
                    print("Fail "+state.shape)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
        if epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def save_checkpoint(self, filename="checkpoint.h5"):
        # Save the primary model and target model together with their optimizer state
        primary_model_path = f"/content/drive/MyDrive/UIT/primary_model_{filename}.h5"
        target_model_path = f"/content/drive/MyDrive/UIT/target_model_{filename}.h5"
        
        self.main_nn.save(primary_model_path)  # Save the entire primary model
        self.target_nn.save(target_model_path)  # Save the entire target model

    def load_checkpoint(self, filename="checkpoint.h5"):
        primary_model_path = f"/content/drive/MyDrive/UIT/primary_model_{filename}"
        target_model_path = f"/content/drive/MyDrive/UIT/target_model_{filename}"
        
        if os.path.exists(primary_model_path) and os.path.exists(target_model_path):
            # Pass custom objects for loss/metrics
            self.main_nn = load_model(primary_model_path, custom_objects={"mse": MeanSquaredError()})
            self.target_nn = load_model(target_model_path, custom_objects={"mse": MeanSquaredError()})
        else:
            print("Checkpoint files not found. Skipping load.")
        
        self.main_nn.trainable = True



