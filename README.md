# Project Name

## Description

This project is a Reinforcement Learning projects. I trained MsPacman Agent to play using Double Deep-Q-Learning and used Python, TensorFlow, Gymnasium, Atari, OpenCV... 

## Installation

Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/tkhangg0910/RL-Pacman-DQN
    ```
2. Navigate to the project directory:
    ```bash
    cd pacman
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project, follow these instructions:

1. At first, you have to install all dependencies
2. agent_helper : Agent class defined for training
   train_pm : Code for training agent
   test_agent : Code for testing agent
    
    If you want to train, you can run the following command:
    ```bash
    python train_pm.py -e NUMBER_OF_EPISODE_YOU_WANNA_TRAIN -s MAX_STEP_FOR_EACH_EP -l[OPTIONAL] PATH_TO_YOUR_CHECKPOINT
    ```


## Testing

To run the tests, use the following command:

```bash
python test_agent.py -m PATH_TO_MODEL_OR_CHECKPOINT 
