
import numpy as np
from sklearn import metrics

#from sklearn.linear_model import LogisticRegression
# depending on the classification model use, we might need to import other packages
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from tqdm import tqdm
import time


from envs import LalEnvTargetAccuracy

from helpers import Minibatch, ReplayBuffer
from dqn import DQN
from Test_AL import policy_rl
from datasets import MNIST_train, MNIST_test

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from tqdm import tqdm
import time

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    print("Name:", gpu.name, "  Type:", gpu.device_type)

N_STATE_ESTIMATION = 30
SIZE = 100
# if we want to train and test RL on the same dataset, use even and odd datapoints for training and testing correspondingly
SUBSET = -1 # -1 for using all datapoints, 0 for even, 1 for odd
N_JOBS = 1 # can set more if we want to parallelise

# The quality is measures according to a given quality measure `quality_method`. 
QUALITY_METHOD = metrics.accuracy_score
# The `tolerance_level` is the proportion of max quality that needs to be achived in order to terminate an episode. 
TOLERANCE_LEVEL = 0.98

dataset = MNIST_train(n_state_estimation=N_STATE_ESTIMATION, subset=SUBSET, size=SIZE)
dataset_test = MNIST_test(n_state_estimation=N_STATE_ESTIMATION, subset=1, size=SIZE)


# Classifier
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

env = LalEnvTargetAccuracy(dataset, model, quality_method=QUALITY_METHOD, tolerance_level=TOLERANCE_LEVEL)
env_test = LalEnvTargetAccuracy(dataset_test, model, quality_method=QUALITY_METHOD, tolerance_level=TOLERANCE_LEVEL)



""" Parameters for training RL """
DIRNAME = './agents/cnn_mnist' # The resulting agent of this experiment will be written in a file

# Replay buffer parameters.
REPLAY_BUFFER_SIZE = 1e4
PRIOROTIZED_REPLAY_EXPONENT = 3

# Agent parameters.
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TARGET_COPY_FACTOR = 0.01
BIAS_INITIALIZATION = 0 # default 0 # will be set to minus half of average duration during warm start experiemnts

# Warm start parameters.
WARM_START_EPISODES = 128 # reduce for test
NN_UPDATES_PER_WARM_START = 100

# Episode simulation parameters.
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_STEPS = 1000

# Training parameters
TRAINING_ITERATIONS = 1000 # reduce for test
TRAINING_EPISODES_PER_ITERATION = 10 # at each training ietration x episodes are simulated
NN_UPDATES_PER_ITERATION = 60 # at each training iteration x gradient steps are made

# Validation and test parameters
N_VALIDATION = 500 # reduce for test
N_TEST = 500 # reduce for test
VALIDATION_TEST_FREQUENCY = 100 # every x iterations val and test are performed

replay_buffer = ReplayBuffer(buffer_size=REPLAY_BUFFER_SIZE, 
                             prior_exp=PRIOROTIZED_REPLAY_EXPONENT)

def reset_weights(model):
    """Initialize weights of Neural Networks
    """
    session = keras.backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


f = open("./training.txt", mode='at', encoding='utf-8')
f.writelines("### Collecting eposodes\n")
f.close()

print("### Collecting eposodes")
# Keep track of episode duration to compute average
episode_durations = []

for _ in tqdm(range(WARM_START_EPISODES)):
    time.sleep(0.1)
    print('.', end='')
    #reset_weights(model)
    
    # Reset the environment to start a new episode
    # classifier_state contains vector representation of state of the environment (depends on classifier)
    # next_action_state contains vector representations of all actions available to be taken at the next step
    classifier_state, next_action_state = env.reset(n_start=10)
    terminal = False
    episode_duration = 0
    # before we reach a terminal state, make steps
    while not terminal:
        # Choose a random action
        action = np.random.randint(0, env.n_actions)
        # taken_action_state is a vector corresponding to a taken action
        taken_action_state = next_action_state[:,action]
        next_classifier_state, next_action_state, reward, terminal = env.step(action)
        # Store the transition in the replay buffer
        replay_buffer.store_transition(classifier_state, 
                                       taken_action_state, 
                                       reward, next_classifier_state, 
                                       next_action_state, terminal)
        # Get ready for next step
        classifier_state = next_classifier_state
        episode_duration += 1 
    episode_durations.append(episode_duration)
# compute the average episode duration of episodes generated during the warm start procedure
av_episode_duration = np.mean(episode_durations)
print('Average episode duration = ', av_episode_duration)

BIAS_INITIALIZATION = -av_episode_duration/2


agent = DQN(experiment_dir=DIRNAME,
            observation_length=N_STATE_ESTIMATION,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            target_copy_factor=TARGET_COPY_FACTOR,
            bias_average=BIAS_INITIALIZATION)


for _ in range(NN_UPDATES_PER_WARM_START):
    print('.', end='')
    # Sample a batch from the replay buffer proportionally to the probability of sampling.
    minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
    # Use batch to train an agent. Keep track of temporal difference errors during training.
    td_error = agent.train(minibatch)
    # Update probabilities of sampling each datapoint proportionally to the error.
    replay_buffer.update_td_errors(td_error, minibatch.indeces)

f = open("./training.txt", mode='at', encoding='utf-8')
f.writelines("## Updating replay buffer done\n")
f.close()
print("## Updating replay buffer done")

train_episode_rewards = []

i_episode = 0
for iteration in range(TRAINING_ITERATIONS):

    # GENERATE NEW EPISODES
    # Compute epsilon value according to the schedule.
    epsilon = max(EPSILON_END, EPSILON_START-iteration*(EPSILON_START-EPSILON_END)/EPSILON_STEPS)
    print(iteration, end=': ')
    f = open("./training.txt", mode='at', encoding='utf-8')
    f.write("Training iter:{}".format(iteration))
    f.close()
    # Simulate training episodes.
    
    for l in range(TRAINING_EPISODES_PER_ITERATION):
        # Reset the environment to start a new episode.
        classifier_state, next_action_state = env.reset()
        print(".", end='')

        #f = open("./training.txt", mode='at', encoding='utf-8')
        #f.write(".")
        #f.close()

        terminal = False
        # Keep track of stats of episode to analyse it in tensorboard.
        episode_reward = 0
        episode_duration = 0
        episode_summary = tf.compat.v1.Summary()
        # Run an episode.

        max_iter = 1000
        i = 0
        while not terminal:
            # Let an agent choose an action.
            action = agent.get_action(classifier_state, next_action_state)
            # Get a prob of a datapoint corresponding to an action chosen by an agent.
            # It is needed just for the tensorboard analysis.
            rlchosen_action_state = next_action_state[0,action]
            
            # With epsilon probability, take a random action.
            if np.random.ranf() < epsilon: 
                action = np.random.randint(0, env.n_actions)
            # taken_action_state is a vector that corresponds to a taken action
            taken_action_state = next_action_state[:,action]
            # Make another step.
            next_classifier_state, next_action_state, reward, terminal = env.step(action)
            # Store a step in replay buffer
            replay_buffer.store_transition(classifier_state, 
                                           taken_action_state, 
                                           reward, 
                                           next_classifier_state, 
                                           next_action_state, 
                                           terminal)
            # Change a state of environment.
            classifier_state = next_classifier_state
            # Keep track of stats and add summaries to tensorboard.
            print(reward)
            episode_reward += reward
            
            episode_duration += 1
            episode_summary.value.add(simple_value=rlchosen_action_state, 
                                      tag="episode/rlchosen_action_state")
            episode_summary.value.add(simple_value=taken_action_state[0], 
                                      tag="episode/taken_action_state")
            if i == max_iter:
                i = 0
                terminal = True
            else:
                i = i + 1

        f = open("./training.txt", mode='at', encoding='utf-8')
        f.writelines("\n")
        f.writelines("Train {} : {}\n".format(l, episode_reward))
        f.close()

        # Add summaries to tensorboard
        episode_summary.value.add(simple_value=episode_reward, 
                                  tag="episode/episode_reward")
        episode_summary.value.add(simple_value=episode_duration, 
                                  tag="episode/episode_duration")
        i_episode += 1
        agent.summary_writer.add_summary(episode_summary, i_episode)
        agent.summary_writer.flush()
        
    # VALIDATION AND TEST EPISODES
    episode_summary = tf.Summary()
    if iteration%VALIDATION_TEST_FREQUENCY == 0:
        # Validation episodes are run. Use env for it.
        all_durations = []

        """for i in range(N_VALIDATION):
            done = False
            state, next_action_state = env.reset()

            j=0
            while not(done):
                action = policy_rl(agent, state, next_action_state)        
                taken_action_state = next_action_state[:,action]
                next_state, next_action_state, reward, done = env.step(action)
                state = next_state
                
                if j == max_iter:
                    j = 0
                    done = True
                else:
                    j = j + 1
                
            all_durations.append(len(env.episode_qualities))
        episode_summary.value.add(simple_value=np.mean(all_durations), 
                                  tag="episode/train_duration")"""
        # Test episodes are run. Use env_test for it.
        all_durations = []
        
        for i in range(N_TEST):
            test_reward = []     # added by seungbo
            done = False
            state, next_action_state = env_test.reset()
            k = 0
            while not(done):
                action = policy_rl(agent, state, next_action_state)        
                taken_action_state = next_action_state[:,action]
                next_state, next_action_state, reward, done = env_test.step(action)
                test_reward.append(reward)    # added by seungbo
                state = next_state

                if k == max_iter:
                    k = 0
                    done = True
                else:
                    k = k + 1

            f = open("./training.txt", mode='at', encoding='utf-8')
            f.writelines("\n")
            f.writelines("Test {} : ".format(i))
            for ele in test_reward:
                f.write("{}, ".format(ele))
            f.close()

            all_durations.append(len(env_test.episode_qualities))
        episode_summary.value.add(simple_value=np.mean(all_durations), 
                                  tag="episode/test_duration")
    
    episode_summary.value.add(simple_value=epsilon, 
                              tag="episode/epsilon")
    agent.summary_writer.add_summary(episode_summary, iteration)
    agent.summary_writer.flush()
            
    # NEURAL NETWORK UPDATES
    for _ in range(NN_UPDATES_PER_ITERATION):
        minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
        td_error = agent.train(minibatch)
        replay_buffer.update_td_errors(td_error, minibatch.indeces)

f.close()