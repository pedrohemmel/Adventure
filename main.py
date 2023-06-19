# For atari game actions
# import numpy as np
import random
import gymnasium as gym

# For deep learning model with keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam


# For build Agent with keras rl
# from rl.agents import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy




def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg', nb_actions=actions, nb_steps_warmup=10000)
    return dqn

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation="relu", input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


env = gym.make("ALE/Adventure-v5", render_mode="human")
height, width, channels = env.observation_space.shape
states = env.observation_space.shape[0]
actions = env.action_space.n

model = build_model(height, width, channels, actions)
# model.summary()

# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-4))
# dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    truncated = False
    score = 0

    while not done or truncated:
        env.render()
        action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
        if reward != 0.0:
            print(reward)
    print(f"Episode: {episode}, Score: {score}")
env.close()


