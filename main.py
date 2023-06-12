import random
import gymnasium as gym
import pygame

env = gym.make("ALE/Adventure-v5")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random_choice([0, actions])
        n_state, reward, done, info = env.step(action)
        score += reward
        print(f"Episode: {episode}, Score: {score}")
