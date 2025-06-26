import torch
import torch.nn as nn
import torch.optim as optim
import random

# Defining the environment
class NumberGuessingGame:
    def __init__(self, target):
        # Target is a float between 0 and 1
        self.target = target
    
    def step(self, guess):
        # Reward is higher as the guess gets closer to the target
        reward = -abs(self.target - guess)  # Maximize reward by minimizing the negative difference
        return reward

# Defining the agent
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc = nn.Linear(1, 1)  # Linear layer
        self.sigmoid = nn.Sigmoid()  # Ensures output is between 0 and 1

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train(agent, game, optimizer, epochs):
    for epoch in range(epochs):
        # Random input to vary the agent's input each time
        state = torch.rand(1, 1)  # Generates a float between 0 and 1
        guess = agent(state)
        
        # Calculate reward based on how close the guess is to the target
        reward = game.step(guess)
        
        # Loss is the negative reward
        loss = -reward
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Guess: {guess.item():.4f}, Target: {game.target:.4f}")

# Setting a random target number between 0 and 1
target_number = random.random()
game = NumberGuessingGame(target_number)
agent = Agent()
optimizer = optim.SGD(agent.parameters(), lr=0.1)

# Training the agent
train(agent, game, optimizer, 1000)
