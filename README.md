# Self-driving car with Deep-Q Network

This project aims at learning to a car how to drive on a circuit using deep reinforcement learning. I used a neural network to predict the reward being in a certain state and taking a certain action.

## Getting started

Start by installing the requirements:
```
sudo pip3 install -r requirements
```
To test my trained agent in a greedy way (saved in the .h5 file):
```
python3 -m scripts.run_test --model='weights.h5'
```

To train your own reinforcement learning agent with some parameters:
```
python3 -m scripts.run_train --num_episodes=X --output='my_weights.h5'
```
### Parameters available for the training

* num_episodes : how many episodes lasts the training
* max_steps : maximum number of steps in an episode
* minibatch_size : how many samples are used to train the neural network at each episode
* gamma : discount factor between 0 and 1, represents how much we value the future steps over the next step
* learning_rate : how fast we learn from batch while training the neural network
* output : location where the weights will be saved
* ui : True if you want to display the graphical interface and False if not

## My project

### Principal components

In the scope of the project, the circuit build with shapely, the car with its basic actions and the ui were provided, since it wasn't the main interest of this project. I built myself the components of the AI behind the car that is contained in the files environment.py and agent.py

#### Environment 
The environment links the car and the circuit, i.e makes the car advance depending on its action, determines if the car has crashed and thus if the game has ended. **Very important : the reward given to the car depending on its state is defined in this file**

#### Agent

Most of the AI logic is coded there, key functions : 
* build_model : build the neural network that will be used to learn reward
* act : make a move, either the best one predicted by our neural network or a random one if we follow a non greedy strategy
* updateEpsilon : change epsilon as the training goes to deal with the exploration/exploitation tradeoff
* replay : train the neural network using the memory of moves to fit the neural network with the reward expectation for the move that has been made

### My methodology 

#### Training the model step by step

Firstly I had hard time to train the car in one time, so what I did to understand how to proceed was to train one time the model to achieve a first decent score on the circuit, saving weights of the neural network. Training the model again etc...

I firsly started with the reward function being the product between the speed and the squared minimum of distances to the walls measured by the car. If the car crashes, he gets a -1 penalty. This reward encourages the car to move further by being careful. I took 0.7 as value for gamma and trained and neural networks with &1 hidden layer, because the problem is quite simple. My optimizer was Adam whose I could control its learning rate.
Here are the different step I took to obtain a first model capable of driving around the track:
* 1rst step : 5000 steps, learning rate = 0.01, minibatch size = 64, greedy policy : linear. I reached a score of 18% of the circuit completed.
* 2nd step : 3000 step, learning rate = 0.001, minibatch size = 100, greedy policy unchanged (but starting at 0.9). I reached a score of 30% of the circuit completed.
* 3rd step : 3000 step, learning rate = 0.0001, minibatch size = 128, greedy policy unchanged (but starting at 0.7). I reached a score of 55% of the circuit completed.
* 4th step : 3000 step, learning rate = 0.00001, minibatch size = 150, greedy policy unchanged (but starting at 0.5). I reached a score of 86% of the circuit completed.
* 5th step : 2000 step, learning rate = 0.000005, minibatch size = 200, greedy policy unchanged (but starting at 0.3). I reached a score of 154% of the circuit completed.

#### Greedy policy 

The greedy policy I found by test and trial was the following : start by exploiting a lot new solutions (epsilon start at 1 : full random actions ) and reduce epsilon until it reaches 0.5.

Then continue reducing it but slowly because between 0.2 and 0.5 is the moment when the algorithm learn the most.

Once we reach 0.2, we continue reducing until 0, this is the moment when the algorithm fine-tune its choices.

#### Choosing the reward function
The reward function is the most important thing in our model because it will teach the car what actions are good or not. 


#### Choosing the neural network 

Since the problem is very simple and since it musn't be slow to train, we don't need here a deep neural network. I then opted for a very simple model with one hidden layer of 32 neurons with ReLu activation.

As optimizer, I chose Adam with the learning rate specified in parameters. After multiple tries, I found that a small learning rate was required, indeed if the learning rate is too big, the model can't learn precisely enough to pass complex corners.

The neural network chosen and the optimizer are quite important to improve the efficiency of the training, although it is less important than the reward function or the exploration/exploitation tradeoff.


