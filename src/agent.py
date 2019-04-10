import logging
import keras.models
import os
import random
import numpy as np
import matplotlib.pyplot as plt 

from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Activation, Dropout
from keras.optimizers import Adam, Adagrad
from keras import backend as K

class DQLAgent(object):
    def __init__(
            self, state_size=-1, action_size=-1,
            max_steps=200, gamma=1, epsilon=1.0, learning_rate=0.1,num_episodes=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.memory = deque(maxlen=2000)
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = learning_rate  # learning_rate
        self.num_episodes = num_episodes
        self.memory_epsilon = 0
        self.step = False
        self.best_score = 0
        self.results = []
        if self.state_size > 0 and self.action_size > 0:
            self.model = self.build_model()

        self.count = 0

    def build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = Sequential()
        model.add(Dense(64,input_dim=self.state_size))
        model.add(Activation('relu'))

        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def updateEpsilon(self):
        """This function change the value of self.epsilon to deal with the
        exploration-exploitation tradeoff as time goes"""
        if self.epsilon>0.5:
            self.epsilon -= 0.5/(self.num_episodes*0.1)
        elif self.epsilon>0.3:
            self.epsilon -= 0.2/(self.num_episodes*0.2)
        elif self.epsilon>0.1:
            self.epsilon -= 0.2/(self.num_episodes*0.3)
        elif self.epsilon>0:
            self.epsilon-= 0.1/(self.num_episodes*0.35)

    def save(self, output: str):
        self.model.save(output)

    def load(self, filename):
        if os.path.isfile(filename):
            self.model = keras.models.load_model(filename)
            self.state_size = self.model.layers[0].input_shape[1]
            self.action_size = self.model.layers[-1].output.shape[1]
            return True
        else:
            logging.error('no such file {}'.format(filename))
            return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=True):
        sampling = random.random()
        if(sampling<self.epsilon and not(greedy)):
            return random.randint(0,12)
        else:
            prediction = self.model.predict(state)
            return np.argmax(prediction)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.updateEpsilon()

    def setTitle(self, env, train, name, num_steps, returns):
        h = name
        if train:
            h = 'Iter {} ($\epsilon$={:.2f})'.format(self.count, self.epsilon)
        end = '\nreturn {:.2f}'.format(returns) if train else ''

        env.mayAddTitle('{}\nsteps: {} | {}{}'.format(
            h, num_steps, env.circuit.debug(), end))

    def run_once(self, env, train=True, greedy=False, name=''):
        self.count += 1
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        returns = 0
        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            action = self.act(state, greedy=greedy)
            next_state, reward, done, completed = env.step(action, greedy)
            next_state = np.reshape(next_state, [1, self.state_size])

            if train:
                self.remember(state, action, reward, next_state, done)

            returns = returns * self.gamma + reward
            state = next_state
            if done:
                return returns, num_steps, completed

            self.setTitle(env, train, name, num_steps, returns)

        return returns, num_steps, completed

    def train(
            self, env, episodes, minibatch, output='weights.h5', render=False):
        for e in range(episodes):
            r, _, completed = self.run_once(env, train=True, greedy=False)            
            self.results.append(completed)
            print("episode: {}/{}, return: {}, e: {:.2}, completed:{}".format(
                e, episodes, r, self.epsilon,completed))
            if completed > self.best_score:
                self.best_score = completed
                self.save(output)
                print('Model saved')
            if len(self.memory) > minibatch:
                self.replay(minibatch)
        # Finally runs a greedy one
        results_aggregated = [np.mean(self.results[i-200:i]) for i in range(200,len(self.results))]
        np.save(open('./array_results.npy','wb'),results_aggregated)
        plt.plot(results_aggregated)
        plt.show()
        r, n, completed = self.run_once(env, train=False, greedy=True)
        print("Greedy return: {} in {} steps : complete {} of the track".format(r, n, completed))
