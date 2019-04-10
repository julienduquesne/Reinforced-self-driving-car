"""
To train a DQL Agent to drive a car, from the carl/ directory run

python3 -m scripts.run_train
"""

import argparse

from src.agent import DQLAgent
from src.circuit import Circuit
from src.environment import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--output', type=str, default='weights.h5')
    parser.add_argument('--ui', type=str, default='false')
    args = parser.parse_args()

    circuit = Circuit(
        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (7,1), (6, 0)], width=0.3)

    render = args.ui.lower() != 'false'
    env = Environment(circuit=circuit, render=render)

    agent = DQLAgent(
        state_size=len(env.current_state), action_size=len(env.actions),
        gamma=args.gamma, learning_rate=args.learning_rate,num_episodes=args.num_episodes)

    agent.train(
        env, episodes=args.num_episodes, minibatch=args.minibatch_size,
        output=args.output)
