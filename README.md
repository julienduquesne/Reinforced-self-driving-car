# Self-driving car with Deep-Q Network

Start by installing the requirements:
```
sudo pip3 install -r requirements
```
To train your reinforcement learning agent with some parameters:
```
python -m scripts.run_train --num_episodes=X --output='my_weights.h5'
```

To test your trained agent in a greedy way (saved in the .h5 file):
```
python -m scripts.run_test --model='my_weights.h5'
```