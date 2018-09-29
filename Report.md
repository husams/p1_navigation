# Project 1: Navigation

The purpose of the project is to train an agent to collect yellow bananas while avoiding blue bananas. The environment is considered solved when the agent reaches average score of 13+ on 100 consecutive episodes.


### Actions:

The environment has the following 4 discrete values for the actions:

- **`0`** - Forward.
- **`1`** - Backward.
- **`2`** - Left.
- **`3`** - Right.

### State

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

### Reward

The agent should receive reward of +1 for each yellow banana and -1 one for blue

## Learning Algorithm

  I used Double Q-Learning which was introduced by Deep Mind paper "[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf")" as an imporovment to there original algorithm Deep Q-Learning (DQN).
