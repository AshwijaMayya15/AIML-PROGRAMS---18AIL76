import numpy as np
import gym
import matplotlib.pyplot as plt
env=gym.make('FrozenLake-v1')
np.set_printoptions(precision=6,suppress=True)

learning_rate=0.8
discount_factor=0.95
exploration_prob=0.3
num_episode=1000
state_space_size=env.observation_space.n
action_space_size=env.action_space.n
q_table=np.zeros((state_space_size,action_space_size))

for episode in range(num_episode):
  state=env.reset()
  done=False
  while not done:
    if np.random.uniform(0,1)<exploration_prob:
      action=env.action_space.sample()
    else:
      action=np.argmax(q_table[state,:])
    next_state,reward,done,_=env.step(action)
    q_table[state, action] =(1-learning_rate) * q_table[state,action] + \
                                                   learning_rate * (reward + discount_factor * np.max(q_table[next_state,:]))
    state=next_state
total_reward=0
num_eval_episode=10

for _ in range(num_eval_episode):
  state=env.reset()
  done=False
  while not done:
    action=np.argmax(q_table[state,None])
    next_state, reward, done, _ = env.step(action)

    total_reward+=reward
    state=next_state
average_reward=total_reward/num_eval_episode
print(f"Average reward ove {num_eval_episode} episodes:{average_reward}")
print("\n Learning table",q_table)
plt.plot(q_table)
plt.xlabel('episode')
plt.ylabel('Q value')
plt.show()
