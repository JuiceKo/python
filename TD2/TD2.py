import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """

    old_value = Q[s, a]
    max_next = np.max(Q[sprime, :])
    new_value = old_value + alpha * (r + gamma * max_next - old_value)
    Q[s, a] = new_value

    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """

    n_actions = Q.shape[1]

    if np.random.rand() < epsilone:
        action = np.random.randint(n_actions)
    else:
        action = np.argmax(Q[s, :])

    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01
    gamma = 0.8
    epsilon = 0.2

    n_epochs = 20
    max_itr_per_epoch = 100
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            S = Sprime

            if done:
                break

        print("n°", e, " : Reward : ", r)
        rewards.append(r)

    print("Reward moyenne : ", np.mean(rewards))
    print("\n")

    """
    Evaluate the q-learning algorithm
    """

    n_test_episodes = 5
    test_rewards = []

    for e in range(n_test_episodes):
        S, _ = env.reset()
        r = 0

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=0.0)
            Sprime, R, done, _, info = env.step(A)
            r += R
            S = Sprime
            if done:
                break

        print("Test ", e, " : Reward = ", r)
        test_rewards.append(r)

    print("Récompense moyenne = ", np.mean(test_rewards))

    env.close()
