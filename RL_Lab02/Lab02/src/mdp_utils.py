import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def create_environment(is_slippery=True, map_name="4x4"):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

def q_from_v(env, V, state, action, gamma):
    total = 0.0
    for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
        total += probability * (reward + (0.0 if terminated else gamma * V[next_state]))
    return total

def action_values(env, V, state, gamma):
    return np.array([
        q_from_v(env, V, state, action, gamma)
        for action in range(env.action_space.n)
    ])

def policy_evaluation_sweep(env, policy, V, gamma):
    new_V = np.zeros_like(V, dtype=float)
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            new_V[state] += policy[state, action] * q_from_v(
                env, V, state, action, gamma
            )
    return new_V

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n, dtype=float)
    for iteration in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < theta:
            return V, iteration
    return V, max_iterations

def policy_evaluation_with_deltas(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n, dtype=float)
    deltas = []
    for iteration in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break
    return V, iteration, deltas

def greedy_policy_from_value(env, V, gamma=0.99):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        policy[state] = int(np.argmax(action_values(env, V, state, gamma)))
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)
    for iteration in range(1, max_iterations + 1):
        V, _ = policy_evaluation(
            env, np.eye(n_actions)[policy], gamma, theta
        )
        new_policy = greedy_policy_from_value(env, V, gamma)
        if np.array_equal(policy, new_policy):
            return new_policy, V, iteration
        policy = new_policy
    return policy, V, max_iterations

def value_iteration_sweep(env, V, gamma):
    new_V = np.zeros_like(V, dtype=float)
    for state in range(env.observation_space.n):
        new_V[state] = np.max(action_values(env, V, state, gamma))
    return new_V

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n, dtype=float)
    deltas = []
    for iteration in range(1, max_iterations + 1):
        new_V = value_iteration_sweep(env, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break
    return V, iteration, deltas

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    rewards = []
    lengths = []
    successes = 0
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        length = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            length += 1
        rewards.append(total_reward)
        lengths.append(length)
        if total_reward > 0:
            successes += 1
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "min_episode_length": int(np.min(lengths)),
        "max_episode_length": int(np.max(lengths)),
    }

def print_frozenlake_policy(env, policy):
    side = int(np.sqrt(env.observation_space.n))
    for r in range(side):
        row = []
        for c in range(side):
            state = r * side + c
            tile = env.unwrapped.desc[r, c].decode()
            if tile == "H":
                row.append("H")
            elif tile == "G":
                row.append("G")
            else:
                row.append(ACTION_SYMBOLS[int(policy[state])])
        print(" ".join(row))

def validate_mdp(P, n_states, n_actions):
    if len(P) != n_states:
        return False
    for state in range(n_states):
        if len(P[state]) != n_actions:
            return False
        for action in range(n_actions):
            transitions = P[state][action]
            total = sum(t[0] for t in transitions)
            if not np.isclose(total, 1.0):
                print(f"Invalid transition at state={state}, action={action}")
                return False
    return True
