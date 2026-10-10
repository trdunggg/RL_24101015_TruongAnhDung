"""
mc_utils.py - Các hàm dùng chung cho Lab03 (Monte Carlo Methods).

Quy ước episode:  S0, A0, R1, S1, A1, R2, ..., ST
Trong code:       episode = [(state, action, reward), ...]
                  phần tử thứ t là (S_t, A_t, R_{t+1}).

Không dùng thư viện RL có sẵn: mọi thuật toán MC đều tự cài đặt.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Đường dẫn & hằng số
# ----------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = LAB_DIR / "figures"
DATA_DIR = LAB_DIR / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "Blackjack-v1"
SEED = 42
N_TRAIN = 500_000          # số episode train MC control (>= 100000 theo đề)
BLACKJACK_ACTION_NAMES = {0: "stick (dừng)", 1: "hit (rút bài)"}


# ----------------------------------------------------------------------------
# Policy đơn giản
# ----------------------------------------------------------------------------
def stick_on_20_policy(state):
    """Policy cố định của Blackjack: player_sum >= 20 thì stick (0), ngược lại hit (1)."""
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1


def make_random_policy(n_actions, seed=None):
    """Tạo random policy: chọn action đều ngẫu nhiên, không phụ thuộc state."""
    rng = np.random.default_rng(seed)

    def policy(state):
        return int(rng.integers(n_actions))

    return policy


def make_policy_from_dict(policy_dict, default_action=0):
    """Biến dict {state: action} thành hàm policy(state). State chưa gặp -> default_action."""

    def policy(state):
        return policy_dict.get(state, default_action)

    return policy


def moving_average(values, window=1000):
    """Trung bình trượt (valid mode). Trả về mảng độ dài len(values) - window + 1."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        raise ValueError("Số phần tử nhỏ hơn window")
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


# ----------------------------------------------------------------------------
# Episode & Return
# ----------------------------------------------------------------------------
def generate_episode(env, policy, seed=None, max_steps=1000):
    """
    Sinh một episode bằng cách cho agent (policy) tương tác với env.

    Tham số:
        env: môi trường gymnasium.
        policy: hàm policy(state) -> action.
        seed: nếu không None thì env.reset(seed=seed) (chỉ nên truyền ở episode đầu).
        max_steps: số bước tối đa (phòng episode quá dài).
    Trả về:
        episode: list các tuple (state, action, reward) với reward = R_{t+1}.
    """
    episode = []
    state, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        action = int(policy(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, float(reward)))
        state = next_state
        if terminated or truncated:
            break
    return episode


def compute_returns(rewards, gamma=1.0):
    """
    Tính return G_t cho mọi timestep, duyệt NGƯỢC từ cuối episode:
        G_t = R_{t+1} + gamma * G_{t+1},  G_T = 0
    Tham số:
        rewards: [R_1, ..., R_T]  (rewards[t] = R_{t+1}).
    Trả về:
        list [G_0, G_1, ..., G_{T-1}] cùng độ dài với rewards.
    """
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def first_visit_indices(states):
    """Trả về dict {state: chỉ số t xuất hiện ĐẦU TIÊN trong episode}."""
    first = {}
    for t, state in enumerate(states):
        if state not in first:
            first[state] = t
    return first


# ----------------------------------------------------------------------------
# Monte Carlo prediction V(s)
# ----------------------------------------------------------------------------
def mc_prediction_with_history(env, policy, n_episodes, gamma=1.0, first_visit=True,
                               seed=None, track_states=None, checkpoints=None):
    """
    MC prediction tổng quát (first-visit hoặc every-visit) + ghi lại lịch sử estimate.

    Trả về:
        V: dict {state: trung bình return}
        returns_count: dict {state: số lần cập nhật}
        history: dict {state: [estimate tại mỗi checkpoint]} (nan nếu chưa gặp)
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    track_states = list(track_states or [])
    checkpoint_set = set(checkpoints or [])
    history = {s: [] for s in track_states}

    for ep in range(1, n_episodes + 1):
        episode = generate_episode(env, policy, seed=seed if ep == 1 else None)
        states = [s for s, _, _ in episode]
        rewards = [r for _, _, r in episode]
        G = compute_returns(rewards, gamma)
        first_idx = first_visit_indices(states)

        for t, state in enumerate(states):
            if first_visit and first_idx[state] != t:
                continue                       # first-visit: bỏ các lần xuất hiện sau
            returns_sum[state] += G[t]
            returns_count[state] += 1

        if ep in checkpoint_set:
            for s in track_states:
                c = returns_count.get(s, 0)
                history[s].append(returns_sum[s] / c if c > 0 else np.nan)

    V = {s: returns_sum[s] / returns_count[s] for s in returns_count}
    return V, dict(returns_count), history


def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0, seed=None):
    """First-Visit MC prediction. Trả về (V, returns_count)."""
    V, counts, _ = mc_prediction_with_history(env, policy, n_episodes, gamma,
                                              first_visit=True, seed=seed)
    return V, counts


def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0, seed=None):
    """Every-Visit MC prediction. Trả về (V, returns_count)."""
    V, counts, _ = mc_prediction_with_history(env, policy, n_episodes, gamma,
                                              first_visit=False, seed=seed)
    return V, counts


# ----------------------------------------------------------------------------
# Q(s,a), greedy, epsilon-greedy
# ----------------------------------------------------------------------------
def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0, seed=None):
    """
    First-visit MC ước lượng Q(s,a) cho một policy cho trước.
    Trả về Q: dict {state: np.array(n_actions)}; cặp (s,a) chưa gặp có giá trị 0.
    """
    n_actions = env.action_space.n
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    for ep in range(n_episodes):
        episode = generate_episode(env, policy, seed=seed if ep == 0 else None)
        G = compute_returns([r for _, _, r in episode], gamma)
        seen = set()
        for t, (state, action, _) in enumerate(episode):
            if (state, action) in seen:
                continue
            seen.add((state, action))
            returns_sum[state][action] += G[t]
            returns_count[state][action] += 1

    Q = {}
    for state in returns_sum:
        counts = np.maximum(returns_count[state], 1)
        Q[state] = returns_sum[state] / counts
    return Q


def greedy_action(Q, state, n_actions):
    """Action có Q lớn nhất tại state (np.argmax). State chưa gặp -> action 0."""
    if state not in Q:
        return 0
    return int(np.argmax(Q[state][:n_actions]))


def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    """
    Với xác suất epsilon chọn action ngẫu nhiên đều trong mọi action (exploration);
    với xác suất 1 - epsilon chọn greedy action (exploitation).
    rng: numpy.random.Generator.
    """
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return greedy_action(Q, state, n_actions)


def update_action_value_incremental(Q, N, state, action, G):
    """
    Cập nhật trung bình tăng dần:
        N(s,a) <- N(s,a) + 1
        Q(s,a) <- Q(s,a) + (G - Q(s,a)) / N(s,a)
    Q, N: dict/defaultdict {state: np.array(n_actions)}.
    """
    N[state][action] += 1
    Q[state][action] += (G - Q[state][action]) / N[state][action]
    return Q[state][action]


# ----------------------------------------------------------------------------
# On-policy MC control
# ----------------------------------------------------------------------------
def on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42,
                         epsilon_decay=1.0, epsilon_min=0.0):
    """
    On-policy First-Visit MC Control với epsilon-greedy policy.

    Mỗi episode: sinh episode bằng epsilon-greedy theo Q hiện tại -> tính return ->
    first-visit update Q(s,a) bằng incremental mean -> cập nhật policy greedy.
    Trả về:
        Q: defaultdict {state: np.array(n_actions)}
        policy: dict {state: greedy action} (policy greedy cuối cùng)
        episode_rewards: list tổng reward của từng episode (trong lúc train, có exploration)
    """
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))
    policy = {}
    episode_rewards = []
    eps = epsilon

    for ep in range(n_episodes):
        behavior = lambda s, e=eps: epsilon_greedy_action(Q, s, n_actions, e, rng)
        episode = generate_episode(env, behavior, seed=seed if ep == 0 else None)
        rewards = [r for _, _, r in episode]
        G = compute_returns(rewards, gamma)

        seen = set()
        for t, (state, action, _) in enumerate(episode):
            if (state, action) in seen:
                continue                                   # first-visit theo cặp (s,a)
            seen.add((state, action))
            update_action_value_incremental(Q, N, state, action, G[t])
            policy[state] = greedy_action(Q, state, n_actions)   # policy improvement

        episode_rewards.append(sum(rewards))
        eps = max(epsilon_min, eps * epsilon_decay)

    return Q, policy, episode_rewards


def train_or_load_control(n_episodes=N_TRAIN, epsilon=0.1, gamma=1.0, seed=SEED,
                          env_id=ENV_ID, use_cache=True):
    """
    Train on_policy_mc_control; lưu kết quả vào data/ để các bài sau dùng lại
    (tránh train lại nhiều lần). Xóa file .pkl trong data/ để train lại từ đầu.
    """
    import gymnasium as gym

    cache = DATA_DIR / f"mc_control_{env_id}_{n_episodes}_{epsilon}_{gamma}_{seed}.pkl"
    env = gym.make(env_id)
    n_actions = env.action_space.n
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        Q = defaultdict(lambda: np.zeros(n_actions), saved["Q"])
        env.close()
        return Q, saved["policy"], saved["episode_rewards"]

    Q, policy, rewards = on_policy_mc_control(env, n_episodes, gamma, epsilon, seed)
    env.close()
    with open(cache, "wb") as f:
        pickle.dump({"Q": dict(Q), "policy": policy, "episode_rewards": rewards}, f)
    return Q, policy, rewards


# ----------------------------------------------------------------------------
# Đánh giá policy
# ----------------------------------------------------------------------------
def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    """
    Chạy policy (hàm policy(state)) trên n_episodes episode MỚI, không học.
    Trả về dict: win_rate, loss_rate, draw_rate, mean_reward, std_error.
    """
    wins = losses = draws = 0
    totals = []
    for ep in range(n_episodes):
        episode = generate_episode(env, policy, seed=seed if ep == 0 else None)
        total = sum(r for _, _, r in episode)
        totals.append(total)
        if total > 0:
            wins += 1
        elif total < 0:
            losses += 1
        else:
            draws += 1
    totals = np.array(totals)
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": float(totals.mean()),
        "std_error": float(totals.std(ddof=1) / np.sqrt(n_episodes)),
    }


# ----------------------------------------------------------------------------
# Trực quan hóa
# ----------------------------------------------------------------------------
def save_figure(fig, filename):
    """Lưu figure vào Lab03/figures/ và đóng figure. Trả về đường dẫn."""
    import matplotlib.pyplot as plt

    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:                                    # trong Jupyter/VS Code notebook: hiển thị luôn hình
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            display(fig)
    except ImportError:
        pass
    plt.close(fig)
    print(f"Đã lưu: {path}")
    return path


PLAYER_SUMS = list(range(12, 22))      # 12..21
DEALER_CARDS = list(range(1, 11))      # 1 (Ace)..10


def blackjack_grid(table, usable_ace, fill=np.nan):
    """
    Chuyển dict {(player_sum, dealer_card, usable_ace): giá trị} thành ma trận
    (hàng = player_sum 12..21, cột = dealer_card 1..10) cho một giá trị usable_ace.
    """
    grid = np.full((len(PLAYER_SUMS), len(DEALER_CARDS)), fill, dtype=float)
    for i, ps in enumerate(PLAYER_SUMS):
        for j, dc in enumerate(DEALER_CARDS):
            key = (ps, dc, usable_ace)
            if key in table:
                grid[i, j] = table[key]
    return grid


def plot_blackjack_grid(grid, title, filename, cmap="RdYlGn", vmin=None, vmax=None,
                        annotate=True, colorbar_label="V(s)"):
    """Vẽ heatmap player_sum x dealer_card và lưu vào figures/."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DEALER_CARDS)))
    ax.set_xticklabels(["A"] + [str(c) for c in DEALER_CARDS[1:]])
    ax.set_yticks(range(len(PLAYER_SUMS)))
    ax.set_yticklabels(PLAYER_SUMS)
    ax.set_xlabel("Dealer showing card")
    ax.set_ylabel("Player sum")
    ax.set_title(title)
    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    return save_figure(fig, filename)
