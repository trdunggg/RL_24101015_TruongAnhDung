"""
main.py - Bài 36: Monte Carlo Agent hoàn chỉnh (Blackjack-v1, môi trường phụ FrozenLake-v1).

Chạy:  python src/main.py
Các mục in ra theo đúng 20 mục của đề bài. Hình lưu vào Lab03/figures/.
"""
import runpy
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mc_utils import (BLACKJACK_ACTION_NAMES, N_TRAIN, blackjack_grid, compute_returns,
                      epsilon_greedy_action, evaluate_policy, every_visit_mc_prediction,
                      first_visit_mc_prediction, generate_episode, make_policy_from_dict,
                      make_random_policy, mc_action_value_prediction, moving_average,
                      plot_blackjack_grid, save_figure, stick_on_20_policy,
                      train_or_load_control)

STUDENT_ID = "MSSV"
STUDENT_NAME = "Ho Ten"
MAIN_ENV = "Blackjack-v1"
OPTIONAL_ENVS = ["FrozenLake-v1"]


def section(number, title):
    """In tiêu đề một mục của mini-project."""
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")


def main():
    env = gym.make(MAIN_ENV)
    n_actions = env.action_space.n
    random_policy = make_random_policy(n_actions, seed=7)

    section(1, "Environment đã chọn")
    print(f"{STUDENT_ID} - {STUDENT_NAME} | môi trường chính: {MAIN_ENV} | phụ: {OPTIONAL_ENVS}")
    print("Gymnasium:", gym.__version__, "| NumPy:", np.__version__)

    section(2, "Observation / action space")
    print("observation_space:", env.observation_space, "= (player_sum, dealer_card, usable_ace)")
    print("action_space     :", env.action_space, "->", BLACKJACK_ACTION_NAMES)

    section(3, "Một episode mẫu (random policy)")
    episode = generate_episode(env, random_policy, seed=1)
    print(f"Episode dài {len(episode)} bước")

    section(4, "Trajectory  (state, action, reward)")
    for t, (s, a, r) in enumerate(episode):
        print(f"  t={t}: state={s}, action={BLACKJACK_ACTION_NAMES[a]}, reward={r}")

    section(5, "Reward")
    rewards = [r for _, _, r in episode]
    print("rewards:", rewards, "| total:", sum(rewards))

    section(6, "Return (gamma = 1.0)")
    print("G_t:", compute_returns(rewards, gamma=1.0))

    section(7, "Discounted return")
    demo_rewards = [0, 0, 1]
    for gamma in (1.0, 0.9, 0.5):
        print(f"  rewards={demo_rewards}, gamma={gamma} -> {compute_returns(demo_rewards, gamma)}")

    section(8, "Fixed policy (stick on 20)")
    fixed_stats = evaluate_policy(env, stick_on_20_policy, n_episodes=10000, seed=0)
    print({k: round(v, 4) for k, v in fixed_stats.items()})

    section(9, "First-Visit MC Prediction")
    V_first, n_first = first_visit_mc_prediction(env, stick_on_20_policy, 100_000, seed=0)
    print(f"{len(V_first)} state được ước lượng")
    print("V(20,10,False) =", round(V_first[(20, 10, False)], 4))

    section(10, "Every-Visit MC Prediction")
    V_every, n_every = every_visit_mc_prediction(env, stick_on_20_policy, 100_000, seed=0)
    print(f"{len(V_every)} state được ước lượng")
    print("V(20,10,False) =", round(V_every[(20, 10, False)], 4))

    section(11, "So sánh hai prediction method")
    common = [s for s in V_first if s in V_every]
    mad_bj = float(np.mean([abs(V_first[s] - V_every[s]) for s in common]))
    print(f"Blackjack : mean |V_first - V_every| = {mad_bj:.6f}  "
          "(state không lặp lại trong một episode -> hai phương pháp trùng nhau)")
    fl = gym.make(OPTIONAL_ENVS[0])                        # môi trường phụ
    fl_policy = lambda: make_random_policy(fl.action_space.n, seed=1)
    fl_ep = generate_episode(fl, fl_policy(), seed=0)
    print(f"FrozenLake: episode mẫu dài {len(fl_ep)} bước, "
          f"G_0 (gamma=0.95) = {compute_returns([r for _, _, r in fl_ep], 0.95)[0]:.4f}")
    Vf, _ = first_visit_mc_prediction(fl, fl_policy(), 20000, gamma=0.95, seed=0)
    Ve, _ = every_visit_mc_prediction(fl, fl_policy(), 20000, gamma=0.95, seed=0)
    fl_common = [s for s in Vf if s in Ve]
    mad_fl = float(np.mean([abs(Vf[s] - Ve[s]) for s in fl_common]))
    print(f"FrozenLake: mean |V_first - V_every| = {mad_fl:.6f}  (state lặp lại -> khác nhau)")
    fl.close()

    for usable_ace, name in [(False, "no_usable_ace"), (True, "usable_ace")]:
        plot_blackjack_grid(blackjack_grid(V_first, usable_ace),
                            f"V(s) - stick on 20 - usable_ace={usable_ace}",
                            f"blackjack_value_{name}.png", vmin=-1, vmax=1)

    section(12, "Q(s,a) (random policy, first-visit)")
    Q_random = mc_action_value_prediction(env, random_policy, 200_000, seed=0)
    for state in [(20, 10, False), (13, 2, False)]:
        print(f"  Q{state} = stick {Q_random[state][0]:+.3f}, hit {Q_random[state][1]:+.3f}")

    section(13, "Epsilon-greedy")
    rng = np.random.default_rng(0)
    Q_fake = {"s": np.array([1.0, 5.0])}
    picks = [epsilon_greedy_action(Q_fake, "s", 2, 0.1, rng) for _ in range(10000)]
    print(f"epsilon=0.1, Q=[1,5]: tần suất greedy = {np.mean(np.array(picks) == 1):.3f} "
          "(lý thuyết 0.95)")

    section(14, "On-policy MC Control")
    Q, policy, episode_rewards = train_or_load_control(n_episodes=N_TRAIN, epsilon=0.1)
    print(f"Đã train {N_TRAIN} episode (epsilon=0.1), policy có {len(policy)} state.")

    section(15, "Training curve")
    window = 1000
    curve = moving_average(episode_rewards, window)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(window, len(episode_rewards) + 1), curve)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean reward (moving avg, window={window})")
    ax.set_title("On-policy MC control - learning curve")
    ax.grid(alpha=0.3)
    save_figure(fig, "mc_convergence.png")
    print(f"Mean reward đầu: {curve[0]:+.4f} -> cuối: {curve[-1]:+.4f}")

    section(16, "Learned policy")
    for state in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
        a = int(np.argmax(Q[state]))
        print(f"  state={state}: Q={np.round(Q[state], 3)} -> {BLACKJACK_ACTION_NAMES[a]}")
    for usable_ace, name in [(False, "no_usable_ace"), (True, "usable_ace")]:
        plot_blackjack_grid(blackjack_grid(policy, usable_ace),
                            f"Learned policy (0=stick, 1=hit), usable_ace={usable_ace}",
                            f"blackjack_policy_{name}.png", cmap="coolwarm", vmin=0, vmax=1,
                            colorbar_label="action")

    section(17, "Evaluation (episode mới, không explore)")
    learned_policy = make_policy_from_dict(policy)
    results = {
        "random": evaluate_policy(env, random_policy, 100_000, seed=123),
        "fixed (stick on 20)": evaluate_policy(env, stick_on_20_policy, 100_000, seed=123),
        "MC learned": evaluate_policy(env, learned_policy, 100_000, seed=123),
    }

    section(18, "So sánh random / fixed / learned")
    print(f"{'policy':<22}{'win':>8}{'loss':>8}{'draw':>8}{'mean reward':>14}")
    for name, r in results.items():
        print(f"{name:<22}{r['win_rate']:>8.3f}{r['loss_rate']:>8.3f}{r['draw_rate']:>8.3f}"
              f"{r['mean_reward']:>+10.4f} ± {r['std_error']:.4f}")
    names = list(results)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(names, [results[n]["mean_reward"] for n in names],
           yerr=[results[n]["std_error"] for n in names], capsize=4)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Mean reward / episode")
    ax.set_title("So sánh policy (100000 episode đánh giá)")
    save_figure(fig, "policy_performance.png")

    # Hai hình còn lại: First vs Every-Visit (Bài 23) và so sánh epsilon (Bài 29).
    src_dir = Path(__file__).resolve().parent
    runpy.run_path(str(src_dir / "bai23.py"))
    runpy.run_path(str(src_dir / "bai29.py"))

    section(19, "Nhận xét")
    best = max(results, key=lambda n: results[n]["mean_reward"])
    print(f"- Policy tốt nhất theo mean reward: {best} "
          f"({results[best]['mean_reward']:+.4f}).")
    print(f"- Learned so với random: "
          f"{results['MC learned']['mean_reward'] - results['random']['mean_reward']:+.4f} reward/episode.")
    print(f"- Blackjack: mean |V_first - V_every| = {mad_bj:.4f}; FrozenLake: {mad_fl:.4f}.")
    print("- Mean reward của learned policy vẫn thường âm vì Blackjack có lợi thế cho nhà cái;")
    print("  MC không thể làm kỳ vọng dương, chỉ làm tốt nhất có thể.")
    print("- Reward lúc train thấp hơn lúc đánh giá do còn exploration (epsilon = 0.1).")

    section(20, "Kết luận")
    print("Monte Carlo học V(s), Q(s,a) và policy chỉ từ episode, không cần model môi trường.")
    print("Cần nhiều episode vì return có phương sai lớn; kết quả là xấp xỉ, không đảm bảo tối ưu.")

    env.close()


if __name__ == "__main__":
    main()
