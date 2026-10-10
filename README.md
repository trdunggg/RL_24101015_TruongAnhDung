# Lab03 - Monte Carlo Methods

## Thông tin sinh viên

- Họ tên: Ho Ten
- MSSV: MSSV
- Lớp:

## Môi trường đã chọn

- Môi trường chính: `Blackjack-v1`
- Môi trường phụ: `FrozenLake-v1`
- Lý do lựa chọn: Blackjack là episodic, ngắn, state rời rạc `(player_sum, dealer_card, usable_ace)` nên
  phù hợp trực tiếp với MC prediction/control. Trong Blackjack một state không lặp lại trong cùng một episode,
  nên thêm FrozenLake (state lặp lại) để thấy sự khác nhau giữa First-Visit và Every-Visit.

## Mục tiêu

Học V(s), Q(s,a) và policy **chỉ từ episode** do agent tương tác với môi trường, không dùng model
transition/reward: sinh episode, tính return, First/Every-Visit MC prediction, Q(s,a), epsilon-greedy,
On-policy MC control, đánh giá policy.

## Cài đặt

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Cách chạy

```bash
# chạy toàn bộ mini-project (Bài 36), sinh đủ hình trong figures/
python src/main.py

# hoặc chạy từng bài
python src/bai01.py
...
python src/bai36.py
```

Notebook: `notebooks/Lab03_MSSV_HoTen.ipynb` (Kernel: Python của `.venv`, Run All).
Lần chạy đầu train 500000 episode (khoảng vài chục giây); kết quả được cache trong `data/`.

## Cấu trúc

```text
Lab03/
├── README.md, requirements.txt
├── src/        mc_utils.py (hàm dùng chung), bai01.py ... bai36.py, main.py
├── notebooks/  Lab03_MSSV_HoTen.ipynb
├── figures/    mc_convergence.png, first_vs_every_visit.png, policy_performance.png, epsilon_comparison.png, ...
├── videos/
└── data/       cache kết quả train
```

## Episode và Return

Episode `S0, A0, R1, S1, A1, R2, ..., ST` lưu dạng `[(state, action, reward), ...]` (`generate_episode`).
Return tính ngược từ cuối: `G_t = R_(t+1) + gamma * G_(t+1)` (`compute_returns`).

## First-Visit MC

Chỉ cập nhật tại lần xuất hiện đầu tiên của state trong episode (`first_visit_indices`, `first_visit_mc_prediction`).

## Every-Visit MC

Cập nhật ở mọi lần state xuất hiện (`every_visit_mc_prediction`).

## Action-value Q(s,a)

`mc_action_value_prediction` ước lượng Q(s,a) (first-visit theo cặp (s,a)); `greedy_action` dùng `np.argmax`.

## Epsilon-greedy

`epsilon_greedy_action`: xác suất epsilon chọn ngẫu nhiên, 1 - epsilon chọn greedy.

## On-policy MC Control

`on_policy_mc_control`: sinh episode bằng epsilon-greedy theo Q -> tính return -> first-visit update Q(s,a)
bằng incremental mean -> cập nhật policy greedy -> lặp lại.

## Kết quả

> Điền sau khi chạy `python src/main.py` trên máy của bạn (số liệu phụ thuộc lần chạy):

| Policy | Win | Loss | Draw | Mean reward |
|---|---|---|---|---|
| random | | | | |
| fixed (stick on 20) | | | | |
| MC learned | | | | |

First vs Every-Visit: mean |V_first - V_every| Blackjack = ..., FrozenLake = ...

## Learning Curve

![learning curve](figures/mc_convergence.png)

## So sánh policy

![policy performance](figures/policy_performance.png)
![first vs every](figures/first_vs_every_visit.png)
![epsilon](figures/epsilon_comparison.png)

## Nhận xét

(Viết nhận xét của bạn từ số liệu ở trên và 25 câu hỏi trong notebook.)

## Tài liệu tham khảo

- Sutton & Barto, *Reinforcement Learning: An Introduction*, chương Monte Carlo Methods.
- Gymnasium documentation: Blackjack, FrozenLake.
- Tài liệu môn học Học tăng cường.
