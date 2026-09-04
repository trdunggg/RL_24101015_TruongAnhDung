# Lab02 - Markov Decision Process và Dynamic Programming

## Thông tin sinh viên
- Họ tên:
- MSSV:
- Lớp:
- GitHub username:

## Mục tiêu
Cài đặt Markov Chain, MDP, Bellman backup, Policy Evaluation, Policy Improvement,
Policy Iteration và Value Iteration bằng Python, sau đó áp dụng cho FrozenLake-v1.

## Cấu trúc
- `src/bai01.py` đến `src/bai36.py`: 36 bài
- `src/mdp_utils.py`: các hàm dùng chung
- `src/main.py`: chương trình tổng hợp
- `figures/`: biểu đồ
- `requirements.txt`: thư viện

## Cài đặt
```bash
pip install -r requirements.txt
```

## Cách chạy
```bash
python src/bai01.py
python src/bai24.py
python src/bai29.py
python src/bai32.py
python src/main.py
```

## Thuật toán
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration
- Simulation đánh giá policy

## Kết quả FrozenLake
Các bài 16–36 sử dụng `FrozenLake-v1`, map `4x4`, và hỗ trợ cả
`is_slippery=False` và `is_slippery=True`.

## Lưu ý
Optimal policy được sinh từ value function/Policy Iteration, không hard-code.
Transition probability được cộng đầy đủ trong Bellman backup.
