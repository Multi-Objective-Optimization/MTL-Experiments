# Thí nghiệm Phân loại Đa nhiệm: Multi-Task Classification (MTC)

> **Tương ứng với:** Phụ lục C (Supplementary — Multi-Label Classification) trong bài báo  
> **Mahapatra & Rajan, ICML 2020** — *"Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization"*

---

## Mục tiêu thí nghiệm

Đánh giá EPO Search trên bài toán **phân loại đa nhiệm (multi-task classification)** với nhiều task phân loại đồng thời. Thí nghiệm này mở rộng đánh giá từ bài toán 2 task (multiMNIST) sang nhiều task phân loại hơn, kiểm tra tính mở rộng của EPO Search với số task tăng lên.

> **Ứng dụng thực tế được đề cập trong bài báo:** Phân loại multi-class với dữ liệu imbalanced — xem mỗi class là một task và dùng EPO Search để re-weight class-specific losses theo preferences người dùng (Cui et al., 2019).

---

## Bối cảnh & Động lực

Trong phân loại đa nhiệm, các task thường **xung đột với nhau** (trade-off). Ví dụ:
- Trong multi-task recommender systems: tối ưu semantic relevance có thể xung đột với revenue
- Trong nhận dạng cảm xúc: một số cảm xúc dễ nhầm lẫn

EPO Search cho phép nhà thiết kế chỉ định **preference vector** để kiểm soát mức độ ưu tiên giữa các task, tìm ra nghiệm Pareto exact phù hợp với yêu cầu.

---

## Cấu hình thí nghiệm

| Thành phần | Chi tiết |
|---|---|
| **Kiến trúc mạng** | Fully-connected FNN (Feed-forward Neural Network) |
| **Hàm loss** | Cross-entropy loss cho mỗi task |
| **Optimizer** | SGD (Stochastic Gradient Descent) |
| **Evaluation** | Accuracy per-task, Loss per-task theo preference vectors |

---

## Các phương pháp so sánh

1. **Individual (Baseline):** Học từng task phân loại độc lập
2. **LinScalar:** Tổ hợp tuyến tính các cross-entropy losses
3. **PMTL (Pareto MTL):** Phân vùng Pareto front bằng reference vectors
4. **EPO Search:** Tìm preference-specific Pareto optimal solution chính xác

---

## Cách chạy thí nghiệm

**Bước 1: Huấn luyện tất cả phương pháp**

```bash
python individual_train.py   # Baseline: học từng task riêng
python linscalar_train.py    # Linear Scalarization
python epo_train.py          # EPO Search (phương pháp đề xuất)
python pmtl_train.py         # Pareto MTL (baseline)
```

Kết quả lưu vào `results/` dưới dạng file `.pkl`.

**Bước 2: Visualize kết quả**

```bash
python display_results.py
```

Sinh ra biểu đồ so sánh accuracy và loss theo preference vectors.

---

## Cấu trúc file

```
mtc/
├── README.md
├── individual_train.py     # Training baseline (individual tasks)
├── linscalar_train.py      # Training với Linear Scalarization
├── epo_train.py            # Training với EPO Search
├── pmtl_train.py           # Training với Pareto MTL
├── display_results.py      # Visualize và so sánh kết quả
├── models/
│   └── model_fnn.py        # Kiến trúc Feed-forward Neural Network
├── solvers/
│   ├── epo_lp.py           # Giải LP tìm hướng non-dominating (EPO core)
│   └── min_norm_solvers.py # Solver tìm hướng min-norm
└── results/
    └── indiv_emotion_fnn_200.pkl  # Kết quả Individual training đã lưu
```

---

## Liên hệ với các thí nghiệm khác

- **[multiMNIST](../multiMNIST/README.md):** Phân loại ảnh 2 task với LeNet (Main Paper Section 5.2)
- **[mtr](../mtr/README.md):** Hồi quy 8 task với FNN (Main Paper Section 5.2)
- **[toy_experiments](../toy_experiments/README.md):** Thí nghiệm tổng hợp kiểm tra thuật toán (Main Paper Section 5.1)
