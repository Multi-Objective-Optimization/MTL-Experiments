# Thí nghiệm Phân loại Đa nhiệm: Multi-MNIST & Fashion

> **Tương ứng với:** Mục 5.2 (Real Data — Classification) trong bài báo  
> **Mahapatra & Rajan, ICML 2020** — *"Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization"*

---

## Mục tiêu thí nghiệm

Đánh giá khả năng EPO Search tìm được nghiệm Pareto tối ưu **đúng theo preference vector** người dùng cho bài toán phân loại đa nhiệm trên dữ liệu ảnh. So sánh với các baseline: Linear Scalarization, PMTL, và Individual training.

---

## Cấu hình thí nghiệm

| Thành phần | Chi tiết |
|---|---|
| **Kiến trúc mạng** | LeNet (LeCun et al., 1998) — dùng chung cho mọi phương pháp |
| **Hàm loss** | Cross-entropy loss cho mỗi task |
| **Optimizer** | SGD (Stochastic Gradient Descent) |
| **Preference vectors** | 5 vectors khác nhau (hiển thị dưới dạng tia r⁻¹ trong biểu đồ) |
| **Số task** | m = 2 |

---

## Các phương pháp so sánh

1. **Individual (Baseline):** Học từng task độc lập, không chia sẻ tham số
2. **LinScalar:** Kết hợp tuyến tính các loss theo preference vector (L1-normalized)
3. **PMTL (Pareto MTL):** Chia không gian giải pháp thành K vùng con bằng reference vectors (Lin et al., 2019); chạy 200 iterations
4. **EPO Search:** Thuật toán đề xuất — kết hợp gradient descent và controlled ascent để tìm đúng EPO solution theo preference

---

## Kết quả kỳ vọng (Hình 7 trong bài báo)

- **EPO Search** cho accuracy per-task cao nhất trong mọi lần chạy, và các nghiệm nằm gần nhất với preference vector tương ứng (r⁻¹ ray)
- **PMTL** tìm được nghiệm Pareto optimal nhưng không đảm bảo đúng preference
- **LinScalar** hoạt động kém nhất vì không thể tìm nghiệm ở phần lõm của Pareto front

---

## Cách chạy thí nghiệm

**Bước 1: Huấn luyện tất cả các phương pháp**

```bash
python individual_train.py   # Baseline: học từng task riêng
python linscalar_train.py    # Linear Scalarization
python epo_train.py          # EPO Search (phương pháp đề xuất)
python pmtl_train.py         # Pareto MTL (baseline)
```

Kết quả sẽ được lưu thành các file `.pkl` trong thư mục `results/`.

**Bước 2: Visualize kết quả**

```bash
python display_results.py
```

Sinh ra biểu đồ so sánh accuracy (hàng trên) và loss (hàng dưới) theo preference vectors cho 3 dataset.

---

## Cấu trúc file

```
multiMNIST/
├── README.md
├── individual_train.py     # Training baseline (individual tasks)
├── linscalar_train.py      # Training với Linear Scalarization
├── epo_train.py            # Training với EPO Search
├── pmtl_train.py           # Training với Pareto MTL
├── display_results.py      # Visualize và so sánh kết quả
├── models/
│   └── model_lenet.py      # Kiến trúc LeNet dùng chung
├── solvers/
│   ├── epo_lp.py           # Giải LP để tìm hướng non-dominating (EPO core)
│   └── min_norm_solvers.py # Solver tìm hướng min-norm (dùng cho MGDA/PMTL)
└── results/                # Thư mục lưu file .pkl kết quả
```
