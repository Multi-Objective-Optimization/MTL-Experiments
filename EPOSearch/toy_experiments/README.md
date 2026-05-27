# Thí nghiệm trên Bài toán MOO Tổng hợp (Toy Experiments)

> **Tương ứng với:** Mục 5.1 (Synthetic Data) + Phụ lục B, C trong bài báo  
> **Mahapatra & Rajan, ICML 2020** — *"Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization"*

---

## Mục tiêu thí nghiệm

Kiểm tra và so sánh các thuật toán tối ưu đa mục tiêu trên **bài toán tổng hợp có Pareto front đã biết**, cho phép đánh giá chính xác:
- Độ chính xác của nghiệm tìm được so với EPO lý tưởng
- Khả năng tìm đúng preference-specific Pareto optimal solution
- Tính bền vững với điều kiện khởi tạo khác nhau
- Hành vi khi EPO solution không tồn tại

---

## Module `problems/` — Các bài toán MOO tổng hợp

### `toy_biobjective.py` — Bài toán 2 mục tiêu (Hình 1, 5, 6 trong bài báo)

Hai hàm mục tiêu **không lồi** lấy từ Lin et al. (2019):

```
l₁(θ) = 1 - exp(-||θ - 1/√n||²)
l₂(θ) = 1 - exp(-||θ + 1/√n||²)
```

- **Không gian giải pháp:** θ ∈ Rⁿ (n = 20 chiều trong thí nghiệm)
- **Đặc điểm:** Tập multi-objective O không lồi trong objective space Rᵐ
- **Pareto front:** Biết trước → có thể đo độ lệch của nghiệm tìm được
- **Tập nghiệm Pareto (B):** {θ ∈ Rⁿ | -1/√n ≤ θ ≤ 1/√n}

> **Ý nghĩa:** Linear scalarization không thể tìm nghiệm ở phần **lõm** của Pareto front (vùng màu xám trong Hình 1a bài báo).

### `toy_triobjective.py` — Bài toán 3 mục tiêu

Mở rộng lên 3 task để kiểm tra tính mở rộng và khả năng trace Pareto front trên simplex 3 chiều (Hình 2 trong bài báo).

---

## Module `solvers/` — Các thuật toán giải MOO

| File | Thuật toán | Mô tả |
|---|---|---|
| `linscalar.py` | **Linear Scalarization** | Minimize s(θ) = rᵀl(θ) — không tìm được nghiệm ở vùng lõm |
| `moo_mtl.py` + `min_norm_solvers_numpy.py` | **MGDA (Désidéri, 2012)** | Tìm hướng descent trong Convex Hull của gradients — hội tụ về Pareto optimal tùy ý |
| `pmtl.py` / `pmtl_gpu.py` | **Pareto MTL (Lin et al., 2019)** | Chia solution space thành K vùng con bằng reference vectors |
| `epo_search.py` + `epo_lp.py` | **EPO Search (đề xuất)** | Kết hợp gradient descent + controlled ascent, giải LP để tìm hướng non-dominating |

---

## Các thí nghiệm trong Main Paper

### 1. So sánh 4 Solvers — `compare_solvers.py`
> **Tương ứng:** Hình 1d và phần thí nghiệm Section 5.1

So sánh độ phân tán và độ chính xác của nghiệm Pareto optimal cho các preference vectors khác nhau:
- **LinScalar:** Bỏ sót vùng lõm của Pareto front (Hình 1a)
- **MGDA:** Không dùng preference vector → nghiệm tùy ý (Hình 1b)
- **PMTL:** Chia vùng nhưng không đảm bảo đúng preference (Hình 1c)
- **EPO Search:** Tìm đúng nghiệm tại preference vector bất kỳ (Hình 1d)

```bash
python compare_solvers.py
```

### 2. So sánh với Khởi tạo khác nhau — `compare_init.py`
> **Tương ứng:** Hình 6 trong bài báo

Kiểm tra tính bền vững với điều kiện khởi tạo:

```bash
python compare_init.py
```

| `init_type` | Mô tả | Kỳ vọng |
|---|---|---|
| `"easy"` | θ⁰ ∈ B (gần Pareto front) | Cả PMTL và EPO đều hội tụ |
| `"hard"` | θ⁰ ∉ B (xa Pareto front) | PMTL không hội tụ đúng, EPO vẫn trace được Pareto front |

> **Kết quả quan trọng (Hình 6):** Khi khởi tạo xa (hard), PMTL không cập nhật ở phase 2 và chỉ tìm được Pareto front trong 2/4 lần chạy. EPO Search luôn descend + ascend có kiểm soát để đến đúng preference-specific solution.

### 3. So sánh Restricted vs Relaxed Descent — `compare_descent.py`
> **Tương ứng:** Hình 4 trong bài báo + Phụ lục B

So sánh hai biến thể của EPO LP:

```bash
python compare_descent.py
```

| Kiểu descent | Mô tả | Hành vi |
|---|---|---|
| **Relaxed Descent** | Không có ràng buộc thêm | Dao động quanh r⁻¹ ray trước khi hội tụ |
| **Restricted Descent** | Thêm ràng buộc trong LP (constraint J*) | Kiểm soát ascent, ngăn objective values phân kỳ |

---

## Các thí nghiệm trong Supplementary

### 4. Khi EPO không tồn tại — `empty_epo.py`
> **Tương ứng:** Phụ lục C

Dùng preference vectors mà EPO solution **không tồn tại** cho bài toán biobjective:

```bash
python empty_epo.py
```

> **Kết quả:** Thuật toán dừng tại nghiệm Pareto optimal có **non-uniformity thấp nhất** μᵣ(l(θ*)) — tức là nghiệm gần preference nhất có thể.

### 5. Trace Pareto Front — `trace_pf.py`
> **Tương ứng:** Phụ lục C + Hình 2

Dùng EPO Search để trace toàn bộ Pareto front của bài toán 3 mục tiêu bằng cách tuần tự tìm EPO cho nhiều preference vectors:

```bash
python trace_pf.py
```

> **Ứng dụng:** Designer có thể dùng EPO Search như một công cụ khám phá Pareto front một cách có hệ thống.

### 6. PMTL vs EPO Search — Many Objectives — `simulation.py` + `simulation_vis.py`
> **Tương ứng:** Phụ lục C

Thí nghiệm tổng hợp so sánh khả năng mở rộng khi tăng số mục tiêu m:

```bash
python simulation.py       # Chạy thí nghiệm, lưu simulation.pkl
python simulation_vis.py   # Visualize kết quả
```

> **Kết quả:** PMTL cần số reference vectors tăng theo hàm mũ với m → EPO Search **scale tuyến tính** theo chiều gradient (complexity O(m²n)).

---

## Cấu trúc thư mục

```
toy_experiments/
├── README.md
├── compare_solvers.py      # [Main] So sánh 4 solvers
├── compare_init.py         # [Main] So sánh với khởi tạo easy/hard
├── compare_descent.py      # [Main] Restricted vs Relaxed descent
├── empty_epo.py            # [Suppl.] Khi EPO không tồn tại
├── trace_pf.py             # [Suppl.] Trace Pareto front (triobjective)
├── simulation.py           # [Suppl.] Many-objective scaling experiment
├── simulation_vis.py       # [Suppl.] Visualize simulation results
├── problems/
│   ├── __init__.py
│   ├── toy_biobjective.py  # 2 mục tiêu không lồi
│   └── toy_triobjective.py # 3 mục tiêu
└── solvers/
    ├── linscalar.py            # Linear Scalarization
    ├── moo_mtl.py              # MGDA-based MOO
    ├── min_norm_solvers_numpy.py  # Min-norm solver (NumPy)
    ├── min_norm_solvers.py        # Min-norm solver (PyTorch/GPU)
    ├── pmtl.py                 # Pareto MTL (CPU)
    ├── pmtl_gpu.py             # Pareto MTL (GPU)
    ├── epo_lp.py               # EPO LP solver
    └── epo_search.py           # EPO Search algorithm
```
