# EPO Search - Exact Pareto Optimal Search for Multi-Task Learning

> **Bài báo:** *"Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization"*  
> **Tác giả:** Debabrata Mahapatra, Vaibhav Rajan (National University of Singapore)  
> **Venue:** ICML 2020 (Proceedings of the 37th International Conference on Machine Learning)  
> **Code gốc:** https://github.com/dbmptr/EPOSearch

---

## Tóm tắt bài báo

Multi-Task Learning (MTL) thường gặp tình huống các task **xung đột nhau** - cải thiện task này làm giảm hiệu quả task kia. Các phương pháp MTL hiện tại hoặc tìm nghiệm Pareto tùy ý (MGDA), hoặc chia vùng Pareto front nhưng không đảm bảo đúng preference (PMTL), hoặc bỏ sót vùng lõm của Pareto front (Linear Scalarization).

### Đóng góp chính

Bài báo đề xuất **EPO Search** - thuật toán MTL đầu tiên dựa trên gradient có thể tìm **preference-specific Pareto optimal solution** chính xác:

| Tính năng | EPO Search | LinScalar | MGDA | PMTL |
|---|:---:|:---:|:---:|:---:|
| Tìm đúng preference vector | ✅ | ❌ | ❌ | ❌ |
| Xử lý vùng lõm Pareto front | ✅ | ❌ | ✅ | ✅ |
| Bền vững với khởi tạo | ✅ | ✅ | ✅ | ❌ |
| Scale với số task lớn | ✅ O(m²n) | ✅ | ✅ | ❌ (mũ) |
| Đảm bảo hội tụ lý thuyết | ✅ | ✅ | ✅ | — |

---

## Ý tưởng cốt lõi của EPO Search

### Vấn đề: Tìm Preference-Specific Pareto Optimal

Cho preference vector r = (r₁, ..., rₘ) với rⱼ > 0, tìm θ* ∈ P (Pareto front) sao cho:

> **Nếu rᵢ ≥ rⱼ thì lᵢ(θ*) ≤ lⱼ(θ*)** (loss tỉ lệ nghịch với preference)

Nghiệm như vậy gọi là **Exact Pareto Optimal (EPO)** solution.

### Giải pháp: Kết hợp Descent + Controlled Ascent

EPO Search định nghĩa **Non-Uniformity** của điểm θ hiện tại so với preference r:

```
μᵣ(l(θ)) = KL(l̂(θ) ‖ 1/m)
```

trong đó l̂ⱼ = rⱼlⱼ / Σrⱼ'lⱼ' là normalized weighted loss. μᵣ = 0 khi và chỉ khi nghiệm nằm trên tia r⁻¹ (tức là r ⊙ l ∝ 1).

**Thuật toán (mỗi iteration):**
1. Tính gradients gⱼ = ∇θlⱼ và ma trận C = GᵀG
2. Tính loss adjustments aⱼ = rⱼ(log(l̂ⱼ / 1/m) - μᵣ)
3. Tìm β* bằng cách giải **Linear Program (LP)** m chiều:
   - Maximize β^T C(a·μᵣ + 1·(1-μᵣ)) — kết hợp balancing direction + descent direction
   - Ràng buộc controlled ascent: ngăn objective values phân kỳ
4. Cập nhật: θ^{t+1} = θᵗ - η·G·β*

**Hai chế độ hoạt động:**
- **Balancing mode** (μᵣ > 0): Di chuyển θ về phía tia r⁻¹ (giảm non-uniformity)
- **Pure descent mode** (μᵣ = 0): Descent dọc theo Pareto front để đến EPO solution

---

## Cấu trúc thư mục

```
EPOSearch/
│
├── README.md                        # Tổng quan (file này)
├── requirements.txt                 # Python dependencies
│
├── toy_experiments/                 # Thí nghiệm trên bài toán MOO tổng hợp (§5.1, Appx B,C)
│   ├── README.md
│   ├── compare_descent.py           # So sánh các hướng descent
│   ├── compare_init.py              # So sánh khởi tạo khác nhau
│   ├── compare_solvers.py           # So sánh tổng hợp các solver
│   ├── empty_epo.py                 # Demo EPO trên bài toán rỗng
│   ├── simulation.py                # Mô phỏng tìm Pareto front
│   ├── simulation_vis.py            # Visualize kết quả mô phỏng
│   ├── trace_pf.py                  # Trace Pareto front theo preference
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── toy_biobjective.py       # Bài toán 2 mục tiêu (từ Lin et al., 2019)
│   │   └── toy_triobjective.py      # Bài toán 3 mục tiêu
│   └── solvers/
│       ├── epo_lp.py                # LP core của EPO Search
│       ├── epo_search.py            # EPO Search solver
│       ├── linscalar.py             # Linear Scalarization
│       ├── min_norm_solvers.py      # Min-norm solver (MGDA, PyTorch)
│       ├── min_norm_solvers_numpy.py# Min-norm solver (NumPy)
│       ├── moo_mtl.py               # MOO-MTL / MGDA solver
│       ├── pmtl.py                  # Pareto MTL solver
│       └── pmtl_gpu.py              # Pareto MTL solver (GPU)
│
├── multiMNIST/                      # Phân loại ảnh đa nhiệm (§5.2 Classification)
│   ├── README.md
│   ├── epo_train.py                 # Training với EPO Search
│   ├── individual_train.py          # Training từng task riêng lẻ
│   ├── linscalar_train.py           # Training với Linear Scalarization
│   ├── pmtl_train.py                # Training với Pareto MTL
│   ├── display_results.py           # Visualize kết quả
│   ├── models/
│   │   └── model_lenet.py           # LeNet model cho Multi-MNIST
│   ├── solvers/
│   │   ├── epo_lp.py
│   │   └── min_norm_solvers.py
│   └── results/                     # Kết quả đã lưu (.pkl)
│
├── mtr/                             # Hồi quy đa mục tiêu (§5.2 Regression)
│   ├── README.md
│   ├── epo_train.py
│   ├── individual_train.py
│   ├── linscalar_train.py
│   ├── pmtl_train.py
│   ├── display_results.py
│   ├── models/
│   │   └── model_fnn.py             # FNN 64→32→16→8→8 cho River Flow
│   ├── solvers/
│   │   ├── epo_lp.py
│   │   └── min_norm_solvers.py
│   └── results/
│
└── mtc/                             # Phân loại đa nhiệm nhiều task (Appx C)
    ├── README.md
    ├── epo_train.py
    ├── individual_train.py
    ├── linscalar_train.py
    ├── pmtl_train.py
    ├── display_results.py
    ├── models/
    │   └── model_fnn.py             # FNN cho multi-task classification
    ├── solvers/
    │   ├── epo_lp.py
    │   └── min_norm_solvers.py
    └── results/
        └── indiv_emotion_fnn_200.pkl
```

---

## Tóm tắt các thí nghiệm

| Folder | Bài toán | Dataset | Model | # Tasks | Section |
|---|---|---|---|---|---|
| [`toy_experiments/`](toy_experiments/README.md) | Synthetic MOO | Hàm không lồi (Lin et al., 2019) | — | 2–3+ | §5.1, Appx B,C |
| [`multiMNIST/`](multiMNIST/README.md) | Image Classification | Multi-MNIST, Multi-Fashion, Multi-Fashion+MNIST | LeNet | 2 | §5.2 |
| [`mtr/`](mtr/README.md) | Regression | River Flow / Mississippi River | FNN 64→32→16→8→8 | 8 | §5.2 |
| [`mtc/`](mtc/README.md) | Classification | Multi-task classification | FNN | Nhiều | Appx C |

---

## Các phương pháp được so sánh

### 1. Linear Scalarization (LinScalar)
Minimize tổ hợp tuyến tính: `min θ s(θ) = rᵀl(θ)`
- **Hạn chế:** Không tìm được nghiệm ở **phần lõm** của Pareto front (vì O không lồi)

### 2. MGDA — Multiple Gradient Descent Algorithm (Désidéri, 2012)
Tìm hướng descent trong Convex Hull của gradients; Sener & Koltun (2018) mở rộng cho deep networks.
- **Hạn chế:** Không nhận input preference → nghiệm Pareto tùy ý

### 3. Pareto MTL — PMTL (Lin et al., 2019)
Chia solution space thành K vùng con bằng K reference vectors.
- **Hạn chế:** Cần số reference vectors tăng theo hàm mũ với m; không đảm bảo đúng preference; không bền vững với khởi tạo xa

### 4. EPO Search (Đề xuất — Mahapatra & Rajan, 2020) ⭐
Kết hợp gradient descent và controlled ascent để traverse Pareto front đến đúng preference-specific EPO solution.
- **Ưu điểm:** Đảm bảo hội tụ lý thuyết, bền vững khởi tạo, scale O(m²n)

---

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

**Dependencies chính:**
- PyTorch (deep learning)
- NumPy, SciPy
- GLPK / scipy.optimize (LP solver cho EPO)
- Matplotlib (visualization)

---

## Lý thuyết hội tụ (tóm tắt)

**Theorem 1:** Hướng balancing dbal đảm bảo giảm non-uniformity μᵣ.

**Theorem 2:** Tồn tại step size η₀ > 0 sao cho với mọi η ∈ [0, η₀]:
```
l(θ^{t+1}) ∈ A^r_{lt}   (admissible set)
```
tức là nghiệm mới nằm trong tập chứa EPO solution.

**Corollary 1:** Dãy λᵗ (relative maximum values) đơn điệu giảm → dãy admissible sets hội tụ.

**Claim 2:** Khi EPO solution tồn tại và θ* là regular Pareto optimal, hướng d_nd = 0 khi và chỉ khi θ* ∈ Pᵣ (EPO solution).

---

## Trích dẫn

```bibtex
@inproceedings{mahapatra2020multi,
  title={Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization},
  author={Mahapatra, Debabrata and Rajan, Vaibhav},
  booktitle={Proceedings of the 37th International Conference on Machine Learning},
  pages={6597--6607},
  year={2020},
  organization={PMLR}
}
```
