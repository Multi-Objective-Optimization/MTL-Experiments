# Thí nghiệm Hồi quy Đa mục tiêu: Multi-Target Regression (MTR)

> **Tương ứng với:** Mục 5.2 (Real Data — Regression) trong bài báo  
> **Mahapatra & Rajan, ICML 2020** — *"Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization"*

---

## Mục tiêu thí nghiệm

Đánh giá EPO Search trên bài toán **hồi quy đa mục tiêu** với 8 task đồng thời. Thí nghiệm kiểm tra khả năng mở rộng (scalability) của EPO Search khi số task tăng lên (m = 8), so với PMTL vốn yêu cầu số reference vectors tăng theo hàm mũ.

---

## Bộ dữ liệu: River Flow (Mississippi River)

| Thông số | Giá trị |
|---|---|
| **Nguồn** | Spyromitros-Xioufis et al., 2016 |
| **Bài toán** | Dự báo lưu lượng nước tại 8 trạm đo trên sông Mississippi |
| **Số task (m)** | 8 (mỗi task = dự báo 1 trạm) |
| **Số đặc trưng (features)** | 64 (đo lưu lượng tại các thời điểm 6, 12, 18, 24, 36, 48, 60 giờ trước) |
| **Số target** | 8 |
| **Train set** | 6,300 mẫu |
| **Test set** | 2,700 mẫu |

> **Lý do chọn dataset này:** Dữ liệu đo lưu lượng tại các trạm cùng một con sông tự nhiên có tương quan với nhau → MTL giúp cải thiện dự báo so với học từng task riêng lẻ.

---

## Cấu hình thí nghiệm

| Thành phần | Chi tiết |
|---|---|
| **Kiến trúc mạng** | Fully-connected FNN, 4 lớp: 64 → 32 → 16 → 8 → 8 |
| **Số tham số (n)** | 6,896 |
| **Hàm loss** | Mean Squared Error (MSE) cho mỗi task |
| **Preference vectors** | 20 vectors ngẫu nhiên từ R⁸₊ (Σrⱼ = 1) |
| **Evaluation metric** | **Relative Loss Profile (RLP):** r ⊙ l — tích element-wise giữa preference và loss |

---

## Các phương pháp so sánh

1. **Baseline (Individual):** 8 mô hình FNN riêng biệt, mỗi mô hình học 1 task
2. **LinScalar:** Tổ hợp tuyến tính loss theo preference (gradient bị triệt tiêu nhau)
3. **PMTL (Pareto MTL):** Hiệu quả khi m nhỏ, nhưng số reference vectors cần tăng mũ khi m tăng
4. **EPO Search:** Scale tuyến tính với số chiều gradient → phù hợp khi m lớn

---

## Kết quả kỳ vọng (Hình 8 trong bài báo)

- **EPO Search** có RLP thấp nhất (gần 0) tại hầu hết các task — nghiệm nằm gần preference vector nhất
- **PMTL** gặp khó khăn với m = 8 do không thể đủ reference vectors
- **LinScalar** hoạt động kém do gradient từ các task triệt tiêu nhau
- **Baseline (Individual)** không tận dụng được tương quan giữa các task (dự báo site này giúp cải thiện site khác do đều trên cùng một con sông)

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

Sinh ra biểu đồ **RLP (r ⊙ l)** so sánh mean ± std của 4 phương pháp trên 8 task (tương ứng Hình 8 trong bài báo).

---

## Cấu trúc file

```
mtr/
├── model_fnn.py            # Kiến trúc FNN (64→32→16→8→8)
├── individual_train.py     # Training baseline (individual tasks)
├── linscalar_train.py      # Training với Linear Scalarization
├── epo_train.py            # Training với EPO Search
├── pmtl_train.py           # Training với Pareto MTL
├── epo_lp.py               # Giải LP tìm hướng non-dominating (EPO core)
├── min_norm_solvers.py     # Solver tìm hướng min-norm
├── display_results.py      # Visualize và so sánh RLP
├── latex_utils.py          # Tiện ích xuất bảng LaTeX
├── data/
│   ├── readme.md           # Mô tả dữ liệu River Flow
│   └── DataExploration.ipynb  # Notebook khám phá dữ liệu
└── results/                # Thư mục lưu file .pkl kết quả
```
