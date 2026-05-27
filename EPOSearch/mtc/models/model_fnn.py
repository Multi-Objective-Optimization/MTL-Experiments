"""
FNN Model cho bài toán Multi-Task Classification — Phân loại cảm xúc (model_fnn.py)
======================================================================================
Module này định nghĩa kiến trúc mạng nơ-ron đầy đủ kết nối (Fully-connected Neural
Network — FNN) cho bài toán Multi-Task Classification (MTC) trên tập dữ liệu
phân loại cảm xúc (emotion classification).

Bài toán MTC (Multi-Task Emotion Classification):
    - Input  : Vector đặc trưng văn bản (n_feats chiều).
    - Output : 6 nhãn nhị phân tương ứng với 6 loại cảm xúc:
               [anger, disgust, fear, joy, sadness, surprise].
    - n_tasks = 6 (6 bài toán phân loại nhị phân song song).
    - Loss   : BCELoss (Binary Cross-Entropy) cho từng task.

Kiến trúc (ví dụ với n_feats=2048, n_tasks=6):
    Input (2048,)
        → Linear(2048 → 1024) + Tanh
        → Linear(1024 → 512)  + Tanh
        → Linear(512 → 256)   + Tanh
        → Linear(256 → 128)   + Tanh
        → Linear(128 → 64)    + Tanh
        → Linear(64 → 32)     + Tanh
        → Linear(32 → 16)     + Tanh
        → Linear(16 → 8)      + Tanh
        → Linear(8 → 6)       + Sigmoid  ← output: xác suất từng cảm xúc ∈ (0,1)
    Output (6,) — xác suất độc lập cho 6 cảm xúc

So sánh với mtr/model_fnn.py:
    - MTC dùng BCELoss + Sigmoid output (phân loại nhị phân).
    - MTR dùng MSELoss + không có activation ở output (hồi quy liên tục).
    - Cả hai đều có cùng cấu trúc FNN halving, chỉ khác ở lớp output.

Các lớp chính:
    RegressionTrain : Wrapper tính BCELoss cho từng task.
    RegressionModel : Kiến trúc FNN tự động giảm chiều + Sigmoid output.
    weights_init    : Khởi tạo Xavier uniform với scale 0.1.
"""

import torch
import torch.nn as nn


class RegressionTrain(torch.nn.Module):
    """
    Wrapper module bao ngoài RegressionModel để tính multi-task BCE loss.

    Sử dụng BCELoss (Binary Cross-Entropy) thay vì MSELoss vì đây là
    bài toán phân loại nhị phân (mỗi task dự đoán xác suất 0/1).

    BCELoss = -[y log(ŷ) + (1-y) log(1-ŷ)]

    Attributes:
        model (RegressionModel) : Mô hình FNN với Sigmoid output.
        loss  (nn.BCELoss)      : BCE loss với reduction="none" để tính per-sample.
    """

    def __init__(self, model):
        """
        Khởi tạo RegressionTrain.

        Args:
            model (RegressionModel): Mô hình FNN đã khởi tạo với n_feats và n_tasks.
        """
        super(RegressionTrain, self).__init__()

        self.model = model
        # BCELoss với reduction="none": trả về ma trận loss (B, n_tasks)
        # thay vì scalar → cho phép tính mean theo từng task riêng biệt
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, x, ts):
        """
        Forward pass: tính BCE loss trung bình cho từng task.

        Args:
            x  (torch.Tensor) : Batch input features, shape (B, n_feats).
            ts (torch.Tensor) : Batch nhãn nhị phân, shape (B, n_tasks).
                                 ts[b, i] ∈ {0, 1} — nhãn của mẫu b cho task i.

        Returns:
            torch.Tensor: Vector loss, shape (n_tasks,).
                          loss[i] = mean_over_batch(BCE(ŷ_i, y_i)).
                          Dùng làm đầu vào l cho EPO_LP.get_alpha().

        Lưu ý:
            - Model output ys đã qua Sigmoid → ∈ (0, 1), phù hợp với BCELoss.
            - .mean(dim=0): tính trung bình theo batch → shape (n_tasks,).
        """
        # Forward: (B, n_feats) → (B, n_tasks), giá trị ∈ (0, 1) nhờ Sigmoid
        ys = self.model(x)

        # BCE loss per element: shape (B, n_tasks)
        # Sau đó lấy mean theo batch (dim=0) → (n_tasks,)
        task_loss = self.loss(ys, ts).mean(dim=0)

        return task_loss

    def randomize(self):
        """
        Khởi tạo lại ngẫu nhiên trọng số mô hình bằng Xavier uniform (× 0.1).

        Dùng để reset mô hình về trạng thái ban đầu khi thử nghiệm
        với nhiều điểm khởi đầu hoặc chạy nhiều lần độc lập.
        """
        self.model.apply(weights_init)


def weights_init(m):
    """
    Hàm khởi tạo trọng số Xavier uniform với scale nhỏ (× 0.1).

    Xavier initialization + scale 0.1 giúp:
    - Duy trì phương sai gradient ổn định qua các lớp (Xavier).
    - Trọng số ban đầu đủ nhỏ để Sigmoid không bão hòa ngay từ đầu (scale 0.1).

    Args:
        m (nn.Module): Module PyTorch. Được gọi đệ quy qua model.apply().
                       Chỉ áp dụng cho Conv2d và Linear layers.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # Scale nhỏ để tránh Sigmoid bão hòa (output ~ 0 hoặc ~ 1) ở bước đầu
        m.weight.data *= 0.1


class RegressionModel(torch.nn.Module):
    """
    Kiến trúc FNN tự động cho bài toán multi-task binary classification.

    Mô hình xây dựng chuỗi các lớp Linear với số neurons giảm dần (halving)
    từ n_feats xuống n_tasks. Lớp ẩn dùng Tanh, lớp output dùng Sigmoid.

    Sigmoid ở output đảm bảo mỗi output ∈ (0, 1), phù hợp với:
    - BCELoss (yêu cầu input trong khoảng (0, 1)).
    - Ý nghĩa xác suất: P(task i = 1 | x).

    Ví dụ kiến trúc với n_feats=64, n_tasks=6:
        Linear(64 → 32) → Tanh
        Linear(32 → 16) → Tanh
        Linear(16 → 8)  → Tanh
        Linear(8 → 6)   → Sigmoid  ← output

    So sánh với mtr/model_fnn.py:
        - Giống: Cùng cấu trúc halving, cùng dùng Tanh ở lớp ẩn.
        - Khác: MTC dùng Sigmoid ở output; MTR không dùng activation ở output.

    Attributes:
        n_tasks (int)          : Số lượng task (= số output neurons = 6 cảm xúc).
        layers  (nn.ModuleList): Danh sách các lớp Linear được xây dựng tự động.
    """

    def __init__(self, n_feats, n_tasks):
        """
        Khởi tạo FNN với số lớp tự động tính từ n_feats và n_tasks.

        Thuật toán xây dựng lớp:
            1. Bắt đầu với n_neurons = n_feats.
            2. Lặp: thêm Linear(n_neurons → n_neurons//2), giảm đôi.
            3. Dừng khi n_neurons <= n_tasks.
            4. Thêm lớp output Linear(n_neurons → n_tasks).

        Args:
            n_feats (int) : Số đặc trưng đầu vào. Ví dụ: 2048 cho text features.
            n_tasks (int) : Số lượng task. Ví dụ: 6 cho 6 loại cảm xúc.
        """
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks

        # Danh sách các lớp Linear (tự động xây dựng)
        self.layers = nn.ModuleList()

        n_neurons = n_feats
        # Xây dựng các lớp ẩn: halving cho đến khi n_neurons <= n_tasks
        while n_neurons > n_tasks:
            self.layers.append(nn.Linear(n_neurons, int(n_neurons / 2)))
            n_neurons = int(n_neurons / 2)

        # Lớp output: n_neurons → n_tasks (theo sau bởi Sigmoid trong forward)
        self.layers.append(nn.Linear(n_neurons, n_tasks))

        # Lưu ý: các lớp task_i bên dưới được tạo để tương thích giao diện
        # với PMTL nhưng KHÔNG dùng trong forward(). Đây là artifact từ code gốc.
        for i in range(self.n_tasks):
            setattr(self, "task_{}".format(i), nn.Linear(50, 10))

    def forward(self, x, i=None):
        """
        Forward pass qua mạng FNN với Sigmoid ở lớp output.

        Áp dụng tuần tự:
            - Lớp ẩn: Linear → Tanh
            - Lớp output: Linear → Sigmoid

        Sigmoid đảm bảo output ∈ (0, 1), phù hợp với BCELoss.

        Args:
            x (torch.Tensor) : Batch input features, shape (B, n_feats).
            i (int, optional): Không sử dụng trong implementation này
                               (giữ lại để tương thích giao diện).

        Returns:
            torch.Tensor: Xác suất dự đoán, shape (B, n_tasks).
                          Mỗi giá trị ∈ (0, 1) — xác suất task tương ứng là positive.
        """
        y = x  # (B, n_feats)

        for i in range(len(self.layers)):
            y_temp = self.layers[i](y)

            if i < len(self.layers) - 1:
                # Lớp ẩn: Tanh activation ∈ (-1, 1)
                # Giúp normalize signal qua các lớp, tránh exploding values
                y = torch.tanh(y_temp)
            else:
                # Lớp output: Sigmoid activation ∈ (0, 1)
                # Chuyển logits thành xác suất cho phân loại nhị phân
                # σ(z) = 1 / (1 + e^{-z})
                y = torch.sigmoid(y_temp)

        return y  # shape: (B, n_tasks), mỗi giá trị ∈ (0, 1)
