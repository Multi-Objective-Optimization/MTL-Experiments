# lenet base model for Pareto MTL
"""
FNN Model cho bài toán Multi-Task Regression — Dự báo lưu lượng sông (model_fnn.py)
=====================================================================================
Module này định nghĩa kiến trúc mạng nơ-ron đầy đủ kết nối (Fully-connected Neural
Network — FNN) cho bài toán Multi-Task Regression (MTR) trên dữ liệu lưu lượng sông
Mississippi.

Bài toán MTR (Multi-Task Regression):
    - Input  : 64 đặc trưng môi trường (nhiệt độ, lượng mưa, v.v.).
    - Output : Lưu lượng nước tại 8 trạm đo khác nhau dọc sông Mississippi.
    - n_tasks = 8 (8 bài toán hồi quy song song, dùng chung feature extractor).
    - Loss   : MSELoss cho từng task → vector loss (8,) để EPO_LP tối ưu.

Kiến trúc (ví dụ với n_feats=64, n_tasks=8):
    Input (64,)
        → Linear(64 → 32) + Tanh
        → Linear(32 → 16) + Tanh
        → Linear(16 → 8)             ← output layer (không có activation)
    Output (8,) — 8 giá trị hồi quy liên tục

Lưu ý thiết kế:
    - Số lớp và kích thước được xác định TỰ ĐỘNG bằng cách chia đôi n_neurons
      cho đến khi n_neurons <= n_tasks.
    - Lớp ẩn dùng Tanh activation (bounded, phù hợp với hồi quy).
    - Lớp output KHÔNG có activation (raw regression output).
    - Không có task-specific heads (khác với model_lenet.py):
      toàn bộ mạng là shared trunk, output là tất cả n_tasks giá trị.

Các lớp chính:
    RegressionTrain : Wrapper tính MSELoss cho từng task.
    RegressionModel : Kiến trúc FNN với chiều rộng tự động giảm dần.
    weights_init    : Khởi tạo Xavier uniform với scale 0.1.
"""

import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss


class RegressionTrain(torch.nn.Module):
    """
    Wrapper module bao ngoài RegressionModel để tính multi-task MSE loss.

    Module này cung cấp giao diện forward() để tính vector loss (n_tasks,)
    từ batch dữ liệu đầu vào. Kết quả được dùng trực tiếp làm đầu vào l
    cho EPO_LP.get_alpha().

    Attributes:
        model    (RegressionModel) : Mô hình FNN đã khởi tạo.
        mse_loss (MSELoss)         : Hàm Mean Squared Error loss cho hồi quy.
    """

    def __init__(self, model):
        """
        Khởi tạo RegressionTrain.

        Args:
            model (RegressionModel): Mô hình FNN đã khởi tạo với n_feats và n_tasks.
        """
        super(RegressionTrain, self).__init__()

        self.model = model
        # MSELoss cho bài toán hồi quy (giá trị liên tục)
        # Tính loss riêng cho từng task: loss_i = mean((ŷ_i - y_i)²)
        self.mse_loss = MSELoss()

    def forward(self, x, ts):
        """
        Forward pass: tính MSE loss cho tất cả n_tasks task.

        Args:
            x  (torch.Tensor) : Batch input features, shape (B, n_feats).
                                 B là batch size, n_feats là số đặc trưng đầu vào.
            ts (torch.Tensor) : Batch ground truth targets, shape (B, n_tasks).
                                 ts[:, i] là target của task i.

        Returns:
            torch.Tensor: Vector loss, shape (n_tasks,).
                          loss[i] = MSE(ŷ_i, y_i) = mean((ŷ_i - y_i)²).
                          Dùng làm đầu vào l cho EPO_LP.get_alpha().
        """
        n_tasks = self.model.n_tasks

        # Forward qua mô hình: (B, n_feats) → (B, n_tasks)
        ys = self.model(x)

        task_loss = []
        for i in range(n_tasks):
            # ys[:, i]: dự đoán của task i, shape (B,)
            # ts[:, i]: target thực tế của task i, shape (B,)
            task_loss.append(self.mse_loss(ys[:, i], ts[:, i]))

        # Stack thành vector loss (n_tasks,)
        task_loss = torch.stack(task_loss)
        return task_loss

    def randomize(self):
        """
        Khởi tạo lại ngẫu nhiên trọng số mô hình bằng Xavier uniform (× 0.1).

        Dùng để reset mô hình về trạng thái ban đầu khi thử nghiệm
        với nhiều điểm khởi đầu khác nhau hoặc chạy nhiều lần lặp.
        """
        self.model.apply(weights_init)


def weights_init(m):
    """
    Hàm khởi tạo trọng số Xavier uniform với scale nhỏ (× 0.1).

    Scale 0.1 giúp trọng số ban đầu nhỏ, tránh các output lớn ở bước đầu
    huấn luyện (đặc biệt quan trọng với Tanh activation, dễ bão hòa).

    Args:
        m (nn.Module): Module PyTorch. Được gọi đệ quy qua model.apply().
                       Chỉ áp dụng cho Conv2d và Linear layers.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # Scale nhỏ hơn để tránh saturation của Tanh ở các bước đầu training
        m.weight.data *= 0.1


class RegressionModel(torch.nn.Module):
    """
    Kiến trúc FNN tự động cho bài toán multi-task regression.

    Mô hình xây dựng một chuỗi các lớp Linear với kích thước giảm dần
    (halving) từ n_feats xuống n_tasks, sử dụng Tanh activation cho các
    lớp ẩn. Lớp cuối không có activation (raw regression output).

    Ví dụ kiến trúc với n_feats=64, n_tasks=8:
        Linear(64 → 32) → Tanh
        Linear(32 → 16) → Tanh
        Linear(16 → 8)         ← output, không có activation

    Lưu ý:
        - Không có task-specific heads: toàn bộ network là shared.
        - Output là (B, n_tasks) — tất cả task predictions trong 1 forward pass.
        - Không dùng Sigmoid ở output (khác mtc/model_fnn.py):
          regression cần output không giới hạn.

    Attributes:
        n_tasks (int)          : Số lượng task / output neurons.
        layers  (nn.ModuleList): Danh sách các lớp Linear được xây dựng tự động.
    """

    def __init__(self, n_feats, n_tasks):
        """
        Khởi tạo FNN với số lớp tự động tính từ n_feats và n_tasks.

        Thuật toán xây dựng lớp:
            1. Bắt đầu với n_neurons = n_feats.
            2. Lặp: thêm Linear(n_neurons → n_neurons//2), giảm đôi n_neurons.
            3. Dừng khi n_neurons <= n_tasks.
            4. Thêm Linear cuối: n_neurons → n_tasks.

        Args:
            n_feats (int) : Số đặc trưng đầu vào. Ví dụ: 64 cho dữ liệu sông.
            n_tasks (int) : Số lượng task (output). Ví dụ: 8 cho 8 trạm đo.
        """
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks

        # Danh sách các lớp được xây dựng tự động (Dynamic layer construction)
        self.layers = nn.ModuleList()

        n_neurons = n_feats
        # Tạo các lớp ẩn: halving số neurons cho đến khi <= n_tasks
        while n_neurons > n_tasks:
            self.layers.append(nn.Linear(n_neurons, int(n_neurons / 2)))
            n_neurons = int(n_neurons / 2)

        # Lớp output cuối cùng: n_neurons → n_tasks
        # (n_neurons tại đây <= n_tasks, hoặc đúng bằng n_tasks nếu n_feats = 2^k * n_tasks)
        self.layers.append(nn.Linear(n_neurons, n_tasks))

        # Lưu ý: các lớp task-specific head bên dưới (task_i) được tạo để
        # tương thích giao diện với PMTL nhưng KHÔNG dùng trong forward().
        # Đây là artifact từ code gốc — không xóa để duy trì compatibility.
        for i in range(self.n_tasks):
            setattr(self, "task_{}".format(i), nn.Linear(50, 10))

    def forward(self, x, i=None):
        """
        Forward pass qua mạng FNN.

        Áp dụng tuần tự các lớp Linear với Tanh activation ở các lớp ẩn.
        Lớp output cuối không có activation.

        Args:
            x (torch.Tensor) : Batch input, shape (B, n_feats).
            i (int, optional): Không sử dụng trong implementation này
                               (giữ lại để tương thích giao diện).

        Returns:
            torch.Tensor: Predictions, shape (B, n_tasks).
                          Giá trị raw (không qua activation) — phù hợp với MSELoss.
        """
        y = x  # (B, n_feats)

        for i in range(len(self.layers)):
            y_temp = self.layers[i](y)

            if i < len(self.layers) - 1:
                # Lớp ẩn: áp dụng Tanh activation
                # Tanh ∈ (-1, 1), giúp normalize output giữa các lớp
                y = torch.tanh(y_temp)
            else:
                # Lớp output cuối: KHÔNG có activation (raw regression)
                # Để giá trị tự do, không bị giới hạn — phù hợp với hồi quy
                y = y_temp

        return y  # shape: (B, n_tasks)
