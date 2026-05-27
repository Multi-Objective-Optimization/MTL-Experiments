# Mã nguồn này được điều chỉnh từ:
# "Pareto Multi-Task Learning" — Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

"""
LeNet-based Multi-Task Learning Model cho bài toán multiMNIST (model_lenet.py)
===============================================================================
Module này định nghĩa kiến trúc mô hình LeNet dùng cho bài toán phân loại
ảnh đa nhiệm vụ (Multi-Task Classification) trên tập dữ liệu multiMNIST.

Bài toán multiMNIST:
    - Input: ảnh grayscale 36x36 chứa 2 chữ số chồng lên nhau.
    - Task 1: Phân loại chữ số bên trái (0–9).
    - Task 2: Phân loại chữ số bên phải (0–9).
    - Hai task dùng chung phần feature extraction (shared trunk),
      mỗi task có một lớp output head riêng.

Kiến trúc:
    Input (1×36×36)
        → Conv2d(1, 10, kernel=9) + ReLU + MaxPool(2)   → (10×14×14)
        → Conv2d(10, 20, kernel=5) + ReLU + MaxPool(2)  → (20×5×5)
        → Flatten → Linear(500, 50) + ReLU              → (50,)
        → [task_0: Linear(50, 10)]                       → logits task 0
        → [task_1: Linear(50, 10)]                       → logits task 1

Các lớp chính:
    RegressionTrain : Wrapper module tính multi-task loss (CrossEntropyLoss).
    RegressionModel : Kiến trúc LeNet với multiple output heads.
    weights_init    : Hàm khởi tạo trọng số Xavier cho Conv2d và Linear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class RegressionTrain(torch.nn.Module):
    """
    Wrapper module bao ngoài RegressionModel để tính multi-task loss.

    Module này đóng gói mô hình LeNet và cung cấp giao diện forward() thuận tiện
    để tính loss cho từng task hoặc tất cả task cùng lúc. Kết quả trả về là
    vector loss (m,) dùng trực tiếp trong EPO_LP.get_alpha().

    Attributes:
        model    (RegressionModel) : Mô hình LeNet chứa feature extractor và output heads.
        weights  (nn.Parameter)    : Trọng số kết hợp task (không dùng trong EPO,
                                     nhưng giữ lại để tương thích với PMTL).
        ce_loss  (CrossEntropyLoss): Hàm loss phân loại đa lớp cho từng task.
    """

    def __init__(self, model, init_weight):
        """
        Khởi tạo RegressionTrain.

        Args:
            model       (RegressionModel) : Mô hình LeNet đã khởi tạo.
            init_weight (np.ndarray)       : Trọng số ban đầu, shape (n_tasks,).
                                             Dùng để khởi tạo self.weights (cho PMTL).
        """
        super(RegressionTrain, self).__init__()

        self.model = model
        # Trọng số dùng cho linear scalarization / PMTL (không dùng trong EPO)
        self.weights = torch.nn.Parameter(torch.from_numpy(init_weight).float())
        # Cross-entropy loss cho bài toán phân loại 10 lớp (0–9)
        self.ce_loss = CrossEntropyLoss()

    def forward(self, x, ts, i=None):
        """
        Forward pass: tính loss cho một hoặc tất cả các task.

        Args:
            x  (torch.Tensor) : Batch input ảnh, shape (B, 1, H, W).
                                 B là batch size.
            ts (torch.Tensor) : Batch nhãn, shape (B, n_tasks).
                                 ts[:, i] là nhãn của task i (0–9).
            i  (int, optional): Index của task cần tính loss đơn lẻ.
                                 - Nếu i không None: chỉ tính loss cho task i.
                                 - Nếu i là None: tính loss cho tất cả task.

        Returns:
            torch.Tensor:
                - Nếu i không None: scalar loss của task i.
                - Nếu i là None: vector loss shape (n_tasks,),
                  dùng làm đầu vào cho EPO_LP.get_alpha().
        """
        if i is not None:
            # Chế độ single-task: forward riêng cho task i
            # Dùng trong individual training hoặc gradient computation riêng lẻ
            y = self.model(x, i)
            return self.ce_loss(y, ts[:, i])

        # Chế độ multi-task: tính loss cho tất cả task
        n_tasks = self.model.n_tasks
        ys = self.model(x)  # shape: (B, n_tasks, 10)

        task_loss = []
        for i in range(n_tasks):
            # ys[:, i, :] → logits của task i, shape (B, 10)
            # ts[:, i]    → nhãn của task i, shape (B,)
            task_loss.append(self.ce_loss(ys[:, i], ts[:, i]))

        # Stack thành vector loss (n_tasks,) để dùng trong EPO
        task_loss = torch.stack(task_loss)
        return task_loss

    def randomize(self):
        """
        Khởi tạo lại ngẫu nhiên trọng số mô hình bằng Xavier uniform.

        Dùng để thử nghiệm với nhiều điểm khởi đầu khác nhau (random initialization),
        như trong thí nghiệm compare_init.py.
        """
        self.model.apply(weights_init)


def weights_init(m):
    """
    Hàm khởi tạo trọng số Xavier uniform cho Conv2d và Linear layers.

    Xavier initialization giữ phương sai của gradient ổn định qua các lớp,
    giúp tránh vanishing/exploding gradient khi train deep networks.

    Args:
        m (nn.Module): Module PyTorch. Hàm này được gọi đệ quy qua model.apply().
                       Chỉ áp dụng Xavier init cho Conv2d và Linear.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


class RegressionModel(torch.nn.Module):
    """
    Kiến trúc LeNet cho bài toán multi-task image classification.

    Mô hình gồm một shared trunk (feature extractor dùng chung) và
    n_tasks output heads độc lập (mỗi task một head).

    Shared trunk (phần dùng chung):
        Conv2d(1→10, kernel=9) → ReLU → MaxPool(2)
        Conv2d(10→20, kernel=5) → ReLU → MaxPool(2)
        Flatten → Linear(500→50) → ReLU

    Output heads (riêng cho từng task):
        task_i: Linear(50→10)  (10 lớp tương ứng 10 chữ số 0–9)

    Cấu trúc shared trunk giúp các task chia sẻ feature representation,
    trong khi các heads độc lập cho phép mỗi task học đặc trưng riêng.

    Attributes:
        n_tasks (int)    : Số lượng task.
        conv1   (Conv2d) : Lớp tích chập đầu tiên: 1→10 channels, kernel 9×9.
        conv2   (Conv2d) : Lớp tích chập thứ hai: 10→20 channels, kernel 5×5.
        fc1     (Linear) : Fully-connected layer: 500→50.
        task_i  (Linear) : Output head của task i: 50→10. (i = 0, ..., n_tasks-1)
    """

    def __init__(self, n_tasks):
        """
        Khởi tạo LeNet với n_tasks output heads.

        Args:
            n_tasks (int): Số lượng task (số lượng output heads).
                           Thông thường n_tasks=2 cho multiMNIST.
        """
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks

        # --- Shared trunk: Feature extractor dùng chung cho tất cả task ---

        # Lớp conv1: (B, 1, 36, 36) → (B, 10, 28, 28) → sau MaxPool: (B, 10, 14, 14)
        # kernel=9, stride=1, padding=0 → output size = (36 - 9 + 1) = 28
        self.conv1 = nn.Conv2d(1, 10, 9, 1)

        # Lớp conv2: (B, 10, 14, 14) → (B, 20, 10, 10) → sau MaxPool: (B, 20, 5, 5)
        # kernel=5, stride=1, padding=0 → output size = (14 - 5 + 1) = 10
        self.conv2 = nn.Conv2d(10, 20, 5, 1)

        # Fully connected: flatten (B, 20, 5, 5) → (B, 500) → (B, 50)
        self.fc1 = nn.Linear(5 * 5 * 20, 50)

        # --- Task-specific output heads ---
        # Mỗi task có một Linear(50 → 10) riêng biệt
        # Sử dụng setattr để tạo động: self.task_0, self.task_1, ...
        for i in range(self.n_tasks):
            setattr(self, "task_{}".format(i), nn.Linear(50, 10))

    def forward(self, x, i=None):
        """
        Forward pass qua mô hình LeNet.

        Args:
            x (torch.Tensor) : Batch input ảnh, shape (B, 1, H, W).
            i (int, optional): Index task để lấy output từ head cụ thể.
                               - Nếu i không None: trả về logits của task i.
                               - Nếu i là None: trả về logits của tất cả task.

        Returns:
            torch.Tensor:
                - Nếu i không None: logits task i, shape (B, 10).
                - Nếu i là None:    logits tất cả task, shape (B, n_tasks, 10).
        """
        # --- Shared trunk ---

        # Conv1 + ReLU + MaxPool(2×2)
        # (B, 1, 36, 36) → (B, 10, 28, 28) → (B, 10, 14, 14)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        # Conv2 + ReLU + MaxPool(2×2)
        # (B, 10, 14, 14) → (B, 20, 10, 10) → (B, 20, 5, 5)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # Flatten: (B, 20, 5, 5) → (B, 500)
        x = x.view(-1, 5 * 5 * 20)

        # Fully connected + ReLU: (B, 500) → (B, 50)
        x = F.relu(self.fc1(x))

        # --- Task-specific heads ---

        if i is not None:
            # Chỉ tính output cho task i (dùng khi tính gradient riêng lẻ từng task)
            layer_i = getattr(self, "task_{}".format(i))
            return layer_i(x)  # shape: (B, 10)

        # Tính output cho tất cả task
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, "task_{}".format(i))
            outs.append(layer(x))  # mỗi phần tử shape: (B, 10)

        # Stack: list of (B, 10) → (B, n_tasks, 10)
        return torch.stack(outs, dim=1)
