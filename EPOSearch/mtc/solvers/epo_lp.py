"""
EPO Linear Program Solver (epo_lp.py)
======================================
Triển khai bộ giải LP (Linear Program) cho thuật toán EPO Search trong bài toán Multi-Task Learning.

EPO Search tìm điểm Pareto Optimal thỏa mãn chính xác preference vector r
do người dùng chỉ định. Ở mỗi bước, bài toán LP được giải để tìm vector
trọng số alpha tối ưu nhằm tổ hợp gradient các task.

Hai chế độ hoạt động chính:
    1. Balancing mode (\mu_r > epsilon):
       Khi điểm hiện tại chưa nằm trên tia preference r, giải LP để
       kéo hướng cập nhật về tia preference (giảm non-uniformity \mu_r).

    2. Descent mode (\mu_r <= epsilon):
       Khi đã gần hoặc nằm trên tia preference, thực hiện Pareto descent
       thuần túy để giảm đồng thời tất cả losses.
"""

import cvxopt
import cvxpy as cp
import numpy as np


class EPO_LP(object):
    """
    Bộ giải LP để tìm vector trọng số gradient tối ưu trong EPO Search.

    Lớp này xây dựng và giải hai bài toán LP riêng biệt tương ứng với
    hai chế độ hoạt động của thuật toán EPO:
      - prob_bal: LP cho chế độ Balancing (kéo về tia preference).
      - prob_dom: LP cho chế độ Descent với ràng buộc Restricted.
      - prob_rel: LP cho chế độ Descent với ràng buộc Relaxed.

    Ký hiệu toán học:
        m  : số lượng task (số chiều của preference vector và loss vector)
        n  : số lượng tham số mô hình (model parameters)
        r  : preference vector (m,), r_i > 0, Σ r_i = 1
        l  : loss vector (m,), l_i là loss của task i
        G  : gradient matrix (m, n), G[i] = ∇_θ l_i(θ)
        C  : Gram matrix (m, m), C = G @ G.T, C_ij = <∇l_i, ∇l_j>
        alpha  : combination weights (m,), alpha_i >= 0, Σ alpha_i = 1
        d  : hướng cập nhật tổng hợp = Σ alpha_i ∇l_i = G.T @ alpha
    """

    def __init__(self, m, n, r, eps=1e-4):
        """
        Khởi tạo EPO_LP và xây dựng sẵn cấu trúc hai bài toán LP.

        Các bài toán LP được xây dựng một lần trong __init__ với
        cp.Parameter (có thể thay đổi giá trị mà không cần build lại),
        giúp tăng hiệu quả khi gọi get_alpha() nhiều lần.

        Args:
            m   (int)          : Số lượng task / objectives.
            n   (int)          : Số lượng tham số mô hình (model parameters).
            r   (np.ndarray)   : Preference vector, shape (m,).
                                 Xác định tỉ lệ tương đối giữa các task.
                                 Ví dụ: r = [0.5, 0.5] nghĩa là ưu tiên đều 2 task.
            eps (float)        : Ngưỡng non-uniformity \mu_r để chuyển chế độ.
                                 - \mu_r >  eps → Balancing mode.
                                 - \mu_r <= eps → Descent mode.
                                 Mặc định: 1e-4.
        """
        # Tắt output log của solver GLPK để giữ terminal sạch
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

        # --- Lưu siêu tham số ---
        self.m = m          # Số task
        self.n = n          # Số tham số mô hình
        self.r = r          # Preference vector (cố định trong suốt quá trình train)
        self.eps = eps      # Ngưỡng chuyển chế độ
        self.last_move = None  # Ghi nhận chế độ cuối cùng: "bal" hoặc "dom"

        # -----------------------------------------------------------------------
        # Khai báo các cp.Parameter — giá trị được cập nhật mỗi lần gọi get_alpha()
        # -----------------------------------------------------------------------

        # a: Vector điều chỉnh (adjustment), shape (m,)
        # a_i = r_i * (log(l̂_i * m) - \mu_r), dùng để định hướng về tia preference
        self.a = cp.Parameter(m)

        # C: Gram matrix của gradients, shape (m, m)
        # C_ij = <∇l_i, ∇l_j> — đo sự tương quan hướng gradient giữa 2 task
        self.C = cp.Parameter((m, m))

        # Ca: Tích C @ a, shape (m,)
        # Đây là d_bal^T @ G — hướng về tia preference chiếu lên không gian gradient
        self.Ca = cp.Parameter(m)

        # rhs: Vế phải của ràng buộc bất đẳng thức trong Balancing LP, shape (m,)
        self.rhs = cp.Parameter(m)

        # -----------------------------------------------------------------------
        # Biến tối ưu
        # -----------------------------------------------------------------------

        # alpha: Vector trọng số cần tìm, shape (m,)
        # Ràng buộc: α_i >= 0, Σ α_i = 1 (nằm trên simplex chuẩn)
        self.alpha = cp.Variable(m)

        # -----------------------------------------------------------------------
        # LP 1: Balancing Problem (prob_bal)
        # Dùng khi \mu_r > epsilon (điểm hiện tại CHƯA nằm trên tia preference)
        #
        # Mục tiêu: Maximize α^T C a  ≡  Maximize <d_α, d_bal>
        #   → Tìm hướng d_α = G^T α gần nhất với hướng cân bằng d_bal = G^T a
        #
        # Ràng buộc:
        #   α >= 0                    (trọng số không âm)
        #   Σα_i = 1                  (nằm trên simplex)
        #   (C @ α)_i >= rhs_i        (controlled ascent: cho phép một số loss tăng
        #                              có kiểm soát khi kéo về preference)
        # -----------------------------------------------------------------------
        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # Objective: căn chỉnh về d_bal
        constraints_bal = [
            self.alpha >= 0,                  # Trọng số không âm
            cp.sum(self.alpha) == 1,          # Simplex constraint
            self.C @ self.alpha >= self.rhs,  # Controlled ascent constraint
        ]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)

        # -----------------------------------------------------------------------
        # LP 2: Descent Problem (prob_dom & prob_rel)
        # Dùng khi \mu_r <= epsilon (điểm đã GẦN tia preference)
        #
        # Mục tiêu: Maximize Σ_ij (α^T C)_ij  ≡  Maximize ||G^T α||²
        #   → Tìm hướng descent mạnh nhất trên Pareto front
        #
        # prob_dom (Restricted descent):
        #   Thêm ràng buộc: α^T Ca >= -max(0, Ca_i)
        #   → Bảo đảm không đi ngược hướng cân bằng quá mức
        #
        # prob_rel (Relaxed descent):
        #   Chỉ yêu cầu: C @ α >= 0
        #   → Đảm bảo hướng descent là Pareto (không task nào tăng loss)
        # -----------------------------------------------------------------------
        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # Maximize norm của hướng

        # Restricted descent: thêm ràng buộc trên thành phần cân bằng
        constraints_res = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,                              # Simplex
            self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),    # Không đi ngược d_bal
            self.C @ self.alpha >= 0,                             # Pareto descent
        ]

        # Relaxed descent: chỉ yêu cầu Pareto descent
        constraints_rel = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,          # Simplex
            self.C @ self.alpha >= 0,         # Pareto descent (tất cả task không tăng)
        ]

        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP Restricted descent
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP Relaxed descent

        # Lưu kết quả tối ưu LP gần nhất (giá trị objective)
        self.gamma = 0

        # Lưu giá trị non-uniformity \mu_r gần nhất
        self.mu_rl = 0

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        """
        Giải LP để tính vector trọng số alpha tối ưu cho bước gradient.

        Hàm này thực hiện toàn bộ logic của một bước EPO:
          1. Tính non-uniformity \mu_r từ loss vector l và preference r.
          2. Nếu \mu_r > epsilon → giải Balancing LP (kéo về tia preference).
          3. Nếu \mu_r <= epsilon → giải Descent LP (tối ưu Pareto descent).

        Hướng cập nhật tham số sau khi có alpha:
            d = G.T @ alpha      (trong không gian tham số θ)
            θ ← θ - η * d   (gradient descent step)

        Args:
            l      (np.ndarray) : Loss vector, shape (m,).
                                  l_i là scalar loss của task i tại θ hiện tại.
            G      (np.ndarray) : Gradient matrix, shape (m, n) hoặc (m, m) nếu C=True.
                                  G[i] = ∇_θ l_i(θ), mỗi hàng là gradient của 1 task.
            r      (np.ndarray) : Preference vector override, shape (m,).
                                  Nếu None, dùng self.r được đặt khi khởi tạo.
            C      (bool)       : Nếu True, G đã là Gram matrix C = G @ G.T (m, m).
                                  Nếu False (mặc định), tính C = G @ G.T bên trong.
            relax  (bool)       : Nếu True, dùng Relaxed descent (prob_rel).
                                  Nếu False (mặc định), dùng Restricted descent (prob_dom).

        Returns:
            alpha (np.ndarray) : Vector trọng số tối ưu, shape (m,).
                                 alpha_i >= 0, Σ alpha_i = 1.
                                 Dùng để tổ hợp gradient: d = Σ alpha_i ∇l_i.

        Raises:
            AssertionError: Nếu len(l), len(G), len(r) không bằng self.m.
        """
        # Dùng preference vector mặc định nếu không truyền vào
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"

        # Bước 1: Tính rl = r ⊙ l, non-uniformity \mu_r, và adjustment vector a
        #   - rl    : preference-weighted loss, shape (m,)
        #   - mu_rl : scalar non-uniformity \mu_r = KL(l̂ ‖ 1/m), đo khoảng cách từ tia r
        #   - a     : adjustment vector, a_i = r_i * (log(l̂_i * m) - \mu_r)
        rl, self.mu_rl, self.a.value = adjustments(l, r)

        # Bước 2: Cập nhật Gram matrix C = G @ G.T (hoặc dùng trực tiếp nếu C=True)
        # C_ij = <∇l_i, ∇l_j> — inner product của gradient giữa task i và task j
        self.C.value = G if C else G @ G.T

        # Bước 3: Tính Ca = C @ a — hướng cân bằng chiếu lên không gian loss
        # Ca_i > 0: task i cần giảm loss để về preference
        # Ca_i < 0: task i đang quá thấp so với preference, có thể tăng
        self.Ca.value = self.C.value @ self.a.value

        # -----------------------------------------------------------------------
        # Bước 4: Chọn chế độ và giải LP
        # -----------------------------------------------------------------------
        if self.mu_rl > self.eps:
            # --- BALANCING MODE ---
            # Điểm hiện tại CHƯA nằm trên tia preference (non-uniformity cao)
            # → Cần kéo về đúng tia r (controlled ascent cho phép)

            # J: tập task đang "trên mức" preference (Ca_i > 0, cần giảm)
            J = self.Ca.value > 0

            if len(np.where(J)[0]) > 0:
                # Tồn tại ít nhất một task "trên mức"
                # J_star: task có weighted loss lớn nhất (cần ưu tiên giảm nhất)
                J_star_idx = np.where(rl == np.max(rl))[0]

                # Xây dựng RHS cho ràng buộc controlled ascent:
                #   - Task j ∈ J (trên mức): rhs_j = Ca_j (buộc phải giảm)
                #   - Task j ∉ J (dưới mức): rhs_j = -inf (không ràng buộc, có thể tăng)
                #   - Task j* (lớn nhất): rhs_j* = 0 (ít nhất không được tăng)
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf  # Các task dưới mức: cho phép tăng tự do
                self.rhs.value[J_star_idx] = 0  # Task lớn nhất: không tăng
            else:
                # Tất cả task đều "dưới mức" preference (chưa tối ưu hoàn toàn)
                # → Chỉ cần đảm bảo hướng descent không phá vỡ constraint
                self.rhs.value = np.zeros_like(self.Ca.value)

            # Giải Balancing LP
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "bal"  # Ghi nhận chế độ

        else:
            # --- DESCENT MODE ---
            # Điểm đã gần hoặc nằm trên tia preference (\mu_r <= epsilon)
            # → Thực hiện Pareto descent thuần túy để giảm tất cả losses

            if relax:
                # Relaxed: chỉ đảm bảo Pareto (tất cả task không tăng)
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                # Restricted: thêm ràng buộc để không đi ngược hướng cân bằng
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)

            self.last_move = "dom"  # Ghi nhận chế độ

        return self.alpha.value


# ==============================================================================
# Hàm tiện ích
# ==============================================================================

def mu(rl, normed=False):
    """
    Tính non-uniformity \mu_r của phân phối weighted loss.

    Non-uniformity đo khoảng cách giữa phân phối l̂ và phân phối 
    đều 1/m. Cụ thể, \mu_r = KL(l̂ ‖ 1/m) = Σ l̂_i * log(m * l̂_i).

    Args:
        rl     (np.ndarray) : Vector r ⊙ l (preference-weighted losses), shape (m,).
                              Tất cả phần tử phải >= 0.
        normed (bool)       : Nếu True, rl đã được normalize (Σrl_i = 1).
                              Nếu False (mặc định), hàm tự normalize: l̂ = rl / Σrl.

    Returns:
        float : Giá trị non-uniformity \mu_r >= 0.
    Raises:
        ValueError: Nếu có phần tử rl_i < 0 (weighted loss âm — không hợp lệ).
    """
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")

    m = len(rl)
    # Normalize thành phân phối xác suất l̂
    l_hat = rl if normed else rl / rl.sum()

    # Loại bỏ các phần tử quá nhỏ (epsilon machine) để tránh log(0)
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]

    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    """
    Tính adjustment vector a dùng cho Balancing LP.

    Công thức:
        a_i   = r_i * (log(l̂_i * m) - \mu_f)

    Ý nghĩa của a_i:
        - a_i > 0: Task i đang có weighted loss cao hơn mức trung bình → cần giảm.
        - a_i < 0: Task i đang có weighted loss thấp hơn mức trung bình → có thể tăng.
        - a_i = 0: Task i đang đúng mức preference.

    Args:
        l (np.ndarray) : Loss vector, shape (m,). l_i là loss của task i.
        r (np.ndarray or float) : Preference vector, shape (m,) hoặc scalar.
                                  Mặc định r=1 (uniform preference).

    Returns:
        tuple:
            rl     (np.ndarray) : Preference-weighted loss r ⊙ l, shape (m,).
            mu_rl  (float)      : Non-uniformity \mu_r = KL(l̂ ‖ 1/m).
            a      (np.ndarray) : Adjustment vector, shape (m,).
    """
    m = len(l)

    # Tính preference-weighted loss: rl_i = r_i * l_i
    rl = r * l

    # Normalize
    l_hat = rl / rl.sum()

    # Tính non-uniformity \mu_r = KL(l̂ ‖ 1/m)
    mu_rl = mu(l_hat, normed=True)

    # Tính adjustment: a_i = r_i * (log(l̂_i * m) - \mu_r)
    a = r * (np.log(l_hat * m) - mu_rl)

    return rl, mu_rl, a
