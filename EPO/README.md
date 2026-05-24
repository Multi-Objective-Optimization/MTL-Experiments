# Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization - ICML 2020 

## 1. Abstract
- Một yêu cầu phổ biến trong các ứng dụng MTL mà các phương pháp cũ không thể giải quyết được là tìm một nghiệm thỏa mãn các ưu tiên do người dùng chỉ định đối với các hàm mất mát theo từng tác vụ. 
- Chúng tôi phát triển thuật toán MTL đa mục tiêu dựa trên gradient đầu tiên để giải quyết vấn đề này. Cách tiếp cận độc đáo của chúng tôi kết hợp nhiều bước giảm gradient (multiple gradient descent) với bước tăng được kiểm soát cẩn thận (carefully controlled ascent) để duyệt biên Pareto theo một cách có nguyên tắc, điều này cũng làm cho nó bền vững với khởi tạo. 
- Khả năng mở rộng của thuật toán cho phép sử dụng nó trong các mạng sâu quy mô lớn cho MTL. Chỉ giả định tính khả vi của các hàm mất mát theo tác vụ, chúng tôi cung cấp các đảm bảo lý thuyết về sự hội tụ. 
- Các thí nghiệm cho thấy thuật toán của chúng tôi vượt trội so với các phương 
pháp cạnh tranh tốt nhất trên các bộ dữ liệu chuẩn.

## 2. Introduction
Tính hiệu quả của tối ưu hóa đa mục tiêu cho MTL lần đầu tiên được chứng minh bởi \cite{Sener2018}. Thuật toán của họ mở rộng thuật toán MGDA \cite{Desideri2012} để xử lý các gradient có số chiều cao, từ đó phù hợp với MTL quy mô lớn với mạng sâu. Tuy nhiên, phương pháp của họ tìm ra một nghiệm tùy ý từ tập Pareto và không thể được sử dụng để khám phá các nghiệm có sự đánh đổi khác nhau. Hạn chế này được nhận ra bởi \cite{Lin2019}, những người giải quyết một phần vấn đề này bằng cách phân tách bài toán MTL và giải nhiều bài toán con với các sự đánh đổi khác nhau. Phương pháp của họ cho ra một tập hợp các nghiệm Pareto tối ưu phân bố trên biên Pareto.

Trong nhiều ứng dụng MTL, người dùng có thể muốn khám phá các nghiệm với các sự đánh đổi cụ thể dưới dạng sở thích hoặc mức độ ưu tiên giữa các tác vụ. Cho trước các sở thích $r_j$ cho mỗi tác vụ, yêu cầu một nghiệm Pareto tối ưu sao cho với hai tác vụ bất kỳ, nếu $r_i \geq r_j$ thì các mất mát tương ứng thỏa mãn $l_i \leq l_j$. Chúng tôi gọi một nghiệm như vậy là nghiệm Pareto tối ưu theo sở thích cụ thể (preference-specific Pareto optimal).

Việc tìm các nghiệm Pareto tối ưu theo sở thích cụ thể là thách thức và không thể giải quyết bằng tuyến tính hóa vô hướng hay các phương pháp MTL đa mục tiêu hiện có. Một vector sở thích xác định một hướng và dẫn đến một điểm trên biên Pareto. Các phương pháp hiện tại không thể được sử dụng để đạt đến một điểm cụ thể trên biên Pareto. Các đóng góp của chúng tôi trong bài báo này là:

- Chúng tôi phát triển thuật toán MTL đa mục tiêu dựa trên gradient đầu tiên, được gọi là Tìm kiếm Pareto Tối ưu Chính xác (Exact Pareto Optimal Search) để tìm một nghiệm Pareto tối ưu theo sở thích cụ thể.

- Cách tiếp cận độc đáo của EPO Search kết hợp gradient descent và bước tăng được kiểm soát cẩn thận, cho phép nó: 
    - Duyệt biên Pareto cho đến khi nghiệm yêu cầu được tìm thấy, làm cho nó bền vững với khởi tạo
    - Tìm nghiệm Pareto tối ưu gần nhất với sở thích nếu nghiệm chính xác không tồn tại
    - Tìm nhiều nghiệm trên biên Pareto theo cách có nguyên tắc nếu có nhiều sở thích
    - Mở rộng tuyến tính với số chiều gradient và do đó huấn luyện hiệu quả các mạng sâu quy mô lớn cho MTL

- Giả định tính khả vi của các hàm mất mát (không cần hàm lồi), chúng tôi chứng minh rằng EPO Search hội tụ đến nghiệm Pareto tối ưu chính xác theo sở thích cụ thể.

- Các thí nghiệm trên dữ liệu tổng hợp và thực tế chứng minh sự vượt trội của EPO Search so với các phương pháp tiên tiến nhất.

## 3. Preliminaries
Chúng tôi xem xét $m$ tác vụ, mỗi tác vụ có hàm mục tiêu không âm $l_j : \mathbb{R}^n \to \mathbb{R}_+$, $j \in [m]$. Hàm vector $l : \mathbb{R}^n \to \mathbb{R}^m$ là ánh xạ từ không gian nghiệm (solution space) $\mathbb{R}^n$ đến không gian mục tiêu (objective space) $\mathbb{R}^m$. Chúng tôi sử dụng $l$ để ký hiệu cả hàm mất mát lẫn một điểm trong $\mathbb{R}^m$, tùy theo ngữ cảnh. Miền giá trị của $l$, ký hiệu là $\mathcal{O}$, là một tập con của nón dương:

$$
\mathbb{R}^m_+ := \{l \in \mathbb{R}^m \mid l_j \geq 0\ \forall j \in [m]\}. \tag{1}
$$

Quan hệ thứ tự bộ phận cho hai điểm $l^1, l^2 \in \mathbb{R}^m$, ký hiệu $l^1 \geqslant l^2$, được định nghĩa bởi $l^1 - l^2 \in \mathbb{R}^m_+$, tức là $l^1_j \geq l^2_j$ với mọi $j \in [m]$ và bất đẳng thức nghiêm ngặt $l^1 > l^2$ xảy ra khi tồn tại ít nhất một $j$ sao cho $l^1_j > l^2_j$. Về mặt hình học, $l^1 > l^2$ có nghĩa là $l^1$ nằm trong nón dương tại $l^2$, tức là $l^1 \in \{l^2\} + \mathbb{R}^m_+ := \{l^2 + l \mid l \in \mathbb{R}^m\}$ và $l^1 \neq l^2$.

Trong bối cảnh tối thiểu hóa, nghiệm $\theta^1 \in \mathbb{R}^n$ bị thống trị bởi nghiệm khác $\theta^2 \in \mathbb{R}^n$ khi và chỉ khi $l(\theta^1) \geqslant l(\theta^2)$. Lưu ý rằng $l(\theta^1) \not\geqslant l(\theta^2)$ nếu $l(\theta^1) \notin \{l(\theta^2)\} + \mathbb{R}^m_+$. Nghiệm $\theta^*$ là Pareto tối ưu nếu nó không bị thống trị bởi bất kỳ nghiệm nào khác. Tập tất cả các nghiệm Pareto tối ưu toàn cục được cho bởi:

$$
\mathcal{P}_{glo} := \{\theta^* \in \mathbb{R}^n \mid \forall \theta \in \mathbb{R}^n - \{\theta^*\},\ l(\theta^*) \not\geqslant l(\theta)\}. \tag{2}

$$

Chúng tôi quan tâm đến các nghiệm Pareto tối ưu cục bộ:

$$
\mathcal{P} := \left\{\theta^* \in \mathbb{R}^n \;\middle|\; \begin{array}{l} \exists\, \mathcal{N}(\theta^*) \subset \mathbb{R}^n \mid \\ \forall \theta \in \mathcal{N}(\theta^*) - \{\theta^*\},\\ l(\theta^*) \not\geqslant l(\theta) \end{array} \right\}, \tag{3}
$$
trong đó $\mathcal{N}(\theta^*)$ là một lân cận mở của $\theta^*$. Lưu ý rằng $\mathcal{P}_{glo} \subset \mathcal{P}$. Tập các giá trị đa mục tiêu của các nghiệm Pareto tối ưu, $l(\mathcal{P}) \subset \mathcal{O}$, được gọi là biên Pareto (Pareto front).

### 3.1 Gradient-based Multi-Objective Optimization
Trong MOO dựa trên gradient, chúng tôi tìm nghiệm Pareto tối ưu bằng cách bắt đầu từ khởi tạo tùy ý $\theta^0 \in \mathbb{R}^n$ và lặp đi lặp lại tìm nghiệm tiếp theo $\theta^{t+1}$ thống trị nghiệm trước $\theta^t$ (tức là $l^{t+1} \leqslant l^t$, trong đó $l^t := l(\theta^t)$), bằng cách di chuyển $\theta^{t+1} = \theta^t - \eta d$, sao cho bước giảm xảy ra ở mọi mục tiêu: $l^{t+1}_j \leqslant l^t_j$. Điều này chỉ có thể xảy ra nếu $d$ có góc dương với gradient của mọi hàm mục tiêu tại $\theta^t$.

Đặt $g_j = \nabla_\theta l_j$ là gradient của hàm mục tiêu thứ $j$ tại $\theta$, và $G \in \mathbb{R}^{n \times m}$ là ma trận có $g_j$ là cột thứ $j$. Hướng giảm (descent direction) $d_{des}$ được cho bởi $d_{des}^T g_j \geq 0$ với mọi $j \in [m]$. Do đó, di chuyển ngược chiều $d_{des}$, bắt đầu từ $\theta$, dẫn đến sự giảm giá trị mục tiêu, không có thay đổi khi $d_{des}^T g_j = 0$.

\cite{Desideri2012} chứng minh rằng các hướng giảm có thể được tìm thấy trong bao lồi (Convex Hull) của các gradient, được định nghĩa bởi:
$$
\mathcal{CH}_\theta := \{G\beta \mid \beta \in \mathcal{S}^m\}, \tag{4}
$$
trong đó:

$$
\mathcal{S}^m := \left\{\beta \in \mathbb{R}^m_+ \;\middle|\; \sum_{j=1}^m \beta_j = 1\right\} \tag{5}
$$
là đơn hình chính quy (regular simplex) $m$-chiều, và MGDA hội tụ đến một nghiệm Pareto tối ưu cục bộ bằng cách lặp đi lặp lại sử dụng hướng giảm:

$$
d^* = \arg\min_{d \in \mathcal{CH}_\theta} \|d\|_2^2. \tag{6}
$$

\cite{Sener2018} thiết kế phương pháp giải (6) có khả năng mở rộng cho các gradient số chiều cao.

### 3.2 Problem Statement
Cho trước các sở thích tương đối cho mỗi tác vụ $r_j > 0$, $j \in [m]$, chúng tôi muốn tìm nghiệm Pareto tối ưu $\theta_r^* \in \mathcal{P}$ sao cho, nếu $r_i \geq r_{j'}$ thì $l_i(\theta_r^*) \leq l_{j'}(\theta_r^*)$.

**Hạn chế của các phương pháp hiện tại:** Chúng tôi thảo luận ngắn gọn về các cách tiếp cận khả thi để giải bài toán này với các phương pháp MTL hiện có và các hạn chế của chúng. 

- Xét tuyến tính hóa vô hướng sử dụng SOO trong đó các sở thích có thể là trọng số theo tác vụ:

$$
\theta^* = \arg\min_\theta s(\theta) = r^T l(\theta). \tag{7}
$$

- Như đã thảo luận trong \cite{Boyd2004}[Ch 4.7], nếu $\mathcal{O}$ không lồi trong $\mathbb{R}^m$ thì có thể không tìm được $\theta_r^*$ như vậy; và $\theta_r^*$ mong muốn chỉ có thể được tìm thấy nếu $\mathcal{O}$ lồi gần $l(\theta_r^*)$ và khởi tạo $\theta^0$ để giải (7) đủ gần $\theta_r^*$.

Trong các thuật toán dựa trên MGDA (ví dụ \cite{Sener2018}), sử dụng $d_{des}$ chúng tôi chỉ có thể tìm được nghiệm thống trị nghiệm trước, mà không có bất kỳ kiểm soát nào về việc di chuyển về phía sở thích. Do đó, tùy thuộc vào khởi tạo $\theta^0$, thuật toán có thể đạt đến bất kỳ nghiệm Pareto tối ưu cục bộ nào. Điều này cũng đã được xác minh bằng thực nghiệm bởi \cite{Lin2019}.

\noindent
Thuật toán Pareto MTL (PMTL) của \cite{Lin2019} tìm nhiều nghiệm trên biên Pareto. Thuật toán của họ sử dụng nhiều vector tham chiếu $u_k$, $k = 1, \ldots, K$ để phân vùng không gian nghiệm thành $K$ vùng con $\Omega_k := \{\theta \in \mathbb{R}^n \mid u_k^T l(\theta) \geq u_{k'}^T l(\theta)\ \forall k' \neq k\}$ và sau đó có hai pha. Trong pha đầu tiên, bắt đầu từ một điểm ban đầu, họ tìm điểm $\theta^0_0 \in \Omega_k$, sao cho giá trị $u_k$ tương ứng là hướng đa mục tiêu tối ưu $l(\theta_r^*)$. Trong pha thứ hai, họ lặp đi lặp lại sử dụng $d_{des}$ để đạt đến nghiệm Pareto tối ưu $\theta^* \in \mathcal{P}$ gần với $\theta_0^*$ để tìm $l(\theta^*) \in l(\mathcal{P}) \cap l(\Omega_k)$. Tuy nhiên, phương pháp của họ không đảm bảo rằng kết quả pha thứ hai $\theta^*$ cũng nằm trong $\Omega_k$.