try:
    import cupy as cp
except ImportError:
    import numpy as cp
    print("WARNING: cupy not installed. Using numpy instead. GPU features will not work.")

import numpy as np

# Giữ nguyên các import từ thư viện gốc Py-Boost
try:
    from py_boost.gpu.boosting import GradientBoosting, GradientBoostingClassifier
    from py_boost.gpu.losses import Loss, BCELoss, BCEMetric, MSELoss
    from py_boost.multioutput.target_splitter import SingleSplitter
    from py_boost.multioutput.sketching import RandomSamplingSketch
    from py_boost.callbacks.callback import Callback
except ImportError:
    # Dummy classes (giữ nguyên để code chạy được nếu thiếu lib)
    print("WARNING: py_boost not installed. Using dummy classes.")
    class GradientBoostingClassifier:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(X.shape[0])
    class Loss: pass
    class BCELoss:
        def base_score(self, y): return cp.zeros(1)
        def get_grad_hess(self, y, p): return cp.zeros_like(p), cp.zeros_like(p)
        def postprocess_output(self, p): return p
    class MSELoss:
        def get_grad_hess(self, y, p): return cp.zeros_like(p), cp.zeros_like(p)
    class SingleSplitter: 
        def before_iteration(self, info): pass
    class RandomSamplingSketch: 
        def __init__(self, *args, **kwargs): pass
        def before_iteration(self, info): pass
    class Callback: pass
    class BCEMetric: pass


# --- CLASS 1: XỬ LÝ MASKING CHO BÀI TOÁN PHÂN LOẠI (STAGE 1) ---
class MaskedBCELoss(BCELoss):
    """
    Hàm Loss BCE có tích hợp kỹ thuật 'Masking'
    Dùng cho Stage 1 (Outcome Model) khi target là binary
    """
    def __init__(self, return_diff_as_uplift=False):
        self.return_diff_as_uplift = return_diff_as_uplift
        self.clip_value = 1e-6

    def compute_initial_log_odds(self, y_true):
        """
        Tính điểm khởi tạo (Initial Log-odds)
        Bỏ qua các giá trị NaN (Masked) để tính trung bình chuẩn.
        """
        # Tính trung bình (xác suất p) bỏ qua NaN
        means = cp.nanmean(y_true, axis=0)
        means = cp.where(cp.isnan(means), 0, means)
        means = cp.clip(means, self.clip_value, 1 - self.clip_value)
        
        # Chuyển đổi xác suất p sang Log-odds (logit): ln(p / (1-p))
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        """
        Tính Gradient và Hessian với kỹ thuật Masking Gradient.
        """
        # Tạo mask: 1 nếu có dữ liệu, 0 nếu là NaN (Counterfactual không quan sát được)
        mask_valid_data = cp.isnan(y_true)
        
        # Bước 1: Giả vờ tính gradient cho tất cả (điền 0 vào chỗ NaN để ko lỗi)
        grad, hess = super().get_grad_hess(cp.where(mask_valid_data, 0, y_true), y_pred)
        
        # Bước 2: Áp dụng Masking - Gán Gradient = 0 ở những chỗ NaN
        mask_multiplier = (~mask_valid_data).astype(cp.float32)
        grad = grad * mask_multiplier
        hess = hess * mask_multiplier

        return grad, hess

    def postprocess_output(self, y_pred):
        y_pred = super().postprocess_output(y_pred)
        if self.return_diff_as_uplift:
            # Uplift = Treatment - Control
            # Assuming y_pred cols are [Control, Treatment]
            uplift_score = y_pred[:, 1:] - y_pred[:, :1]
            return uplift_score
        return y_pred


# --- CLASS 2: XỬ LÝ MASKING CHO BÀI TOÁN HỒI QUY (STAGE 2 & REGRESSION) ---
class MaskedMSELoss(MSELoss):
    """
    Hàm Loss MSE có tích hợp kỹ thuật 'Masking'.
    Dùng cho Stage 2 (Uplift Model) vì Uplift là số thực (Regression)
    """
    def get_grad_hess(self, y_true, y_pred):
        # Logic Masking tương tự như trên
        mask_valid_data = cp.isnan(y_true)
        
        grad, hess = super().get_grad_hess(cp.where(mask_valid_data, 0, y_true), y_pred)
        
        mask_multiplier = (~mask_valid_data).astype(cp.float32)
        grad = grad * mask_multiplier
        hess = hess * mask_multiplier

        return grad, hess


# --- CLASS 3: BỘ NÃO ĐIỀU PHỐI (ALGORITHM 1) ---
class TwoStageUpliftLoss(Loss, Callback):
    """
    Hàm Loss tổng hợp thực hiện thuật toán 'Two-Stage Gradient Boosting'
    Nó buộc model học 2 nhiệm vụ cùng lúc:
    1. Outcome Prediction (Stage 1)
    2. Uplift Prediction (Stage 2) thông qua 'Surrogate Target'
    """
    def __init__(self, outcome_loss_fn, uplift_start_iter=10, ensemble_weight=0.5):
        """
        Args:
            outcome_loss_fn: Giáo viên dạy Stage 1 (thường là MaskedBinaryCrossEntropyLoss)
            uplift_start_iter: Vòng lặp bắt đầu kích hoạt Stage 2 (để Stage 1 ổn định trước)
            ensemble_weight: Trọng số w để kết hợp kết quả cuối cùng
        """
        self.outcome_loss_fn = outcome_loss_fn
        # Stage 2 luôn dùng MSE vì nó học để khớp với Surrogate Target (số thực)
        self.uplift_loss_fn = MaskedMSELoss()
        
        self.uplift_start_iter = uplift_start_iter
        self.current_iter = 0
        self.ensemble_weight = ensemble_weight

    def before_iteration(self, build_info):
        # Cập nhật số vòng lặp hiện tại để biết khi nào kích hoạt Stage 2
        self.current_iter = build_info['num_iter']

    def compute_initial_log_odds(self, y_true):
        # 1. Tính điểm khởi tạo cho Outcome (Stage 1)
        init_outcome = self.outcome_loss_fn.compute_initial_log_odds(y_true)
        
        # 2. Tính điểm khởi tạo cho Uplift (Stage 2) dựa trên trung bình
        avg_uplift = cp.nanmean(y_true, axis=0)
        init_uplift = avg_uplift[1:] - avg_uplift[:1] # Treat - Control

        # Ghép lại thành vector khởi tạo [Control, Treatment, Uplift]
        return cp.concatenate([init_outcome, init_uplift])

    def get_grad_hess(self, y_true, y_pred):
        # y_pred thường có shape [N, 3]: [Pred_Control, Pred_Treatment, Pred_DirectUplift]
        n_outcome_cols = y_true.shape[1] # Thường là 2 (Control, Treatment)

        # --- STAGE 1: TÍNH GRADIENT CHO OUTCOME MODEL ---
        # Chỉ dùng các cột đầu tiên (Control, Treatment)
        grad_outcome, hess_outcome = self.outcome_loss_fn.get_grad_hess(
            y_true, y_pred[:, :n_outcome_cols]
        )

        # --- STAGE 2: TÍNH GRADIENT CHO UPLIFT MODEL (SURROGATE TARGET) ---
        # 2a. Tạo Surrogate Target (Nhãn Uplift Giả Lập)
        # Lấy dự đoán Outcome hiện tại
        outcome_preds = self.outcome_loss_fn.postprocess_output(y_pred[:, :n_outcome_cols])
        # Tính hiệu số: Surrogate Target = Pred_Treat - Pred_Ctrl
        surrogate_target = outcome_preds[:, 1:] - outcome_preds[:, :1]
        
        # 2b. Tính Gradient dựa trên Surrogate Target này
        if self.current_iter >= self.uplift_start_iter:
            # Nếu đã đến lúc, tính lỗi giữa 'Dự đoán Uplift Trực Tiếp' và 'Surrogate Target'
            grad_uplift, _ = self.uplift_loss_fn.get_grad_hess(
                surrogate_target, y_pred[:, n_outcome_cols:]
            )
        else:
            # Nếu chưa đến lúc, cho Gradient = 0 (Model Uplift ngủ)
            grad_uplift = cp.zeros(
                (grad_outcome.shape[0], grad_outcome.shape[1] - 1), dtype=cp.float32
            )

        hess_uplift = cp.ones_like(grad_uplift)

        # Ghép Gradient của cả 2 Stage lại để trả về cho thư viện
        total_grad = cp.concatenate([grad_outcome, grad_uplift], axis=1)
        total_hess = cp.concatenate([hess_outcome, hess_uplift], axis=1)

        return total_grad, total_hess

    def finalize_and_combine_predictions(self, y_pred): # Đổi tên postprocess_output
        n_outcome_cols = y_pred.shape[1] // 2 + 1
        
        # Lấy dự đoán từ Stage 1 (Outcome)
        outcome_preds = self.outcome_loss_fn.postprocess_output(y_pred[:, :n_outcome_cols])
        uplift_via_outcome = outcome_preds[:, 1:] - outcome_preds[:, :1]
        
        # Lấy dự đoán từ Stage 2 (Direct Uplift)
        uplift_via_direct_model = y_pred[:, n_outcome_cols:]

        # KẾT HỢP (ENSEMBLE) [cite: 257, 296]
        # Prediction = w * Direct_Uplift + (1-w) * (Treat - Ctrl)
        final_prediction = (uplift_via_direct_model * self.ensemble_weight + 
                            uplift_via_outcome * (1 - self.ensemble_weight))
        
        return final_prediction


# --- CLASS 4: QUẢN LÝ VIỆC CHIA NHÁNH CÂY ---
class MultiTaskTargetSplitter(SingleSplitter):
    """
    Bộ tách mục tiêu đa nhiệm[cite: 187].
    Đảm bảo cây quyết định biết cách xử lý output có cấu trúc [Control, Treat, Uplift].
    """
    def before_iteration(self, build_info):
        if build_info['num_iter'] == 0:
            # Xác định số lượng cột outcome
            n_outcome_cols = build_info['data']['train']['grad'].shape[1] // 2 + 1
            
            # Chỉ định nhóm output cho Py-Boost:
            # Nhóm 1: [Control, Treatment] (Học chung)
            # Nhóm 2: [Uplift] (Học riêng)
            self.indexer = [
                cp.arange(n_outcome_cols, dtype=cp.uint64), 
            ] + [
                cp.asarray([x], dtype=cp.uint64) for x in range(n_outcome_cols, n_outcome_cols * 2 - 1)
            ]

    def __call__(self):
        return self.indexer


class StochasticFeatureSketch(RandomSamplingSketch):
    """
    Lớp bổ trợ để lấy mẫu ngẫu nhiên (Stochastic) giúp giảm Overfitting.
    """
    def before_iteration(self, build_info):
        super().before_iteration(build_info)
        self.num_iter = build_info['num_iter']

    def __call__(self, grad, hess):
        # 20% cơ hội dùng lại gradient cũ (Skip sketch) - Kỹ thuật tối ưu tốc độ
        if np.random.rand() > .8:
            return grad, hess
        return super().__call__(grad, hess)


# --- CLASS 5: MODEL CHÍNH (WRAPPER) ---
class TwoStageGradientBoostingUpliftClassifier(GradientBoostingClassifier):
    """
    Class chính để sử dụng. Đóng gói thuật toán Two-Stage Uplift Modeling.
    """
    def __init__(self, 
                 learning_rate=0.05, 
                 max_depth=6, 
                 n_estimators=100, 
                 uplift_ensemble_weight=0.5,
                 **kwargs):
        
        # Cấu hình hàm Loss "2 trong 1"
        self.dual_loss = TwoStageUpliftLoss(
            MaskedBCELoss(), 
            uplift_start_iter=10, 
            ensemble_weight=uplift_ensemble_weight
        )
        
        # Cấu hình bộ tách đa nhiệm
        self.multi_task_splitter = MultiTaskTargetSplitter()
        
        super().__init__(
            loss=self.dual_loss,
            metric=None, 
            target_splitter=self.multi_task_splitter,
            multioutput_sketch=StochasticFeatureSketch(1, smooth=1),
            learning_rate=learning_rate,
            max_depth=max_depth,
            ntrees=n_estimators,
            callbacks=[self.dual_loss], # Đăng ký callback để cập nhật số vòng lặp
            **kwargs
        )
        
    def fit(self, X, y, treatment_indicator): # Đổi tên treatment -> treatment_indicator
        """
        Train model với dữ liệu đầu vào.
        X: Features
        y: Outcome (0/1)
        treatment_indicator: Nhãn nhóm (0=Control, 1=Treatment)
        """
        # Chuẩn bị dữ liệu Input cho hàm Loss Masking (Tạo ma trận có NaN)
        # [N, 2] -> Cột 0: Control, Cột 1: Treatment
        
        y_formatted = np.full((y.shape[0], 2), np.nan, dtype=np.float32)
        
        # Điền dữ liệu vào đúng chỗ, chỗ còn lại để NaN
        mask_control = (treatment_indicator == 0)
        y_formatted[mask_control, 0] = y[mask_control]
        
        mask_treatment = (treatment_indicator == 1)
        y_formatted[mask_treatment, 1] = y[mask_treatment]
        
        # Gọi hàm fit của thư viện mẹ
        return super().fit(X, y_formatted)

    def predict(self, X):
        """
        Dự đoán Uplift Score cuối cùng.
        """
        # Hàm này sẽ tự động gọi 'finalize_and_combine_predictions' trong hàm Loss
        return super().predict(X)