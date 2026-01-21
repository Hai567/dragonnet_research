import numpy as np
from sklearn.metrics import auc

def qini_score(y_true, uplift_score, treatment):
    """
    Hàm tính Qini Coefficient (Chỉ số đánh giá hiệu quả Uplift).
    Giá trị càng cao càng tốt.
    """
    # 1. Sắp xếp khách hàng theo điểm Uplift giảm dần (Ưu tiên người tiềm năng nhất)
    order = np.argsort(uplift_score)[::-1]
    y_true = y_true[order]
    treatment = treatment[order]
    
    # 2. Tính tổng tích lũy (Cumulative Sum)
    y_t_cumsum = np.cumsum(y_true * treatment)
    y_c_cumsum = np.cumsum(y_true * (1 - treatment))
    n_t = np.cumsum(treatment)
    n_c = np.cumsum(1 - treatment)
    
    # Tổng số lượng toàn tập
    N_t_total = n_t[-1]
    N_c_total = n_c[-1]
    
    if N_t_total == 0 or N_c_total == 0:
        return 0.0
        
    # 3. Tính đường Qini Curve
    # Công thức: Gain_Treat - (Tổng_Treat / Tổng_Ctrl) * Gain_Ctrl
    curve = y_t_cumsum - y_c_cumsum * (N_t_total / N_c_total)
    
    # Thêm điểm (0,0) vào đầu
    curve = np.concatenate(([0], curve))
    x = np.arange(len(curve))
    
    # 4. Tính diện tích dưới đường cong (Area Under Curve)
    area = auc(x, curve)
    
    # 5. Trừ đi diện tích đường ngẫu nhiên (Random Line) để chuẩn hóa
    random_area = (len(curve) * curve[-1]) / 2
    return area - random_area