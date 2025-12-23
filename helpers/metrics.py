from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import auc


def calculateUpliftScore(y0_pred, y1_pred):
    '''
    Calculate Uplift Score for each cate (từng khách hàng)
    '''
    # Chuyển về Numpy array
    y0_val = y0_pred
    y1_val = y1_pred

    # 2. TÍNH UPLIFT SCORE (CATE)
    # Uplift = Xác suất mua nếu Treat - Xác suất mua nếu Control
    uplift_score = y1_val - y0_val

    return uplift_score.flatten()

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def get_curve_coords(y_true, uplift_score, treatment):
    """Hàm phụ trợ tính toán toạ độ đường cong (Backend logic)"""
    # 1. Sắp xếp dữ liệu theo điểm Uplift giảm dần (nhắm mục tiêu tốt nhất trước)
    desc_indices = np.argsort(uplift_score)[::-1]
    y_true = np.array(y_true)[desc_indices]
    treatment = np.array(treatment)[desc_indices]

    # 2. Tính tổng tích lũy (Cumulative Sum) - Vectorized
    y_t = np.cumsum(y_true * treatment)           # Tổng outcome nhóm Treatment
    y_c = np.cumsum(y_true * (1 - treatment))     # Tổng outcome nhóm Control
    n_t = np.cumsum(treatment)                    # Số lượng mẫu nhóm Treatment
    n_c = np.cumsum(1 - treatment)                # Số lượng mẫu nhóm Control

    # Tránh chia cho 0
    n_t[n_t == 0] = 1
    n_c[n_c == 0] = 1
    n_total = np.arange(1, len(y_true) + 1)

    return y_t, y_c, n_t, n_c, n_total

def _get_perfect_uplift(y_true, treatment):
    """
    Tạo ra một vector uplift score giả định cho mô hình hoàn hảo (Optimal).
    Ưu tiên xếp tất cả các trường hợp (Y=1, T=1) lên đầu.
    """
    # Logic: Gán điểm cực cao cho những người mua hàng khi có tác động (Y=1, T=1)
    # Đây là trần lý thuyết cho Hillstrom (Binary outcome)
    z_ideal = np.zeros_like(y_true)
    mask_optimal = (y_true == 1) & (treatment == 1)
    z_ideal[mask_optimal] = 100 # Score cao nhất
    return z_ideal

def _calculate_area(x, y):
    """Tính diện tích dưới đường cong sử dụng quy tắc hình thang"""
    return auc(x, y)


def auqc_score(y_true, uplift, treatment):
    """
    1. Absolute Area Under Qini Curve (AUQC)
    """
    # Lấy toạ độ từ helper function có sẵn của bạn
    y_t, y_c, n_t, n_c, n_total = get_curve_coords(y_true, uplift, treatment)
    
    # Công thức Qini: Y_t - (Y_c * N_t / N_c)
    curve_values = y_t - y_c * (n_t / n_c)
    
    # Tạo trục X (Fraction of population) và trục Y
    x_axis = np.concatenate(([0], n_total / n_total[-1]))
    y_axis = np.concatenate(([0], curve_values))
    
    return _calculate_area(x_axis, y_axis)

def auuc_score(y_true, uplift, treatment):
    """
    2. Absolute Area Under Uplift Curve (AUUC)
    """
    # Lấy toạ độ
    y_t, y_c, n_t, n_c, n_total = get_curve_coords(y_true, uplift, treatment)
    
    # Công thức Uplift Curve: (Mean_t - Mean_c) * N_total
    # Lưu ý: n_t và n_c đã được xử lý tránh chia cho 0 trong get_curve_coords
    mean_t = y_t / n_t
    mean_c = y_c / n_c
    curve_values = (mean_t - mean_c) * n_total
    
    # Tạo trục X và Y
    x_axis = np.concatenate(([0], n_total / n_total[-1]))
    y_axis = np.concatenate(([0], curve_values))
    
    return _calculate_area(x_axis, y_axis)

def normalized_auqc_score(y_true, uplift, treatment):
    """
    3. Normalized Area Under Qini Curve (Qini Coefficient)
    Công thức: (Area_Model - Area_Random) / (Area_Optimal - Area_Random)
    """
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    
    # --- A. Tính Area Model ---
    area_model = auqc_score(y_true, uplift, treatment)
    
    # --- B. Tính Area Random ---
    # Random là đường chéo từ (0,0) đến điểm cuối của Model
    # Điểm cuối của Qini Curve thực tế
    y_t, y_c, n_t, n_c, _ = get_curve_coords(y_true, uplift, treatment)
    end_value = (y_t[-1] - y_c[-1] * (n_t[-1] / n_c[-1]))
    area_random = 0.5 * end_value # Diện tích tam giác: 0.5 * đáy(1.0) * cao
    
    # --- C. Tính Area Optimal ---
    optimal_uplift = _get_perfect_uplift(y_true, treatment)
    area_optimal = auqc_score(y_true, optimal_uplift, treatment)
    
    # --- D. Tính Normalize ---
    numerator = area_model - area_random
    denominator = area_optimal - area_random
    
    if denominator == 0: return 0.0
    return numerator / denominator

def normalized_auuc_score(y_true, uplift, treatment):
    """
    4. Normalized Area Under Uplift Curve (Normalized AUUC)
    Công thức: (Area_Model - Area_Random) / (Area_Optimal - Area_Random)
    """
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    
    # --- A. Tính Area Model ---
    area_model = auuc_score(y_true, uplift, treatment)
    
    # --- B. Tính Area Random ---
    # Điểm cuối của Uplift Curve
    y_t, y_c, n_t, n_c, n_total = get_curve_coords(y_true, uplift, treatment)
    mean_t_global = y_t[-1] / n_t[-1]
    mean_c_global = y_c[-1] / n_c[-1]
    end_value = (mean_t_global - mean_c_global) * n_total[-1]
    area_random = 0.5 * end_value
    
    # --- C. Tính Area Optimal ---
    optimal_uplift = _get_perfect_uplift(y_true, treatment)
    area_optimal = auuc_score(y_true, optimal_uplift, treatment)
    
    # --- D. Tính Normalize ---
    numerator = area_model - area_random
    denominator = area_optimal - area_random
    
    if denominator == 0: return 0.0
    return numerator / denominator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def auuc(y_true, t_true, uplift_pred, bins=100, plot=True):
    """
    AUUC (Area Under uplift curve)
    
    Parameters:
    -----------
    y_true: spend
    t_true: treatment
    uplift_pred: uplift score predict
    bins: amount of buckets
    ------------
    Return
    -----------
    auuc
    """
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()
    
    data = pd.DataFrame({
        'y': y_true,
        "t": t_true,
        "pred": uplift_pred
    })
    
    #sort
    data = data.sort_values(by="pred", ascending=False).reset_index(drop=True)

    #split into bucket
    try:
        data["bucket"] = pd.qcut(-data['pred'], bins, labels=False, duplicates="drop")
    except:
        data['bucket'] = pd.cut(-data['pred'], bins, labels=False)
        
    # data = data.dropna(subset=['bucket'])
    if len(data) == 0:
        print("⚠️ All buckets are NaN after binning!")
        return 0.0 
    
    #create random baseline
    control_data = data.loc[data['t']==0.0]
    treatment_data = data.loc[data['t']==1.0]
    
    # print (control_data)
    # print (treatment_data)
    mean_control = control_data['y'].mean()
    mean_treatment = treatment_data["y"].mean()
    
    random_control = (np.random.rand(len(control_data)) -0.5)/ 10000 + mean_control
    random_treatment = (np.random.rand(len(treatment_data))-0.5) /10000 + mean_treatment
    
    data.loc[data['t']==0, 'random'] = random_control
    data.loc[data['t']==1, 'random'] = random_treatment
    
    #Calculate cumulative gain
    
    cumulative_gain = []
    cumulative_random =[]
    population =[]
    bucket_ids = sorted(data['bucket'].unique())
    
    for i in bucket_ids:
        cumulative_data = data.loc[data['bucket'] <= i]
        
        control_group = cumulative_data.loc[cumulative_data['t']==0.0]
        treatment_group =  cumulative_data.loc[cumulative_data['t']==1.0]
        
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        n_total = n_control + n_treatment
        
        if n_total ==0:
            continue
        if n_control==0 or n_treatment ==0:
            continue
        mean_y_control = control_group['y'].mean()
        mean_y_treatment = treatment_group['y'].mean()

        #AUUC formular
        uplift_gain = (mean_y_treatment - mean_y_control) * n_total
        
        mean_random_control = control_group['random'].mean()
        mean_random_treatment = treatment_group['random'].mean()
        random_gain = (mean_random_treatment - mean_random_control) *n_total
        
        cumulative_gain.append(uplift_gain)
        cumulative_random.append(random_gain)
        population.append(n_total)
        
    if len(cumulative_gain) == 0:
        print("⚠️ Warning: No valid buckets found. All buckets have empty treatment or control groups.")
        print(f"Treatment distribution: {(t_true == 1).sum()} treated, {(t_true == 0).sum()} control")
        return 0.0

    cumulative_random[-1] = cumulative_gain[-1]
    
    #normalize
    gap0 = cumulative_gain[-1]
    
    norm_factor = abs(gap0) if abs(gap0) > 1e-9 else 1.0
    
    cumulative_gains_norm = [x / norm_factor for x in cumulative_gain]
    cumulative_rand_norm = [x/ norm_factor for x in cumulative_random]
    
    #normalize x axis
    pop_max = max(population)
    pop_fraction = [p/pop_max for p in population]
    
    #add (0,0)
    x_curve = np.append(0, pop_fraction)
    y_curve = np.append(0, cumulative_gains_norm)
    y_rand = np.append(0, cumulative_rand_norm)
    
    #calcute auc using trapezoid rule
    auuc_score = np.trapezoid(y_curve, x_curve)
    auuc_rand = np.trapezoid(y_rand, x_curve)
    
    #visualize
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x_curve, y_curve, marker='o', markersize =4,
                 label = f"AUUC score = {auuc_score:.4f}", color= "darkgreen")
        plt.plot(x_curve, y_rand, marker='s', markersize=4,
                label=f'Random AUUC={auuc_rand:.4f})', 
                color='gray', linestyle='--', alpha=0.7)
        plt.xlabel("Cumulative percentage of people targeted")
        plt.ylabel("Cumulative uplift")
        plt.title("AUUC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    if gap0 < 0: 
        auuc_score, auuc_rand = np.trapezoid([i + 1 for i in np.append(0, cumulative_gains_norm)], np.append(0,(np.array(population))/max(population))),\
                                    np.trapezoid([i + 1 for i in np.append(0, cumulative_rand_norm)], np.append(0,(np.array(population))/max(population)))
    return auuc_score

def auqc(y_true, t_true, uplift_pred, bins=100, plot=True):
    """
    AUQC (Area uplift under qini curve)
    
    Parameters:
    -----------
    y_true: spend
    t_true: treatment
    uplift_pred: uplift score predict
    bins: amount of buckets
    ------------
    Return
    -----------
    auqc
    """
    y_true = np.array(y_true).flatten()
    t_true = np.array(t_true).flatten()
    uplift_pred = np.array(uplift_pred).flatten()

    data = pd.DataFrame({
        'y': y_true,
        "t": t_true,
        "pred": uplift_pred
    })
    #sort
    data = data.sort_values(by="pred", ascending=False).reset_index(drop=True)
    
    #split into bucket
    try:
        data["bucket"] = pd.qcut(-data['pred'], bins, labels=False, duplicates="drop")
    except:
        data['bucket'] = pd.cut(-data['pred'], bins, labels=False)
    
    #create random baseline
    control_data = data.loc[data['t']==0]
    treatment_data = data.loc[data['t']==1]
    
    mean_control = control_data['y'].mean()
    mean_treatment = treatment_data["y"].mean()
    
    random_control = (np.random.rand(len(control_data)) -0.5)/ 10000 + mean_control
    random_treatment = (np.random.rand(len(treatment_data))-0.5) /10000 + mean_treatment
    
    data.loc[data['t']==0, 'random'] = random_control
    data.loc[data['t']==1, 'random'] = random_treatment
    
    #Calculate cumulative gain
    
    cumulative_gain = []
    cumulative_random =[]
    population =[]
    bucket_ids = sorted(data['bucket'].unique())
    
    for bucket_id in bucket_ids:
        cumulative_data = data.loc[data['bucket'] <= bucket_id]
        
        control_group = cumulative_data.loc[cumulative_data['t']==0]
        treatment_group =  cumulative_data.loc[cumulative_data['t']==1]
        
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        n_total = n_control + n_treatment
        
        if n_control==0 or n_total==0:
            continue
        
        #calculate mean outcome
        sum_y_control = control_group['y'].sum()
        sum_y_treatment = treatment_group['y'].sum()
        
        #AUUC formular
        qini_gain = sum_y_treatment - sum_y_control * (n_treatment/n_control)
        
        sum_random_control = control_group['random'].sum()
        sum_random_treatment = treatment_group['random'].sum()
        random_gain = sum_random_treatment - sum_random_control *(n_treatment/n_control)
        
        cumulative_gain.append(qini_gain)
        cumulative_random.append(random_gain)
        population.append(n_total)
        
    if len(cumulative_gain) == 0:
        print("⚠️ No valid buckets computed!")
        return 0.0
    
        #force random baseline to meet model at endpoint
    if len(cumulative_random) >0:
        cumulative_random[-1] = cumulative_gain[-1]
    
    #normalize
    gap0 = cumulative_gain[-1]
    
    norm_factor = abs(gap0) if abs(gap0) > 1e-9 else 1.0
    
    cumulative_gains_norm = [x / norm_factor for x in cumulative_gain]
    cumulative_rand_norm = [x/ norm_factor for x in cumulative_random]
    
    #normalize x axis
    pop_max = max(population)
    pop_fraction = [p/pop_max for p in population]
    
    #add (0,0)
    x_curve = np.append(0, pop_fraction)
    y_curve = np.append(0, cumulative_gains_norm)
    y_rand = np.append(0, cumulative_rand_norm)
    
    #calcute auc using trapezoid rule
    qini_score = np.trapezoid(y_curve, x_curve)
    qini_rand = np.trapezoid(y_rand, x_curve)
    
    #visualize
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x_curve, y_curve, marker='o', markersize =4,
                 label = f"AUQC score = {qini_score:.4f}", color= "navy")
        plt.plot(x_curve, y_rand, marker='s', markersize=4,
                label=f'Random AUQC={qini_rand:.4f})', 
                color='gray', linestyle='--', alpha=0.7)
        plt.xlabel("Cumulative percentage of people targeted")
        plt.ylabel("Cumulative qini")
        plt.title("AUQC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return qini_score

def lift (y_true, t_true, uplift_pred, h=0.3):
    """
    Lift@h 
    Parameters:
    -------------
    y_true: spend
    t_true: treatment (0/1)
    uplift_pred = uplift score
    h
    bins: amount of buckets
    -------------
    Return
    -------------
    Lift
    """
    
    y = np.array(y_true).flatten()
    t = np.array(t_true).flatten()
    pred = np.array(uplift_pred).flatten()
    df = pd.DataFrame({'y': y, 't': t, 'pred': pred})
    df = df.sort_values(by='pred', ascending=False).reset_index(drop=True)
    top_k = int(np.ceil(len(df) * h))
    top_df = df.iloc[:top_k]
    mean_c = top_df.loc[top_df['t']==0, 'y'].mean()
    mean_t = top_df.loc[top_df['t']==1, 'y'].mean()

    if np.isnan(mean_c) or np.isnan(mean_t):
        return np.nan
    return float(mean_t - mean_c)

    """
    KRCC (Kendall rank correlation coefficient)
    y_true: spend (1d)
    t_true: treatment (0/1) (1d)
    uplift_pred: predicted uplift score (1d)
    bins: number of buckets to aggregate
    Return: kendall tau (float)
    """
    y = np.array(y_true).flatten()
    t = np.array(t_true).flatten()
    pred = np.array(uplift_pred).flatten()
    
    df = pd.DataFrame({'y': y, 't': t, 'pred': pred})
    df = df.sort_values(by='pred', ascending=False).reset_index(drop=True)

    try:
        df['bucket'] = pd.qcut(-df['pred'], bins, labels=False, duplicates='drop')
    except Exception:
        df['bucket'] = pd.cut(-df['pred'], bins, labels=False)

    pred_uplift_list = []
    cate_list = []

    bucket_indices = sorted(df['bucket'].dropna().unique())
    for b in bucket_indices:
        db = df[df['bucket'] == b]

        mean_control = db.loc[db['t'] == 0, 'y'].mean()
        mean_treatment = db.loc[db['t'] == 1, 'y'].mean()

        if pd.isna(mean_control) or pd.isna(mean_treatment):
            continue

        cate_val = float(mean_treatment - mean_control)
        pred_val = float(db['pred'].mean())

        cate_list.append(cate_val)
        pred_uplift_list.append(pred_val)

    if len(cate_list) < 2:
        return np.nan  

    tau, p = kendalltau(pred_uplift_list, cate_list)

    return tau