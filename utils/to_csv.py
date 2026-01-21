import os
import joblib
import pandas as pd
import numpy as np
import glob


BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
INPUT_DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "exported_csvs")

def convert_dict_to_df(data_dict):
    """
    Chuyển đổi dictionary chứa numpy arrays thành Pandas DataFrame
    """
    # 1. Lấy dữ liệu cơ bản: X (Features), y (Outcome), t (Treatment)
    X = data_dict['X']
    y = data_dict['y']
    t = data_dict['t']
    
    # Tạo DataFrame từ X
    # Đặt tên cột là feat_0, feat_1... vì dữ liệu gốc đã drop tên cột
    feature_cols = [f'feat_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    
    # Thêm cột Treatment và Outcome
    df['treatment_group'] = t
    df['outcome'] = y
    
    # 2. Xử lý Propensity Score (nếu có)
    # Biến 'p' thường là ma trận (n_samples, n_treatments)
    if 'p' in data_dict:
        p_scores = data_dict['p']
        if len(p_scores.shape) > 1:
            for i in range(p_scores.shape[1]):
                df[f'propensity_score_t{i}'] = p_scores[:, i]
        else:
            df['propensity_score'] = p_scores

    # 3. Xử lý True Effect (nếu có - chỉ dataset synthetic mới có)
    if 'effect' in data_dict:
        df['true_uplift_effect'] = data_dict['effect']

    return df

def main():
    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    print(f"Reading data from: {INPUT_DATA_DIR}")
    print(f"Saving CSVs to:   {OUTPUT_CSV_DIR}\n")

    # Tìm tất cả các thư mục con (ví dụ: synth1_0, hillstrom_1...)
    subdirs = [d for d in os.listdir(INPUT_DATA_DIR) if os.path.isdir(os.path.join(INPUT_DATA_DIR, d))]

    if not subdirs:
        print("CẢNH BÁO: Không tìm thấy thư mục dữ liệu nào. Hãy chắc chắn bạn đã chạy get_data.py")
        return

    for subdir in subdirs:
        subdir_path = os.path.join(INPUT_DATA_DIR, subdir)
        
        # Tìm các file .pkl trong thư mục con (thường là train.pkl và test.pkl)
        pkl_files = glob.glob(os.path.join(subdir_path, "*.pkl"))
        
        for pkl_file in pkl_files:
            file_name = os.path.basename(pkl_file) # ví dụ: train.pkl
            file_base = os.path.splitext(file_name)[0] # ví dụ: train
            
            # Tạo tên file output: ví dụ synth1_0_train.csv
            output_filename = f"{subdir}_{file_base}.csv"
            output_path = os.path.join(OUTPUT_CSV_DIR, output_filename)
            
            try:
                print(f"Processing: {subdir}/{file_name}...", end=" ")
                
                # Load data
                data = joblib.load(pkl_file)
                
                # Convert
                df = convert_dict_to_df(data)
                
                # Save
                df.to_csv(output_path, index=False)
                print(f"Done! -> {output_filename} ({df.shape})")
                
            except Exception as e:
                print(f"\nLỖI khi xử lý {pkl_file}: {e}")

    print("\nHoàn tất chuyển đổi dữ liệu.")

if __name__ == "__main__":
    main()