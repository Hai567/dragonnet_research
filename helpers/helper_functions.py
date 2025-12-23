import random, os, torch
import numpy as np  


def seed_everything(seed=42):
    random.seed(seed)

    # 2. Hệ điều hành (nếu có dùng hash)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. NumPy (quan trọng cho các hàm của pandas/sklearn dùng np.random)
    np.random.seed(seed)

    # 4. PyTorch (quan trọng cho Dragonnet)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Nếu dùng nhiều GPU

    # Đảm bảo thuật toán chạy giống hệt nhau (hy sinh chút tốc độ)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"--> Đã set Global Seed: {seed}")
    

def print_response_stats(y_data, name="Dataset"):
    total = len(y_data)
    positive = np.sum(y_data)
    rate = positive / total
    
    print(f"--- {name} ---")
    print(f"Total Samples: {total}")
    print(f"Positive (Visits): {int(positive)}")
    print(f"Response Rate: {rate:.2%} (Tỷ lệ 1:{1/rate:.1f})")
    print("-" * 30)