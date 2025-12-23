import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(train_losses, valid_losses=None, title='Learning Curve', figsize=(12, 6)):
    """
    Hàm vẽ biểu đồ Learning Curve (Loss theo Epoch).

    Parameters:
    ----------
    train_losses : list or np.array
        Danh sách giá trị loss của tập huấn luyện.
    valid_losses : list or np.array, optional
        Danh sách giá trị loss của tập validation (nếu có).
    title : str
        Tiêu đề của biểu đồ.
    figsize : tuple
        Kích thước biểu đồ (width, height).
    """
    # Sử dụng style seaborn
    sns.set(style="whitegrid")

    plt.figure(figsize=figsize)

    # Vẽ Train Loss
    # Dùng range để tạo trục x tương ứng với số epoch
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='-', linewidth=2)

    # Vẽ Validation Loss (nếu được truyền vào)
    if valid_losses is not None and len(valid_losses) > 0:
        # Giả định valid được tính mỗi epoch, nếu tần suất khác nhau cần chỉnh lại trục x của valid
        val_epochs = range(1, len(valid_losses) + 1)
        plt.plot(val_epochs, valid_losses, label='Validation Loss', color='orange', linestyle='--', linewidth=2)

    # Trang trí
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()

    plt.show()