import matplotlib.pyplot as plt
import re
import os

def read_metrics(file_path):
    mIoU, mPA, mPrecision, mFallOut, Accuracy = [], [], [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = re.sub(r'^\[.*?\]\s*===>\s*', '', line.strip())
            metrics = re.findall(r'(\w+): ([\d.]+)%', clean_line)
            if metrics:
                metric_dict = {metric: float(value) for metric, value in metrics}
                if {'mIoU', 'mPA', 'mPrecision', 'mFallOut', 'Accuracy'}.issubset(metric_dict):
                    mIoU.append(metric_dict['mIoU'])
                    mPA.append(metric_dict['mPA'])
                    mPrecision.append(metric_dict['mPrecision'])
                    mFallOut.append(metric_dict['mFallOut'])
                    Accuracy.append(metric_dict['Accuracy'])
    return mIoU, mPA, mPrecision, mFallOut, Accuracy


def plot_metric(metric_values, metric_name, save_path='metric_plot.png'):
    epochs = range(1, len(metric_values) + 1)
    plt.figure(figsize=(6, 5))

    plt.plot(epochs, metric_values, 'r-', label='validation')
    plt.plot(epochs, metric_values, 'ro', markersize=3)
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('epoch')
    plt.ylabel(metric_name)

    # === 自動調整 y 軸範圍 ===
    if metric_values:  # 避免空清單
        margin = (max(metric_values) - min(metric_values)) * 0.05  # 留 5% buffer
        y_min = min(metric_values) - margin
        y_max = max(metric_values) + margin
        plt.ylim(y_min, y_max)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# === 主程式 ===
file_path = r"E:\seagrass_training\unet-pytorch-main\logs\U-net_paper_again\performance_log.txt"
mIoU, mPA, mPrecision, mFallOut, Accuracy = read_metrics(file_path)

# 輸出資料夾
output_dir = r"E:\seagrass_training\unet-pytorch-main\logs\U-net_paper_again"
os.makedirs(output_dir, exist_ok=True)

# 繪圖
plot_metric(Accuracy, 'Accuracy', os.path.join(output_dir, 'accuracy.png'))
plot_metric(mIoU, 'mIoU', os.path.join(output_dir, 'miou.png'))
plot_metric(mPA, 'mPA', os.path.join(output_dir, 'mpa.png'))
plot_metric(mPrecision, 'mPrecision', os.path.join(output_dir, 'mprecision.png'))
plot_metric(mFallOut, 'mFallOut', os.path.join(output_dir, 'fallout.png'))
