import os
import matplotlib.pyplot as plt
import re

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
    return {
        'mIoU': mIoU,
        'mPA': mPA,
        'mPrecision': mPrecision,
        'mFallOut': mFallOut,
        'Accuracy': Accuracy
    }

def compare_metrics(metrics1, metrics2, labels=('unet_before', 'unet_after'), save_dir='.'):
    metric_names = ['mIoU', 'mPA', 'mPrecision', 'mFallOut', 'Accuracy']
    
    for name in metric_names:
        values1 = metrics1[name]
        values2 = metrics2[name]
        epochs = range(1, min(len(values1), len(values2)) + 1)

        plt.figure(figsize=(6, 5))
        plt.plot(epochs, values1[:len(epochs)], 'r-', label=labels[0])
        plt.plot(epochs, values2[:len(epochs)], 'b-', label=labels[1])
        plt.plot(epochs, values1[:len(epochs)], 'ro', markersize=3)
        plt.plot(epochs, values2[:len(epochs)], 'bo', markersize=3)

        plt.title(f'{name} Comparison')
        plt.xlabel('epoch')
        plt.ylabel(name)
        #plt.ylim(80, 88)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{name.lower()}_compare.png')
        plt.show()

# === 主程式 ===
file1 = r"E:\seagrass_training\unet-pytorch-main\logs\loss_before\performance_log.txt"#放入第一個模型的文件路徑
file2 = r"E:\seagrass_training\unet-pytorch-main\logs\U-net_paper_again\performance_log.txt"#放入第二個模型的文件路徑
output_dir = r"E:\seagrass_training\comparison_charts"#輸出位置

os.makedirs(output_dir, exist_ok=True)

metrics_zh = read_metrics(file1)
metrics_en = read_metrics(file2)

compare_metrics(metrics_zh, metrics_en, labels=('U-net_old', 'U-net_new'), save_dir=output_dir)#可更改圖表顯示名稱
