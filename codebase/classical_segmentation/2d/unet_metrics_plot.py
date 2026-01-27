import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_losses, val_ious, val_dices, out_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    plt.plot(epochs, val_ious, label='Val IoU', color='tab:green')
    plt.plot(epochs, val_dices, label='Val Dice', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('UNet Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
