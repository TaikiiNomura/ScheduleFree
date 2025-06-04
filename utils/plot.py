import matplotlib.pyplot as plt

def plot_loss_accuracy(
        lr,
        num_epochs,
        batch_size,
        all_train_losses,
        all_train_accuracies,
        all_test_losses,
        all_test_accuracies,
        optimizer_info):
    print(f'lr = {lr}')
    print(f'batch_size = {batch_size}')
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    for name, info in optimizer_info.items():
        plt.plot(range(1, num_epochs + 1), all_train_losses[name][0], label=info['label'], color=info['color'])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(f'batch_size={batch_size}')
    plt.legend()
    plt.grid(True)

    # ---- 損失（テスト）----
    plt.subplot(2, 2, 2)
    for name, info in optimizer_info.items():
        plt.plot(range(1, num_epochs + 1), all_test_losses[name][0], label=info['label'], color=info['color'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'batch_size={batch_size}')
    plt.legend()
    plt.grid(True)

    # ---- 精度（訓練）----
    plt.subplot(2, 2, 3)
    for name, info in optimizer_info.items():
        plt.plot(range(1, num_epochs + 1), all_train_accuracies[name][0], label=info['label'], color=info['color'])
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy (%)')
    plt.title(f'batch_size={batch_size}')
    plt.legend()
    plt.grid(True)

    # ---- 精度（テスト）----
    plt.subplot(2, 2, 4)
    for name, info in optimizer_info.items():
        plt.plot(range(1, num_epochs + 1), all_test_accuracies[name][0], label=info['label'], color=info['color'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'batch_size={batch_size}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()