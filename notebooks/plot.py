import torch
import matplotlib.pyplot as plt
from pathlib import Path
"""
Epoch 1/10
Train loss: 0.2804 | Train acc: 0.8829                                                                                                     
Val loss:   0.2292 | Val acc:   0.9053
Saved best model to: /Users/xinyingjia/Downloads/HISTOIMAG/outputs/checkpoints/resnet18_best.pt

Epoch 2/10
Train loss: 0.2411 | Train acc: 0.9008                                                                                                     
Val loss:   0.2302 | Val acc:   0.9057

Epoch 3/10
Train loss: 0.2250 | Train acc: 0.9082                                                                                                     
Val loss:   0.2360 | Val acc:   0.9027

Epoch 4/10
Train loss: 0.2144 | Train acc: 0.9128                                                                                                     
Val loss:   0.2422 | Val acc:   0.8997

Epoch 5/10
Train loss: 0.2064 | Train acc: 0.9163                                                                                                     
Val loss:   0.2377 | Val acc:   0.9042

Epoch 6/10
Train loss: 0.1983 | Train acc: 0.9190                                                                                                     
Val loss:   0.2322 | Val acc:   0.9052

Epoch 7/10
Train loss: 0.1926 | Train acc: 0.9228                                                                                                     
Val loss:   0.2343 | Val acc:   0.9065

Epoch 8/10
Train loss: 0.1865 | Train acc: 0.9244                                                                                                     
Val loss:   0.2364 | Val acc:   0.9052

Epoch 9/10
Train loss: 0.1799 | Train acc: 0.9277                                                                                                     
Val loss:   0.2306 | Val acc:   0.9055

Epoch 10/10
Train loss: 0.1750 | Train acc: 0.9291                                                                                                     
Val loss:   0.2500 | Val acc:   0.8996

Training finished.
Best val loss: 0.2292
"""
train_loss = [0.2804, 0.2411 ,0.2250, 0.2144, 0.2064, 0.1983, 0.1926 ,0.1865 , 0.1799, 0.1750]
val_loss = [0.2292, 0.2302, 0.2360, 0.2422, 0.2377 ,0.2322, 0.2343, 0.2364, 0.2306, 0.2500]
train_acc = [0.8829, 0.9008, 0.9082, 0.9128, 0.9163 ,0.9190, 0.9228,0.9244, 0.9277, 0.9291 ]
val_acc = [0.9053, 0.9057, 0.9027, 0.8997, 0.9042, 0.9052, 0.9065, 0.9052, 0.9055, 0.8996]

epochs = range(1, 11)

    # Loss curve
plt.figure()
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()

loss_path = "./outputs/figures/loss_curve.png"
plt.savefig(loss_path, dpi=300)


plt.show()


    # Accuracy curve
plt.figure()
plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

acc_path = "./outputs/figures/accuracy_curve.png"
plt.savefig(acc_path, dpi=300)


plt.show()


print(f"Saved loss curve to: {loss_path}")
print(f"Saved accuracy curve to: {acc_path}")