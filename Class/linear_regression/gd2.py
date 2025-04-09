import numpy as np
import matplotlib.pyplot as plt

# 数据集
x = np.array([1, 2, 3], dtype=np.float64)
y = np.array([1, 2, 3], dtype=np.float64)

# 初始化参数
w = 0.0
b = 0.0
learning_rate = 0.01
num_epochs = 3000

# 记录损失值
losses = []
w_values = []
b_values = []

# 梯度下降
for epoch in range(num_epochs):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)

    # 计算梯度
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))

    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db

    # 记录 w, b, loss
    losses.append(loss)
    w_values.append(w)
    b_values.append(b)

    # 每 100 次迭代打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}, w = {w:.4f}, b = {b:.4f}")

print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

# 绘制损失随迭代次数的变化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence")
plt.legend()

# 绘制数据点和拟合直线的变化
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data points")

x_range = np.linspace(min(x), max(x), 100)
y_pred_line = w * x_range + b
plt.plot(x_range, y_pred_line, color='red', label="Fitted line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitted Line after Training")
plt.legend()

plt.show()
