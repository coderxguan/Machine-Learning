import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# 生成随机数据,  y = 3X + 4
X = 2* np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
print(X)
print(y)
# 初始化参数
m, b = np.random.randn(), np.random.randn()

# 设置一个学习率和迭代次数
learning_rate = 0.1
epoch = 1000


m_history, b_history = [], []

# 梯度下降循环
for _ in range(epoch):
    y_pred = m * X + b
    error = y_pred - y
    m_grad = (2 / len(X)) * np.sum(error * X)  # 计算 m 的梯度
    b_grad = (2 / len(X)) * np.sum(error)  # 计算 b 的梯度

    # 根据梯度下降公式更新参数
    m -= learning_rate * m_grad
    b -= learning_rate * b_grad

    # 保存参数更新的历史记录
    m_history.append(m)
    b_history.append(b)


# 绘制真实数据点
plt.scatter(X, y, label="真实数据")

# 绘制拟合直线，注意这里用最终的 m 和 b 计算直线
plt.plot(X, m * X + b, color="red", label="拟合直线")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()






