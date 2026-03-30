# 机器学习实战

基于《机器学习实战》书籍的代码练习，实现常见的机器学习算法。

## 项目结构

```
Machine-Learning/
├── kNN/                    # K近邻算法
│   ├── 1.简单kNN/          # 基础示例
│   ├── 2.海伦约会/         # 案例：约会配对
│   └── 3.数字识别/         # 案例：手写数字识别
├── Decision Tree/          # 决策树
├── Naive Bayes/            # 朴素贝叶斯
│   ├── bayes.py            # 基础实现
│   └── bayes-email.py      # 案例：垃圾邮件分类
└── Class/                  # 回归分析
    ├── linear_regression/  # 线性回归
    └── multiple_regression/# 多元回归
```

## 环境配置

```bash
# 创建虚拟环境
uv venv

# 安装依赖
uv pip install pandas scikit-learn matplotlib jupyterlab
```

## 运行

```bash
# 激活环境
source .venv/bin/activate

# 启动 Jupyter
jupyter lab
```

## 依赖

- Python 3.9+
- numpy
- pandas
- scikit-learn
- matplotlib
- jupyterlab
