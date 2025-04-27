import numpy as np

# 定义起始值、结束值和数字个数
start = 1659
end = 166
n = 32

# 生成一个线性递增的序列，范围从0到1
x = np.linspace(0, 1, n)

# 使用指数函数生成递减序列
# 这里使用了一个负指数函数，随着x增大，函数值递减且速度变快
y = start - (start - end) * (x ** 2)

# 打印生成的序列
print([round(num) for num in y])