# 已知方差、均值，分布符合正太分布，给定一个值，求p值
from scipy.stats import norm
import math
def p_value(mean, std, value):
    z = (value - mean) / std
    return norm.cdf(z)

# 测试
mean = 78.5
std = 1.0
value = 76.7
print(p_value(mean, std, value))