import numpy as np

# 生成所有整数（包含0和488700）
total_numbers = np.arange(0, 488700)  # 共488701个数字

# 随机打乱顺序
np.random.shuffle(total_numbers)

# 分割数据集
train_data = total_numbers[:10000]     # 前10000个作为训练集
test_data = total_numbers[10000:]      # 剩余作为测试集

# 保存为.npy文件
np.save("train.npy", train_data)
np.save("test.npy", test_data)

# 验证结果
print(f"训练集大小: {len(train_data)}, 验证是否包含重复值: {len(np.unique(train_data)) == 10000}")
print(f"测试集大小: {len(test_data)}, 验证总数量: {len(train_data) + len(test_data) == 488701}")