import json
import collections
import matplotlib.pyplot as plt

# 1. 设置支持中文的字体（Windows 系统为微软雅黑）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 2. 读取 JSON 文件
with open(r'D:\\Research\\test\\7b-generate-1.5b-reject.json', 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# 3. 统计 correct_num 出现频次
counts = collections.Counter(item['correct_num'] for item in data_list)

# 4. 计算占比
total_count = sum(counts.values())
distribution = {k: (v / total_count) * 100 for k, v in counts.items()}

# 5. 准备绘图数据
labels = sorted(distribution.keys())           # x轴刻度
percentage_list = [distribution[label] for label in labels]  # y值（百分比）

# 6. 画图
plt.bar(labels, percentage_list)
plt.title('correct_num 的分布占比')
plt.xlabel('correct_num')
plt.ylabel('百分比 (%)')

# 7. 在每个bar上方打印数值
for x, y in zip(labels, percentage_list):
    # 这里将数值保留两位小数并在上方稍作偏移
    plt.text(x, y + 0.1, f"{y:.2f}%", ha='center', va='bottom')

plt.show()
