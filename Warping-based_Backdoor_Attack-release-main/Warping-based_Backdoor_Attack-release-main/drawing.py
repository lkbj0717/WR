import matplotlib.pyplot as plt
import pickle

# 打开eval_result.pkl文件并加载内容
with open('.\\checkpoints\\cifar10\\eval_result.pkl', 'rb') as f:
    eval_data = pickle.load(f)

# 创建三个空列表用于存储 clean_acc, bd_acc, cross_acc
clean_acc = []
bd_acc = []
cross_acc = []

# 遍历所有的评估结果并将它们添加到相应的列表中
for item in eval_data:
    clean_acc.append(item['clean_acc'].item())
    bd_acc.append(item['bd_acc'].item())
    cross_acc.append(item['cross_acc'].item())

# 创建一个用于绘图的x轴，长度与数据点一致
epochs = list(range(1, len(clean_acc) + 1))

# 绘制图形，只显示线条，颜色为稍微深一点的红、黄、蓝
plt.plot(epochs, clean_acc, label='Clean Accuracy', marker='', linestyle='-', color='dodgerblue', alpha=0.9)
plt.plot(epochs, bd_acc, label='Backdoor Accuracy', marker='', linestyle='-', color='red', alpha=0.9)
plt.plot(epochs, cross_acc, label='Cross Accuracy', marker='', linestyle='-', color='gold', alpha=0.9)

# 添加标题和标签
plt.title('WaNet model training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

# 添加图例
plt.legend()

# 显示图形
plt.show()
