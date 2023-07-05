import re
import matplotlib.pyplot as plt

# 初始化列表来保存每个loss的值
loss_disc = []
loss_gen = []
loss_fm = []
loss_mel = []
loss_kl = []

# 从日志数据中提取每个epoch的loss值
with open('./logs/ikaros/train.log', 'r') as file:
    for line in file:
        match = re.search(r'loss_disc=(\d+\.\d+), loss_gen=(\d+\.\d+), loss_fm=(\d+\.\d+),loss_mel=(\d+\.\d+), loss_kl=(\d+\.\d+)', line)
        if match:
            loss_disc.append(float(match.group(1)))
            loss_gen.append(float(match.group(2)))
            loss_fm.append(float(match.group(3)))
            loss_mel.append(float(match.group(4)))
            loss_kl.append(float(match.group(5)))

# 生成曲线图
epochs = range(1, len(loss_disc) + 1)

plt.plot(epochs, loss_disc, label='Loss Disc')
plt.plot(epochs, loss_gen, label='Loss Gen')
plt.plot(epochs, loss_fm, label='Loss FM')
plt.plot(epochs, loss_mel, label='Loss Mel')
plt.plot(epochs, loss_kl, label='Loss KL')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
