import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy

# 1.读取两类数据
test0 = numpy.load("test0a.npy")
x0 = torch.from_numpy(test0)#data
y0 = torch.zeros(20)#label

test1 = numpy.load("test1a.npy")
x1 = torch.from_numpy(test1)
y1 = torch.ones(20)
print  ('测试集0大小为',(x0).shape)
print  ('测试集1大小为',(x1).shape)

# 合并训练数据集,并转化数据类型为浮点型或整型
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)
print("合并后的数据集维度：", x.data.size(), y.data.size())

# 将Tensor放入Variable中
x, y = Variable(x), Variable(y)


# 载入模型和参数
def restore_net():
    net = torch.load('net.pkl')
    # 获得载入模型的预测输出
    pred = net(x)
    # 获得当前softmax层最大概率对应的索引值
    pred = torch.max(F.softmax(pred), 1)[1]
    # 将二维压缩为一维
    pred_y = pred.data.numpy().squeeze()
    label_y = y.data.numpy()
    accuracy = sum(pred_y == label_y) / y.size()
    print("准确率为：",accuracy)
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, cmap='RdYlGn')
# plt.show()

if __name__ == '__main__':
 restore_net()


