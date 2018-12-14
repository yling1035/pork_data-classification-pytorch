import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy

torch.manual_seed(1)  # 设置随机种子，使得每次生成的随机数是确定的
BATCH_SIZE = 150  # 设置batch size
# 1.读取两类数据
train0 = numpy.load("train0.npy")
x0 = torch.from_numpy(train0)#data
y0 = torch.zeros(119)#label

train1 = numpy.load("train1.npy")
x1 = torch.from_numpy(train1)
y1 = torch.ones(215)
# print  ('训练集0大小为',(x0).shape)
# print  ('训练集1大小为',(x1).shape)

# 合并训练数据集,并转化数据类型为浮点型或整型
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)
# print("合并后的数据集维度：", x.data.size(), y.data.size())

# 当不使用batch size训练数据时，将Tensor放入Variable中
# x,y = Variable(x), Variable(y)
# 绘制训练数据
# plt.scatter( x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy())
# plt.show()
# 当使用batch size训练数据时，首先将tensor转化为Dataset格式
torch_dataset = Data.TensorDataset(x, y)

# 将dataset放入DataLoader中
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,  # 设置batch size
    shuffle=True,  # 打乱数据
    num_workers=2  # 多线程读取数据
)

# 2.前向传播过程
class Net(torch.nn.Module):  # 继承基类Module的属性和方法
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()  # 继承__init__功能
        self.hidden = torch.nn.Linear(input, hidden)  # 隐层的线性输出
        self.out = torch.nn.Linear(hidden, output)  # 输出层线性输出

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

# 训练模型的同时保存网络模型参数
def save():
    # 3.利用自定义的前向传播过程设计网络，设置各层神经元数量
    # net = Net(input=10000, hidden=10, output=2)
    # print("神经网络结构：",net)

    # 3.快速搭建神经网络模型
    net = torch.nn.Sequential(
        torch.nn.Linear(10000, 2000),  # 指定输入层和隐层结点，获得隐层线性输出
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),  # 隐层非线性化
        torch.nn.Linear(2000, 200),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 2)  # 指定隐层和输出层结点，获得输出层线性输出
    )
    # 4.设置优化算法、学习率
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer = torch.optim.SGD( net.parameters(), lr=0.008, momentum=0.8)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.8))

    # 5.设置损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    # plt.ion()  # 打开画布，可视化更新过程
    # 6.迭代训练
    for epoch in range(100):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = net(batch_x)  # 输入训练集，获得当前迭代输出值
            loss = loss_func(out, batch_y)  # 获得当前迭代的损失

            optimizer.zero_grad()  # 清除上次迭代的更新梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            if step % 2 == 0:
                # plt.cla()  # 清空之前画布上的内容
                entire_out = net(x)  # 测试整个训练集
                # 获得当前softmax层最大概率对应的索引值
                pred = torch.max(F.softmax(entire_out), 1)[1]
                # 将二维压缩为一维
                pred_y = pred.data.numpy().squeeze()
                label_y = y.data.numpy()
               # plt.scatter(x.data.numpy()[:, 1], x.data.numpy()[:, 2], c=pred_y, cmap='RdYlGn')
                accuracy = sum(pred_y == label_y) / y.size()
                print("第 %d 个epoch，第 %d 次迭代，准确率为 %.2f" % (epoch + 1, step / 2 + 1, accuracy))
                # 在指定位置添加文本
              #  plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 15, 'color': 'red'})
              # plt.pause(2)  # 图像显示时间

    # 7.保存模型结构和参数
    torch.save(net, 'net.pkl')
    # 7.只保存模型参数
    # torch.save(net.state_dict(), 'net_param.pkl')

    # plt.ioff()  # 关闭画布
    # plt.show()

if __name__ == '__main__':
    save()

