import torch
import torch.nn as nn
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 1*1*9
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )  # 1*64*7  N*C*L

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )  # 1*128*5
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 512, 3, 1, 0),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )  # 1*512*3
        self.fc = nn.Linear(3*512, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y = y.reshape(y.size(0), -1)
        y = self.fc(y)
        # print(y.shape)  # torch.Size([1, 1])

        return y


if __name__ == '__main__':
    train_data = torch.load("./train.data")
    # print(train_data)
    test_data = torch.load("./test.data")

    net = Net()
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    for epoch in range(10):
        for i in range(len(train_data) - 9):  # i范围为0-150
            x = train_data[i: i+9]  # 拿到前面的九个数据
            y = train_data[i+9: i+10]  # 拿到第十个数据
            # print(x)
            # print(y)
            x = x.reshape(-1, 1, 9)  # torch.Size([1, 1, 9])
            y = y.reshape(-1, 1)  # torch.Size([1, 1])
            # print(x)
            # print(y)
            out = net(x)  # # torch.Size([1, 1])
            loss = loss_fn(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print("epoch:{}, i:{}, loss:{}".format(epoch, i, loss.item()))

        torch.save(net.state_dict(), "./params.pth")

    label = []
    output = []
    count = []
    plt.ion()
    for i in range(len(test_data) - 9):  # 循环151次
        x = test_data[i: i+9]
        y = test_data[i+9: i+10]
        x = x.reshape(-1, 1, 9)  # torch.Size([1, 1, 9])
        y = y.reshape(-1, 1)  # torch.Size([1, 1])
        # print(test_data)

        out = net(x)
        loss = loss_fn(out, y)

        # print(y)  # tensor([[176.]])
        # print(y.numpy())  # [[176.]]
        # print(y.numpy().reshape(-1))  # [176.]

        # print(out)  # tensor([[174.9553]], grad_fn=<AddmmBackward>)
        # print(out.data)  # tensor([[174.9553]])
        # print(out.data.numpy())  # [[174.95534]]
        # print(out.data.numpy().reshape(-1))  # [174.95534]

        label.append(y.numpy().reshape(-1))
        output.append(out.data.numpy().reshape(-1))
        count.append(i)
        # print(label)  # [array([176.], dtype=float32)]
        # print(output)  # [array([174.95534], dtype=float32)]
        # print(count)  # [0]

        plt.clf()
        label_icon, = plt.plot(count, label, color="red")  # 返回标签图标（元祖）
        out_icon,  = plt.plot(count, output, color="blue")  # 返回输出图标（元祖）
        plt.legend([label_icon, out_icon], ["label", "output"], loc="upper right", fontsize=10)
        plt.pause(0.1)

    plt.savefig("./img.pdf")
    plt.ioff()
    # plt.show()
    r2 = r2_score(label, output)
    var = explained_variance_score(label, output)
    print(r2)
    print(var)








