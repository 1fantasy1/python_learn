import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体（Windows常用）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量定义
MODEL_PATH = './pytorch.pth'

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])


# 神经网络定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 系统功能函数
def show_menu():
    """显示主菜单"""
    print("\n~~~~ 手写数字识别 ~~~~")
    print("1. 重新训练模型")
    print("2. 使用现有模型预测")
    print("3. 验证随机测试样本")
    print("4. 退出")
    return input("请输入选项(1-4): ")


def train_model(model):
    """模型训练函数"""
    trainloader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}: 损失 {running_loss / 100:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), MODEL_PATH)
    print("\n训练完成! 模型已保存")


def test_model(model):
    """模型测试函数"""
    testloader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=64, shuffle=False
    )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'测试准确率: {100 * correct / total:.2f}%')


def predict_image(model):
    """单张图片预测"""
    image_path = input("请输入图片路径: ")
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1)

        plt.imshow(image.cpu().squeeze(), cmap='gray')
        plt.title(f"预测结果: {pred.item()}")
        plt.show()
    except Exception as e:
        print(f"错误: {str(e)}")


def show_random_sample(model):
    """显示随机测试样本"""
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    index = random.randint(0, len(test_data) - 1)
    image, label = test_data[index]

    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax().item()

    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"预测: {pred} | 实际: {label}")
    plt.show()


# 主程序
def main():
    # 添加设备提示
    print(f'\n当前使用设备: {device}')

    # 初始化模型
    model = Net().to(device)

    # 添加详细的模型加载提示
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("检测到已存在的训练模型")
            print("✓ 模型加载成功\n")  # 新增成功提示
        except Exception as e:
            print("⚠ 模型加载失败:", str(e))
            print("正在初始化新模型...")
    else:
        print("未检测到训练模型")
        print("正在初始化新模型...")

    while True:
        choice = show_menu()

        if choice == '1':
            train_model(model)
            test_model(model)
        elif choice == '2':
            predict_image(model)
        elif choice == '3':
            show_random_sample(model)
        elif choice == '4':
            print("已退出")
            break
        else:
            print("无效输入，请重新选择")


if __name__ == "__main__":
    main()
