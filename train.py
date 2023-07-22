import pandas as pd
import time
from utils import fix_seed_torch, draw_loss_pic
import argparse
from model import GCN
from Logger import Logger
from mydataset import MyDataset
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import sys

# 固定随机数种子
fix_seed_torch(seed=2021)

# 设置训练的超参数
parser = argparse.ArgumentParser()
parser.add_argument('--gcn_layers', type=int, default=2, help='the number of gcn layers')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--embedSize', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
args = parser.parse_args()

# 设备是否支持cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = device

# 读取用户特征、天气特征、评分
user_feature = pd.read_csv('./data/user.txt', encoding='utf-8', sep='\t')
item_feature = pd.read_csv('./data/weather.txt', encoding='utf-8', sep='\t')
rating = pd.read_csv('./data/rating.txt', encoding='utf-8', sep='\t')

# 构建数据集
dataset = MyDataset(rating)
trainLen = int(args.ratio * len(dataset))
train, test = random_split(dataset, [trainLen, len(dataset) - trainLen])
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test, batch_size=len(test))

# 记录训练的超参数
start_time = '{}'.format(time.strftime("%m-%d-%H-%M", time.localtime()))
logger = Logger('./log/log-{}.txt'.format(start_time))
logger.info(' '.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

# 定义模型
model = GCN(args, user_feature, item_feature, rating)
model.to(device)

# 定义优化器
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

# 定义损失函数
loss_function = MSELoss()
train_result = []
test_result = []

# 最好的epoch
best_loss = sys.float_info.max
# 'sys.float_info.max'是Python语言中的一个浮点数常量，表示机器可表示的最大浮点数。

# 训练
for i in range(args.n_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        # optimizer.zero_grad()是PyTorch中定义优化器的一个方法，它会将模型中所有可训练的参数的梯度清零。
        # 在训练神经网络时，通常需要在每次迭代之前调用这个函数。因为如果不清零梯度，那么优化器在更新权重时会累加之前的梯度。
        prediction = model(batch[0].to(device), batch[1].to(device))
        # 这段代码是用PyTorch框架调用训练好的模型进行推理预测。
        # 其中，model是一个已经训练好的模型，batch是一个包含输入数据的批次，例如输入的文本序列和对应的标签。batch[0]
        # 表示输入的文本序列，batch[1]表示对应的标签。to(device)是将输入数据放到指定的设备上进行计算，例如GPU。最终，prediction是模型的预测输出结果。
        train_loss = torch.sqrt(loss_function(batch[2].float().to(device), prediction))
        # 反向传播，计算梯度
        train_loss.backward()
        # 使用优化器更新模型参数
        optimizer.step()
    # 将训练过程中每个batch的loss值记录在一个列表train_result中。train_loss.item()返回当前batch的loss值，
    # .item()方法将该值从tensor类型转换为Python float类型，以便可以在列表中保存。
    train_result.append(train_loss.item())

    model.eval()
    for data in test_loader:
        prediction = model(data[0].to(device), data[1].to(device))
        test_loss = torch.sqrt(loss_function(data[2].float().to(device), prediction))
        test_loss = test_loss.item()
        if best_loss > test_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './model/bestModeParms-{}.pth'.format(start_time))
        test_result.append(test_loss)
        logger.info("Epoch{:d}:trainLoss{:.4f},testLoss{:.4f}".format(i, train_loss, test_loss))
        # #else:
        # model.load_state_dict(torch.load("./model/bestModeParms-11-18-19-47.pth"))

        # torch.load("path路径")
        # 表示加载已经训练好的模型
        # 而model.load_state_dict（torch.load(PATH)）表示将训练好的模型参数重新加载至网络模型中

#   user_id=input("请输入用户id")
#   item_num=rating['itemId'].max()+1
#   u=torch.tensor([int(user_id)for i in range(item_num)],dtype=float)
# 气ID".format(user_id))
#   print(i[0]for i in result)


best_epoch, RMSE = sorted(list(enumerate(train_result)), key=lambda x: x[1])[0]
logger.info("Epoch {:d}: bestTrainLoss {:.4f}".format(best_epoch, RMSE))
best_epoch, RMSE = sorted(list(enumerate(test_result)), key=lambda x: x[1])[0]
logger.info("Epoch {:d}: bestTestLoss {:.4f}".format(best_epoch, RMSE))

# 画图
draw_loss_pic(train_result, test_result)

# 加载最佳模型
model.load_state_dict(torch.load('./model/bestModeParms-{}.pth'.format(start_time)))
model.eval()

# 推荐Top-K项目
while True:

  user_id = input("请输入用户id:")
  if user_id == 'quit':
     break

  user_tensor = torch.tensor([int(user_id)] * model.num_item).to(device) #用户ID转换为张量
  item_tensor = torch.arange(model.num_item).to(device) #创建一个张量

  predictions = model(user_tensor, item_tensor)
  top_k_items = torch.topk(predictions, k=3)[1]

  print("推荐项目:")
  for item in top_k_items:
     item = item.item()
     print(item_feature.loc[item]['id'])