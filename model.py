import torch.nn as nn
from utils import *
from torch.nn import Module
import scipy.sparse as sp

# GCN_Layer类的说明
# 这个类在PyTorch中定义了一个图卷积网络（GCN）层。
# GCN层接受三个输入：
# 1.图形：表示数据的图形结构。它是一个包含图的邻接矩阵的稀疏矩阵。
# 2.self-loop：表示图的self-loop，确保每个节点至少有一个连接。它也是一个稀疏矩阵。
# 3.特征：表示图中节点的特征矩阵。它是一个密集矩阵，其中每一行表示一个节点的特征向量。
#
# GCN层由两个线性层组成，后面是LeakyReLU激活函数。这两个线性层用于聚合来自图和节点特征的信息。
# 第一线性层（self.W1）通过将图和自循环矩阵与特征矩阵相乘（使用torc.sparse.mm），
#     然后将结果与权重矩阵相乘，来聚合来自图和节点特征的信息。该操作的输出是一个新的特征矩阵，它从图中每个节点的相邻节点捕获信息。
# 第二线性层（self.W2）通过在初始特征矩阵和第一线性层的结果之间执行逐元素乘法（使用torc.mul），
#     然后将结果与另一权重矩阵相乘，来聚合来自节点特征的信息。该操作允许模型捕捉节点特征和图结构之间的非线性交互。
# 最后，将LeakyReLU激活函数应用于两个线性层输出的和。该激活函数允许模型学习输入数据和输出之间的非线性关系。
#
# 总之，GCN层采用图结构、节点特征和自循环矩阵，使用两个线性层聚合来自图和节点特征的信息，并将LeakyReLU激活函数应用于输出。
# GCN层的输出是一个新的特征矩阵，它从图和节点特征中捕获信息，并可用于下游任务，如节点分类或链接预测。

class GCN_Layer(Module):
    def __init__(self, inF, outF):
        super(GCN_Layer, self).__init__()
        self.W1 = torch.nn.Linear(in_features=inF, out_features=outF)
        self.W2 = torch.nn.Linear(in_features=inF, out_features=outF)

    def forward(self, graph, selfLoop, features):
        part1 = self.W1(torch.sparse.mm(graph + selfLoop, features))
        part2 = self.W2(torch.mul(torch.sparse.mm(graph, features), features))
        return nn.LeakyReLU()(part1 + part2)


# GCN_Layer类定义了一个单独的GCN层，该层采用输入特征矩阵、图邻接矩阵（具有自循环），并应用两个线性变换，然后是泄漏的ReLU激活函数。该层返回输出特征矩阵。
# GCN类是主要的模型类，它将用户和项目特征以及评级数据作为输入，并构建GCN模型。它使用nn定义用户和项目特征的嵌入。嵌入，构建图邻接矩阵，并堆叠多个GCN层。
# 最终的用户和项目嵌入被连接并通过线性层以获得最终嵌入，该最终嵌入随后用于预测用户项目评级。


class GCN(Module):
    def __init__(self, args, user_feature, item_feature, rating):
        super(GCN, self).__init__()
        self.args = args
        self.device = args.device
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.rating = rating
        self.num_user = rating['user_id'].max() + 1
        self.num_item = rating['item_id'].max() + 1

        # user embedding
        self.user_id_embedding = nn.Embedding(user_feature['id'].max() + 1, 32)
        self.user_age_embedding = nn.Embedding(user_feature['age'].max() + 1, 4)
        self.user_gender_embedding = nn.Embedding(user_feature['gender'].max() + 1, 2)
        self.user_occupation_embedding = nn.Embedding(user_feature['occupation'].max() + 1, 8)
        self.user_location_embedding = nn.Embedding(user_feature['location'].max() + 1, 18)

        # item embedding
        self.item_id_embedding = nn.Embedding(item_feature['id'].max() + 1, 32)
        self.item_type_embedding = nn.Embedding(item_feature['type'].max() + 1, 8)
        self.item_temperature_embedding = nn.Embedding(item_feature['temperature'].max() + 1, 8)
        self.item_humidity_embedding = nn.Embedding(item_feature['humidity'].max() + 1, 8)
        self.item_windSpeed_embedding = nn.Embedding(item_feature['windSpeed'].max() + 1, 8)

        # 自循环
        self.selfLoop = self.getSelfLoop(self.num_user + self.num_item)

        # 堆叠GCN层
        self.GCN_Layers = torch.nn.ModuleList()
        for _ in range(self.args.gcn_layers):
            self.GCN_Layers.append(GCN_Layer(self.args.embedSize, self.args.embedSize))
        self.graph = self.buildGraph()
        self.transForm = nn.Linear(in_features=self.args.embedSize * (self.args.gcn_layers + 1),
                                   out_features=self.args.embedSize)


    # 自环，是否考虑自身节点信息
    def getSelfLoop(self, num):
        i = torch.LongTensor(
            [[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val).to(self.device)

    # 1.    torch.sparse.FloatTensor类
    # 使用该类，可定义一个COO类型的稀疏矩阵。
    #
    # # sparse tensor
    # i = torch.LongTensor([[0, 1, 1],
    #                       [2, 1, 0]])
    # d = torch.tensor([3, 6, 9], dtype=torch.float)
    # a = torch.sparse.FloatTensor(i, d, torch.Size([2, 3]))
    # print(a)
    #
    # 得到的输出
    # tensor(indices=tensor([[0, 1, 1],
    #                        [2, 1, 0]]),
    #        values=tensor([3., 6., 9.]),
    #        size=(2, 3), nnz=3, layout=torch.sparse_coo)
    #
    # i为非零元素的索引， d为非零元素的值。根据COO表示的规则，可以得出，在（0，2）位置为3，（1，1）为6，（1，0）位置为9。使用to_dense()
    # 方法可以将COO形式的矩阵转换为普通的Tensor形式，在定义时还需要指定原来稀疏矩阵的大小，使用torch.Size([2, 3])
    # 作为参数定义原有稀疏矩阵的大小。该类还有其他的一些操作，比如add()或者sub()等
    # 原文链接：https: // blog.csdn.net / weixin_51122816 / article / details / 117999250

    def buildGraph(self):
        # 构建链接矩阵
        rating = self.rating.values
        graph = sp.coo_matrix(
            (rating[:, 2], (rating[:, 0], rating[:, 1])), shape=(self.num_user, self.num_item)).tocsr()
        graph = sp.bmat([[sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],
                         [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))]])

        # 拉普拉斯变化
        row_sum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
        col_sum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
        # @ 在Python中表示矩阵乘法
        graph = row_sum_sqrt @ graph @ col_sum_sqrt
        graph = graph.tocoo()
        # 使用torch的稀疏张量表示
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
        return graph.to(self.device)

    def getFeature(self):
        # 根据用户特征获取对应的embedding
        user_id = self.user_id_embedding(torch.tensor(self.user_feature['id']).to(self.device))
        age = self.user_age_embedding(torch.tensor(self.user_feature['age']).to(self.device))
        gender = self.user_gender_embedding(torch.tensor(self.user_feature['gender']).to(self.device))
        occupation = self.user_occupation_embedding(torch.tensor(self.user_feature['occupation']).to(self.device))
        location = self.user_location_embedding(torch.tensor(self.user_feature['location']).to(self.device))
        user_emb = torch.cat((user_id, age, gender, occupation, location), dim=1)

        # 根据天气特征获取对应的embedding
        item_id = self.item_id_embedding(torch.tensor(self.item_feature['id']).to(self.device))
        item_type = self.item_type_embedding(torch.tensor(self.item_feature['type']).to(self.device))
        temperature = self.item_temperature_embedding(torch.tensor(self.item_feature['temperature']).to(self.device))
        humidity = self.item_humidity_embedding(torch.tensor(self.item_feature['humidity']).to(self.device))
        windSpeed = self.item_windSpeed_embedding(torch.tensor(self.item_feature['windSpeed']).to(self.device))
        item_emb = torch.cat((item_id, item_type, temperature, humidity, windSpeed), dim=1)

        # 拼接到一起
        concat_emb = torch.cat([user_emb, item_emb], dim=0)
        return concat_emb.to(self.device)

    def forward(self, users, items):
        features = self.getFeature()
        # clone() 返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
        final_emb = features.clone()
        for GCN_Layer in self.GCN_Layers:
            features = GCN_Layer(self.graph, self.selfLoop, features)
            final_emb = torch.cat((final_emb, features.clone()), dim=1)
        user_emb, item_emb = torch.split(final_emb, [self.num_user, self.num_item])
        user_emb = user_emb[users]
        item_emb = item_emb[items]
        user_emb = self.transForm(user_emb)
        item_emb = self.transForm(item_emb)

        prediction = torch.mul(user_emb, item_emb).sum(1)
        return prediction
