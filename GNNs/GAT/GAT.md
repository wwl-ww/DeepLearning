
[TOC]

----

## 1. 概念
回顾GCN的节点更新公式：
$$
\textbf h_i^{(l+1)}=\sigma\left(\sum_{j \in N_i} \frac{1}{c_{i j}} \textbf h_j^{(l)} W^{(l)}+\textbf b^{(l)}\right)
$$

其中：
$$
\frac{1}{c_{i j}}=\frac{1}{\sqrt{\operatorname{deg}\left(v_i\right)} \cdot \sqrt{\operatorname{deg}\left(v_j\right)}}
$$
我们可以发现，在GCN中，每个节点的邻居节点的权重是相同的，即$\frac{1}{c_{i j}}$是一个定值。

我们的目标是想让这个权重是可变的，也就是一个这样的公式：
$$
\textbf h_i^{(l+1)}=\sigma\left(\sum_{j \in N_i} \alpha_{i j} \textbf h_j^{(l)} W^{(l)}+\textbf b^{(l)}\right)
$$
其中$\alpha_{i j}$是一个可变权重，它表示节点$i$与节点$j$的权重。

----

## 2. 公式
我们的目标是通过一个函数$attention$，使得：
$$
\alpha_{i j}=\operatorname{attention}\left(\textbf h_i^{(l)}, \textbf h_j^{(l)}\right)
$$
这里$attention$的实现有很多种，现在介绍最早的一种。
$$e_{ij}=\operatorname{LeakyReLU}(\textbf{a}[\textbf h_iW^{(l)}||\textbf h_jW^{(l)}])$$

$\textbf h_i$: **(1, f)** 第i个节点的特征向量
$W$: **(f, f')** 权重矩阵
$\textbf h_iW||\textbf h_jW$: **(1, 2f')**  "||"表示拼接
$a$: **(2f', 1)** 用于与$\textbf h_iW||\textbf h_jW$相乘得到标量

下一步学过transformer的朋友应该能猜出来了，自然是"按行"softmax
（这里是在所有邻居节点中做softmax）：
$$
\alpha_{ij}=
\operatorname{softmax}(e_{ij})=
\frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_{i}}\exp(e_{ik})}
$$


$$
=\frac{\exp\left(\operatorname{LeakyReLU}\left(\textbf{a}[{\textbf h_{i}W^{(l)}}||{\textbf h_{j}W^{(l)}}]\right)\right)}{\sum_{k\in\mathcal{N}_{i}}\exp\left(\operatorname{LeakyReLU}\left(\textbf{a}[{\textbf h_{i}W^{(l)}}||{\textbf h_{k}W^{(l)}}]\right)\right)}
$$

然后权重更新即可：
$$
\textbf h_i^{(l+1)}=\sigma\left(\sum_{j \in N_i} \alpha_{i j} \textbf h_j^{(l)} W^{(l)}+\textbf b^{(l)}\right)
$$

下面讲多头注意力。
把$W$ **(f, f')** -> **(T, f, f')** ，把$b$ **(f')** -> **(T, f')** 即可。

*for t in range(T):*
$$e_{ij}^t=\operatorname{LeakyReLU}(\textbf{a}^t[\textbf h_iW^{(l)}_t||\textbf h_jW^{(l)}_t])$$

$$
\alpha_{ij}^t=
\operatorname{softmax}(e_{ij}^t)=
\frac{\exp(e_{ij}^t)}{\sum_{k\in\mathcal{N}_{i}}\exp(e_{ik}^t)}
$$

权重更新：
$$
{\textbf h_i}^{(l+1)}=\coprod_{t=1}^T\sigma\left(\sum_{j\in{N}_i}\alpha_{ij}^t{\textbf h_j}W^{(l)}_t+\textbf b^{(l)}_t\right)
$$

$\coprod_{t=1}^T$ 表示全部concat。
这样得到的${\textbf h_i}^{(l+1)}$ 将会是 **(T, 1, f')**
整个特征矩阵从 **(M, f)** -> **(M, Tf')**

除了全部头concat，还可以求平均，原理类似。特征矩阵从 **(M, f)** -> **(M, f')**

----

## 3. 流程图
模型不复杂，略之

----

## 4. 代码

矩阵的公式推导以及pytorch代码实现在这里:[【模型学习之路】手写+分析GAT](https://blog.csdn.net/wwl412095144/article/details/143567049?spm=1001.2014.3001.5501)
这里我们用pyg实现。一般也都是用的PyG。

```python
import torch
from torch_geometric.data import Data, DataLoader

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_softmax

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.2):
        super(GATConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.W = torch.nn.Parameter(torch.Tensor(heads, in_channels, out_channels))
        self.att = torch.nn.Parameter(torch.Tensor(heads, 2 * out_channels, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, x, edge_index):
        """
        x: (M, f)
        edge_index: (2, e_0)        
        """
        # 添加自环 edge_index: (2, e_0) -> (2, e)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 线性变换 (M, f) @ (heads, f, f') -> (heads, M, f')
        x = x @ self.W

        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i, edge_index, size):
        """
        x_i: (heads, e, f') 所有起点节点的特征矩阵
        x_j: (heads, e, f') 所有终点节点的特征矩阵
        """
        # 计算注意力系数
        # 1. concat: (heads, e, 2 * f')
        # 2. (heads, e, 2 * f') @ (heads, 2 * f', 1) -> (heads, e, 1)
        alpha = torch.cat([x_i, x_j], dim=-1) @ self.att
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # 对每个节点的所有邻居节点的注意力系数进行归一化
        row, col = edge_index
        alpha = scatter_softmax(alpha, dim=1, index=col)
        
        # dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # message函数要返回所有的终点特征矩阵(x_j)
        # (heads, e, f') * (heads, e, 1) -> (heads, e, f')
        return alpha.reshape(self.heads, -1, 1) * x_j

    def update(self, aggr_out):
        """
        aggr_out: (heads, M, f')
        """
        if self.concat and self.heads > 1:
            # 做拼接
            # (heads, M, f') -> (M, heads, f') -> (M, heads*f')
            aggr_out = aggr_out.transpose(0, 1)
            aggr_out = aggr_out.contiguous().view(-1, self.heads * self.out_channels)
        else:
            # 按第一个维度求平均
            # (heads, M, f') -> (M, f')
            aggr_out = aggr_out.mean(dim=0)

        return aggr_out

x1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y1 = torch.tensor([0, 1, 0], dtype=torch.long)

x2 = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y2 = torch.tensor([1, 0, 1], dtype=torch.long)

data1 = Data(x=x1, edge_index=edge_index1, y=y1)
data2 = Data(x=x2, edge_index=edge_index2, y=y2)

loader = DataLoader([data1, data2], batch_size=2)

model = GATConv(2, 3, heads=10)
for batch in loader:
    print(batch.x.shape)
    out = model(batch.x, batch.edge_index)
    print(out.shape)

#output:
torch.Size([6, 2])
torch.Size([6, 30])
```
<br>
自然，这个层已经被PyG实现了，可以直接使用。

```python
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader

x1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y1 = torch.tensor([0, 1, 0], dtype=torch.long)

x2 = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y2 = torch.tensor([1, 0, 1], dtype=torch.long)

data1 = Data(x=x1, edge_index=edge_index1, y=y1)
data2 = Data(x=x2, edge_index=edge_index2, y=y2)

loader = DataLoader([data1, data2], batch_size=2)

model = GATConv(2, 3, heads=10)
for batch in loader:
    print(batch.x.shape)
    out = model(batch.x, batch.edge_index)
    print(out.shape)

##output:
torch.Size([6, 2])
torch.Size([6, 30])
```

----

## 5. 参考资料
1. 为什么GAT适用于有向图? [深入浅出GAT–Graph Attention Networks（图注意力模型）](https://blog.csdn.net/xiao_muyu/article/details/121762806)
