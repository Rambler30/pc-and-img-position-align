from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree

class MyGATConv(MessagePassing):

  def __init__(self,
               in_channels: int | Tuple[int, int],
               out_channels: int,
               heads: int = 1,
               concat: bool = True,
               negative_slope: float = 0.2,
               dropout: float = 0.0,
               add_self_loops: bool = True,
               fill_value: float | Tensor | str = 'mean',
               bias: bool = True,
               **kwargs):
    kwargs.setdefault('aggr', 'add')
    super().__init__(node_dim=0, **kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.heads = heads
    self.concat = concat
    self.negative_slope = negative_slope
    self.dropout = dropout
    self.add_self_loops = add_self_loops
    self.fill_value = fill_value

    # if we are operating in bipartite graphs, we apple separate transformations
    # 'lin_src' and 'lin_dst' to srouce and target nodes
    if isinstance(in_channels, int):
      # non-bipartie graph, we use same transformation on source and tagret nodes
      self.lin_src = Linear(
          in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
      self.lin_dst = self.lin_src
    else:
      # bipartite graph
      self.lin_src = Linear(in_channels[0], heads * out_channels, False, 'glorot')
      self.lin_dst = Linear(in_channels[1], heads * out_channels, False, 'glorot')

    # The learnable parameters to compute attention coefficients
    self.att_src = Parameter(Tensor(1, heads, out_channels))
    self.att_dst = Parameter(Tensor(1, heads, out_channels))

    # bias
    if bias:
      # if concat, the output dim is `heads * out_channels`
      if concat:
        self.bias = Parameter(Tensor(heads * out_channels))
      else:
        self.bias = Parameter(Tensor(out_channels))
    else:
      self.reset_parameters('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    super().reset_parameters()
    self.lin_src.reset_parameters()
    self.lin_dst.reset_parameters()
    glorot(self.att_src)
    glorot(self.att_dst)
    zeros(self.bias)

  def forward(self,
              x: Tensor | OptPairTensor,
              edge_index: Adj,
              size: Size = None,
              return_attention_weights: bool = None):
    H, C = self.heads, self.out_channels

    # First, we project the input node features
    if isinstance(x, Tensor):
      assert x.dim() == 2, "Feature dimension must be 2"
      x_src = x_dst = self.lin_src(x).view(-1, H, C)
    else:
      # transform source and target node features via separate weights
      x_src, x_dst = x
      assert x_src.dim() == 2, "Feature dimension must be 2"
      # self.lin_src(x_src): [N, heads * out_channels]
      x_src = self.lin_src(x_src).view(-1, H, C)
      if x_dst is not None:
        x_dst = self.lin_dst(x_dst).view(-1, H, C)

    x = (x_src, x_dst)

    # Next, we compute node-level attention coefficients, both for source and target nodes
    # x_src: [N,H,C], attr_src: [1,H,C]
    # x_src * attr_src: [N,H,C] * [1,H,C] -> [N,H,C]
    # [N,H,C].sum(dim=-1) -> [N,H]
    alpha_src = (x_src * self.att_src).sum(dim=-1)
    alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(dim=-1)
    alpha = (alpha_src, alpha_dst)

    # add_self_loops
    if self.add_self_loops:
      edge_attr = None
      if isinstance(edge_index, Tensor):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value)
      elif isinstance(edge_index, SparseTensor):
        edge_index = set_diag(edge_index)

    # alpha: [N,H]
    alpha = self.edge_updater(edge_index, alpha=alpha)

    # let's propogate
    # out: [N,H,C]
    out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

    if self.concat:
      # 如果是设置为拼接，那么多头注意力的返回值将会concat在一起
      out = out.view(-1, self.heads * self.out_channels)
    else:
      # 如果不设置为拼接，就取多头注意力的平均值，也就是在H维度上取平均
      out = out.mean(dim=1)
    if self.bias is not None:
      out += self.bias

    if return_attention_weights:
      if isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
          adj = set_sparse_value(edge_index, alpha)
          return out, (adj, alpha)
      else:
        return out, (edge_index, alpha)
    else:
      return out

  def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor,
                  ptr: OptTensor) -> Tensor:
    # edge-level attention coefficients for source and target nodes
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.leaky_relu(alpha, self.negative_slope)
    # index是edge_index中的target node索引, softmax默认根据dim=0进行计算
    alpha = softmax(alpha, index=index, ptr=ptr)
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)
    return alpha

# dataset = Planetoid(root = 'dataset',name = 'Cora')
# data = dataset[0]
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float)  # 节点特征
edge_index = torch.tensor([[0, 3], [1, 4]], dtype=torch.long)  # 边索引
net = MyCustomConv()
h_nodes = net(x, edge_index)
print(h_nodes.shape)
