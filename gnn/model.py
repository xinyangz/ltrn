import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from pytorch_utils import MixedDropout, MixedLinear


class MLP(nn.Module):

  def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
    super().__init__()

    fcs = [MixedLinear(num_features, hidden_size, bias=False)]
    for i in range(nlayers - 2):
      fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
    fcs.append(nn.Linear(hidden_size, num_classes, bias=False))
    self.fcs = nn.ModuleList(fcs)

    self.drop = MixedDropout(dropout)

  def forward(self, X):
    embs = self.drop(X)
    embs = self.fcs[0](embs)
    for fc in self.fcs[1:]:
      embs = fc(self.drop(F.relu(embs)))
    return embs


class PPRGo(nn.Module):

  def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
    super().__init__()
    self.mlp = MLP(num_features, num_classes, hidden_size, nlayers,
                        dropout)

  def forward(self, X, ppr_scores, ppr_idx):
    logits = self.mlp(X)
    propagated_logits = scatter(logits * ppr_scores[:, None],
                                ppr_idx[:, None],
                                dim=0,
                                dim_size=ppr_idx[-1] + 1,
                                reduce='sum')
    return propagated_logits


class AttentionHead(nn.Module):

  def __init__(self, input_size, head_size):
    super().__init__()
    self.qk_proj = nn.Linear(input_size, head_size, bias=False)
    self.k_proj = nn.Linear(input_size, head_size, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self,
              query,
              key,
              value,
              group_idx,
              prior_scores,
              return_rankings=False):
    # query: (batch_neighbor, input_size)
    # key: (batch_neighbor, input_size)
    x_query = self.qk_proj(query)
    x_key = self.k_proj(key)
    e_attn = self.sigmoid(x_query.mul(x_key).sum(-1))
    output = scatter(value * e_attn[:, None] * prior_scores[:, None],
                     group_idx[:, None],
                     dim=0,
                     dim_size=group_idx[-1] + 1,
                     reduce='sum')
    if return_rankings:
      return output, e_attn
    else:
      return output


class GNNAttn(nn.Module):

  def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
    super().__init__()
    self.mlp = MLP(num_features, num_classes, hidden_size, nlayers,
                        dropout)
    self.attention_head = AttentionHead(num_classes, 32)  # TODO: hard coded

  def forward(self,
              source_attr,
              neighbor_attr,
              ppr_scores,
              ppr_idx,
              neighbor_idx,
              return_rankings=False):
    x_source = self.mlp(source_attr)
    x_neighbor = self.mlp(neighbor_attr)
    if return_rankings:
      propagated_logits, attn_weights = self.attention_head(
          x_source,
          x_neighbor,
          x_neighbor,
          ppr_idx,
          ppr_scores,
          return_rankings=True)
      return propagated_logits, ppr_idx, ppr_scores, attn_weights, neighbor_idx
    else:
      propagated_logits = self.attention_head(x_source, x_neighbor, x_neighbor,
                                              ppr_idx, ppr_scores)
      return propagated_logits
