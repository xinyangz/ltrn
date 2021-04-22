import gzip
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, Subset

from ppr import topk_ppr_matrix

logger = logging.getLogger(__name__)


def load_label_map(data_dir: Path, dataset: str) -> Dict[str, int]:
  """Read a mapping from label name to label index"""
  label_map = dict()
  with Path(data_dir, dataset, "label_list.txt").open("r") as f:
    for i, line in enumerate(f):
      label_map[line.strip()] = i
  return label_map


def load_metadata(fname: Path) -> Dict[str, Dict[str, float]]:
  """Read node metadata from json line file. Input metadata field can either be
  a list of metadata strings, or a dictionary of metadata string mapped to its weight."""
  doc2meta: Dict[str, Dict[str, float]] = dict()
  with Path(fname).open("r") as f:
    for line in f:
      meta_obj = json.loads(line)
      doc_id = str(meta_obj["ID"])
      assert "ID" in meta_obj and "metadata" in meta_obj
      if isinstance(meta_obj["metadata"], dict):
        doc2meta[doc_id] = {str(k): float(v) for k, v in meta_obj["metadata"].items()}
      elif isinstance(meta_obj["metadata"], list):
        doc2meta[doc_id] = {str(m): 1. for m in meta_obj["metadata"]}
      else:
        raise RuntimeError("Metadata field error")
  return doc2meta


def load_doc_feature(fname: Path) -> Dict[str, List[float]]:
  """Load document feature from bert embedding file"""
  doc2feature = dict()
  with gzip.open(fname, "rt") as f:
    for line in f:
      line_split = line.strip().split("\t")
      doc_id = line_split[0]
      vec = torch.Tensor([float(x) for x in line_split[1].split(",")])
      doc2feature[doc_id] = vec
  return doc2feature


def build_network(df_docs: pd.DataFrame,
                  doc2meta: Dict[str, Dict[str, float]]) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[str, int]]:
  """Build a network connecting document nodes to metadata nodes.
  Returns adj_matrix, doc2node, meta2node"""
  IDs = df_docs.ID.tolist()
  meta_nodes = set()
  for meta_dict in doc2meta.values():
    meta_nodes |= set(meta_dict.keys())
  ## nodes
  # doc nodes: 0...len(asin)
  doc2node: Dict[str, int] = dict(zip(IDs, range(len(IDs))))
  # meta nodes: len(asin)...len(asin) + len(meta)
  meta2node: Dict[str, int] = dict(zip(meta_nodes, range(len(IDs), len(IDs) + len(meta_nodes))))
  logger.info(f"{len(doc2node)} doc nodes, {len(meta2node)} meta nodes")
  n_total_nodes = len(doc2node) + len(meta2node)
  ## edges
  rows = []
  cols = []
  datas = []
  for doc_id, meta_dict in doc2meta.items():
    doc_node = doc2node[doc_id]
    for meta, weight in meta_dict.items():
      meta_node = meta2node[meta]
      rows.extend([doc_node, meta_node])
      cols.extend([meta_node, doc_node])
      datas.extend([weight, weight])
  adj_matrix = sp.csr_matrix((datas, (rows, cols)), shape=(n_total_nodes, n_total_nodes), dtype=float)
  return adj_matrix, doc2node, meta2node


def load_labeled_seeds(
    fname: Path,
    seed_limit: Optional[int] = None,
    with_probs=False,
) -> Union[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]]]:
  """Load seed ASINs (and probs) from file."""
  logger.info(f"Load seeds from {fname}")
  lb2seeds = defaultdict(list)
  with open(fname, "r") as f:
    for line in f:
      line_split = line.strip().split("\t")
      label = line_split[0]
      seed = line_split[1]
      if len(line_split) == 3:
        prob = float(line_split[2])
        lb2seeds[label].append((seed, prob))
      elif len(line_split) == 2:
        lb2seeds[label].append((seed, 1.0))
      else:
        raise RuntimeError("Wrong seed file format, expect (label ID) or (label ID prob)")
  lb2seeds: Dict[str, List[Tuple[str, float]]] = dict(lb2seeds)
  # set limit
  if seed_limit:
    for lb, seeds in lb2seeds.items():
      if len(seeds) > seed_limit:
        sampled_seeds = random.sample(seeds, seed_limit)
        lb2seeds[lb] = sampled_seeds
  if with_probs:
    return lb2seeds
  else:
    lb2seeds_no_probs: Dict[str, List[str]] = dict()
    for lb, seed_probs in lb2seeds.items():
      lb2seeds_no_probs[lb] = [asin for asin, probs in seed_probs]
    return lb2seeds_no_probs


class GNNDataset(Dataset):
  """An ASIN dataset for GNN. With graph structure induced
  by ASIN-metadata and ASIN features from BERT embedding.

  For each batch, return ((features, ppr_scores, neighbor_idx), labels)"""

  def __init__(
      self,
      data_args,
      mode="train",  # TODO: what about dev set?
      alpha=0.15,
      topk=50,
      epsilon=1e-4):
    """
    Args:
      - data_args
      - mode: train/dev/test
      - alpha: reset probability for PPR
      - topk: number of top PPR neighbor to keep for each node
      - epsilon: convergence threshold for PPR computation"""
    self.alpha = alpha
    self.topk = topk
    self.epsilon = epsilon

    is_eval = False
    if mode == "dev" or mode == "test":
      is_eval = True
    elif mode != "train":
      raise RuntimeError(f"Mode {mode} not supported")

    # load ID, label
    data_dir = data_args.data_dir
    dataset = data_args.dataset
    df_docs = pd.read_csv(Path(data_dir, dataset, "train_docs.csv"), dtype=str)
    df_label = pd.read_csv(Path(data_dir, dataset, "train_labels.csv"), dtype=str)

    # load pseudo labels
    if data_args.pseudo_label_file:
      logging.info(f"Loading pseudo labels from {data_args.pseudo_label_file}")
      df_pseudo_label = pd.read_csv(data_args.pseudo_label_file, dtype=str)
      df_label = pd.concat([df_label, df_pseudo_label], ignore_index=True)

    # load label mapping
    label_map = load_label_map(data_dir, dataset)

    # load metadata
    doc2meta = load_metadata(Path(data_dir, dataset, "train_metadata.jsonl"))

    # handle dev & test modes
    if is_eval:
      # mode = dev
      # must have labels
      # 1. append docs and metadata to train dataset -> build a network w/ train+dev
      # 2. set labels to dev labels
      # 3. set nodes to enuemrate as dev docs

      # mode = test
      # may or may not have labels
      # 1. append docs and metadata to train dataset -> build a network w/ train+test
      # 2. set doc nodes to enumerate as test docs
      df_eval_docs = pd.read_csv(Path(data_dir, dataset, f"{mode}_docs.csv"), dtype=str)
      if mode == "dev":
        assert "label" in df_eval_docs.columns
      has_label = "label" in df_eval_docs.columns
      if has_label:
        df_eval_doc_texts = df_eval_docs.drop(columns=["label"])
        df_eval_label = df_eval_docs[["ID", "label"]]
      eval_start_idx = len(df_docs)

      doc2meta_eval = load_metadata(Path(data_dir, dataset, f"{mode}_metadata.jsonl"))

      df_docs = pd.concat([df_docs, df_eval_doc_texts], ignore_index=True)
      doc2meta.update(doc2meta_eval)
      # end handling dev & test

    # build network
    logger.info("build network")
    adj_matrix, doc2node, meta2node = build_network(df_docs, doc2meta)

    # load features
    # doc features: load from a bert embedding file
    # meta features: zeros (aggregate from neighboring nodes)
    doc_ids = df_docs.ID.tolist()

    train_feature_file = Path(data_args.feature_dir, "train.emb.tsv.gz")
    logger.info(f"load features from {train_feature_file}")
    doc2feature = load_doc_feature(train_feature_file)
    if is_eval:
      eval_feature_file = Path(data_args.feature_dir, f"{mode}.emb.tsv.gz")
      logger.info(f"load features from {eval_feature_file}")
      doc2feature_eval = load_doc_feature(eval_feature_file)
      doc2feature.update(doc2feature_eval)

    doc_features = torch.stack([doc2feature[doc_id] for doc_id in doc_ids])
    feat_dim = doc_features.shape[-1]
    meta_features = torch.zeros(len(meta2node), feat_dim)
    features = torch.cat((doc_features, meta_features), dim=0)
    logger.info(f"Feature matrix size: {features.shape}")

    # set nodes for enumeration
    # use only labeled nodes for training and testing
    enum_nodes = []
    labels = None
    if is_eval:
      # assume doc nodes come first, train docs come before eval docs
      enum_nodes = list(range(eval_start_idx, len(doc2node)))
      if has_label:
        labels = []
        for i, (doc_id, label_str) in enumerate(zip(df_eval_label.ID, df_eval_label.label)):
          assert doc_id == doc_ids[enum_nodes[i]]
          labels.append(label_map[label_str])
    else:
      labels = []
      for doc_id, label_str in zip(df_label.ID, df_label.label):
        enum_nodes.append(doc2node[doc_id])
        labels.append(label_map[label_str])

    # compute top K PPR neighbors
    # only doc node neighbors are kept
    logger.info("Compute PPR")
    doc_node_indices = list(range(len(doc2node)))
    self.ppr_matrix = topk_ppr_matrix(adj_matrix, alpha, epsilon, doc_node_indices, topk, keep_nodes=doc_node_indices)

    # data to keep in class object
    self.doc_ids = doc_ids
    self.doc2node = doc2node
    self.features = features
    self.enum_nodes = torch.LongTensor(enum_nodes)
    self.labels = torch.LongTensor(labels)
    self.adj_matrix = adj_matrix
    self.n_classes = len(label_map)
    self.use_all = False

  def __len__(self):
    if self.use_all:
      return len(self.doc_ids)
    else:
      return len(self.enum_nodes)

  def __getitem__(self, indices: List[int]):
    # idx is a list of indices (using batch_sampler)
    # cache disabled due to extensive memory usage
    if self.use_all:
      raw_indices = indices
    else:  # use only labeled doc nodes
      raw_indices = self.enum_nodes[indices]
    # use raw indices to access adj_matrix and ppr_matrix
    ppr_matrix = self.ppr_matrix[raw_indices]
    ppr_matrix.eliminate_zeros()
    source_idx, neighbor_idx = ppr_matrix.nonzero()
    ppr_scores = ppr_matrix.data

    attr_matrix = self.features[neighbor_idx]
    ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)
    source_idx = torch.tensor(source_idx, dtype=torch.long)

    if self.labels is None or self.use_all:
      labels = None
    else:
      labels = self.labels[indices]
    return ((attr_matrix, ppr_scores, source_idx), labels)

  def doc_indices(self):
    """Returns node indices of documents of interest. All documents if
    in full mode, labeled documents if in labeled mode"""
    if self.use_all:
      return list(range(len(self.doc_ids)))
    else:
      return self.enum_nodes

  def get_doc_id(self, idx):
    if self.use_all:
      return self.doc_ids[idx]
    else:
      return self.doc_ids[self.enum_nodes[idx]]

  def use_all_document_nodes(self):
    """Switch the dataset to enumerate all document nodes, labeled & unlabeled."""
    self.use_all = True;

  def use_labeled_document_nodes(self):
    """Switch the dataset to enumerate only labeled document nodes."""
    self.use_all = False



class GNNAttnDataset(GNNDataset):
  """An GNN dataset for PPR-Attention-GNN. With graph structure induced
  by GNN-metadata and GNN features from BERT embedding.

  For each batch, return ((source_attr, neighbor_attr, ppr_scores, source_idx, neighbor_idx), labels)"""

  def __getitem__(self, indices):
    # idx is a list of indices (using batch_sampler)
    # cache disabled due to extensive memory usage
    if self.use_all:
      raw_indices = indices
    else:  # use only labeled doc nodes
      raw_indices = self.enum_nodes[indices]
    ppr_matrix = self.ppr_matrix[raw_indices]
    ppr_matrix.eliminate_zeros()
    source_idx, neighbor_idx = ppr_matrix.nonzero()
    ppr_scores = ppr_matrix.data

    # size: (batch_neighbors, feature_dim)
    source_attr = self.features[source_idx]
    neighbor_attr = self.features[neighbor_idx]
    ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)
    source_idx = torch.tensor(source_idx, dtype=torch.long)
    neighbor_idx = torch.tensor(neighbor_idx, dtype=torch.long)

    if self.labels is None or self.use_all:
      labels = None
    else:
      labels = self.labels[indices]
    return ((source_attr, neighbor_attr, ppr_scores, source_idx, neighbor_idx), labels)
