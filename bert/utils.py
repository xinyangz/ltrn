from operator import itemgetter
from pathlib import Path
from typing import Dict

import numpy as np


def take_topk(dictionary, K, return_tuple=False, return_dict=False):
  d = sorted(dictionary.items(), key=itemgetter(1), reverse=True)
  if return_tuple:
    return d[:K]
  if return_dict:
    return {k: v for (k, v) in d[:K]}
  else:
    return [item[0] for item in d][:K]


def load_label_map(data_dir: Path, dataset: str) -> Dict[str, int]:
  """Read a mapping from label name to label index"""
  label_map = dict()
  with Path(data_dir, dataset, "label_list.txt").open("r") as f:
    for i, line in enumerate(f):
      label_map[line.strip()] = i
  return label_map

