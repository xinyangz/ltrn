"""Take confident prediction and build new pseudo label set"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
      format="[Pseudo Label Gen] %(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    type=Path,
                    default=Path("/data/xinyaz/data/SP3"),
                    help="Path to dataset.")
parser.add_argument("--dataset",
                    type=str,
                    default="amlc-high",
                    help="Name of the dataset.")
parser.add_argument("--pred_file_graph",
                    type=Path,
                    default=None,
                    help="GNN model prediction.")
parser.add_argument("--id_file_graph",
                    type=Path,
                    default=None,
                    help="GNN model ID file aligned with prediction.")
parser.add_argument("--pred_file_text",
                    type=Path,
                    default=None,
                    help="Text model prediction.")
parser.add_argument("--id_file_text",
                    type=Path,
                    default=None,
                    help="Text model ID file aligned with prediction.")
parser.add_argument("--conf_threshold_graph",
                    type=float,
                    default=0.95,
                    help="Confidence threshold for GNN prediction.")
parser.add_argument("--conf_threshold_text",
                    type=float,
                    default=0.5,
                    help="Confidence threshold for GNN prediction.")
parser.add_argument(
    "--topk",
    type=int,
    default=50,
    help="Number of confident examples to take from each category.")
parser.add_argument("--output_file", type=Path, default=None, help="Output file for merged pseudo labels.")
args = parser.parse_args()


def load_doc_ids(fname):
  doc_ids = []
  with open(fname, "r") as f:
    for line in f:
      doc_ids.append(line.strip())
  return doc_ids


def extend_seed(pred_probs, pred_ids, ind2lb, threshold=0.5):
  preds = np.argmax(pred_probs, axis=1)
  scores = pred_probs[range(pred_probs.shape[0]), preds]
  lb2preds = defaultdict(list)
  for i, (lb, score) in enumerate(zip(preds, scores)):
    # filter out predictions that are bellow conf_threshold
    if score < threshold:
      continue
    lb2preds[lb].append((i, score))
  # take top predictions
  lb2preds = dict(lb2preds)
  extended_seeds = []
  for lb_ind, pred_scores in lb2preds.items():
    pred_scores = sorted(pred_scores, key=lambda x: x[1], reverse=True)
    for ind, _ in pred_scores[:args.topk]:
      extended_seeds.append((ind2lb[lb_ind], pred_ids[ind]))
  return extended_seeds


def eval_seed(id2lb, seeds):
  n_correct = 0
  for pred_lb, doc_id in seeds:
    if pred_lb == id2lb[doc_id]:
      n_correct += 1
  if len(seeds) == 0:
    acc = 0
  else:
    acc = n_correct / len(seeds)
  logger.info(f"{len(seeds)} seeds, acc {acc:.4f}")


def remove_ori_labels(preds, ids, ori_seeds):
  ori_labeled_docs = set([doc_id for _, doc_id in ori_seeds])
  filtered_indices = []
  filtered_ids = []
  for i, doc_id in enumerate(ids):
    if doc_id not in ori_labeled_docs:
      filtered_indices.append(i)
      filtered_ids.append(doc_id)
  return preds[filtered_indices, :], filtered_ids


def merge_seeds(old_seeds, ext_seeds_a, ext_seeds_b):
  old_ids = set([doc_id for lb, doc_id in old_seeds])
  ext_ids_a = set([doc_id for lb, doc_id in ext_seeds_a])
  ext_ids_b = set([doc_id for lb, doc_id in ext_seeds_b])
  logger.info(
      f"Old {len(old_ids)}, source a: {len(ext_ids_a)}, source b: {len(ext_ids_b)}"
  )
  logger.info(f"source a&b overlap {len(ext_ids_a & ext_ids_b)}")
  logger.info(f"source a & old overlap {len(ext_ids_a & old_ids)}")
  logger.info(f"source b & old overlap {len(ext_ids_b & old_ids)}")
  logger.info(f"new {len((ext_ids_a | ext_ids_b) - old_ids)}")
  filtered_ids = ext_ids_a | ext_ids_b | old_ids
  merged_ids = set()
  new_seeds = []
  for lb, doc_id in old_seeds:
    if doc_id not in merged_ids and doc_id in filtered_ids:
      new_seeds.append((lb, doc_id))
      merged_ids.add(doc_id)
  for lb, doc_id in ext_seeds_a:
    if doc_id not in merged_ids and doc_id in filtered_ids:
      new_seeds.append((lb, doc_id))
      merged_ids.add(doc_id)
  for lb, doc_id in ext_seeds_b:
    if doc_id not in merged_ids and doc_id in filtered_ids:
      new_seeds.append((lb, doc_id))
      merged_ids.add(doc_id)
  return new_seeds

# read label map
label_map_file = Path(args.data_dir, args.dataset, "label_list.txt")
ind2lb = dict()
lb2ind = dict()
with open(label_map_file, "r") as f:
  for i, line in enumerate(f):
    ind2lb[i] = line.strip()
    lb2ind[line.strip()] = i

# read original labels
logger.info("Read original labels")
df_label = pd.read_csv(Path(args.data_dir, args.dataset, "train_labels.csv"), dtype=str)
ori_seeds = []
for doc_id, lb in zip(df_label.ID, df_label.label):
  ori_seeds.append((lb, doc_id))

# try to read full labels
full_label_file = Path(args.data_dir, args.dataset, "train_labels_all.csv")
has_full_labels = full_label_file.is_file()
if has_full_labels:
  logger.info("Found full labels for training set! Using them for pseudo label evaluation")
  df_full_label = pd.read_csv(full_label_file, dtype=str)
  df_full_label.ID = df_full_label.ID.astype(str)
  id2lb = dict(zip(df_full_label.ID, df_full_label.label.astype(str)))

# read model predictions
pred_probs_graph = np.load(args.pred_file_graph)
pred_probs_text = np.load(args.pred_file_text)
pred_ids_graph = load_doc_ids(args.id_file_graph)
pred_ids_text = load_doc_ids(args.id_file_text)

# remove original labels from predictions
pred_probs_graph, pred_ids_graph = remove_ori_labels(pred_probs_graph, pred_ids_graph, ori_seeds)
pred_probs_text, pred_ids_text = remove_ori_labels(pred_probs_text, pred_ids_text, ori_seeds)

ext_seeds_graph = extend_seed(pred_probs_graph, pred_ids_graph, ind2lb,
                              args.conf_threshold_graph)
ext_seeds_text = extend_seed(pred_probs_text, pred_ids_text, ind2lb,
                             args.conf_threshold_text)

if has_full_labels:
  eval_seed(id2lb, ext_seeds_graph)
  eval_seed(id2lb, ext_seeds_text)


merged_seeds = merge_seeds(ori_seeds, ext_seeds_graph, ext_seeds_text)
labels, ids = list(zip(*merged_seeds))


df_merged = pd.DataFrame({"ID": ids, "label": labels})
df_merged.to_csv(args.output_file, index=False)
