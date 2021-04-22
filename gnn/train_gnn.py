"""Train PPR-Attention-GNN with given seeds."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from dataset import GNNAttnDataset
from model import GNNAttn
from trainer import GNNTrainer

logger = logging.getLogger(__name__)


def parse_args():
  parser = argparse.ArgumentParser()
  # data arguments
  parser.add_argument("--data_dir", type=Path, default=Path("/data/xinyaz/data/SP3"), help="Path to dataset.")
  parser.add_argument("--dataset", type=str, default="amlc-high", help="Name of the dataset.")
  parser.add_argument("--pseudo_label_file", type=Path, default=None, help="Path to pseudo labels.")
  parser.add_argument(  # TODO: wire up bert model
      "--feature_dir", type=Path, default=None, help="Path to document features.")
  # training arguments
  parser.add_argument("--output_dir", type=Path, default=None, help="Path to model checkpoints and output.")
  parser.add_argument("--iter_num", type=int, default=0, help="Iteration number")
  parser.add_argument("--max_steps", type=int, default=3000, help="Number of training steps.")
  parser.add_argument(
      "--eval_steps",  # TODO: make validation optional
      type=int,
      default=300,
      help="Number of training steps per evaluation.")
  parser.add_argument("--batch_size", type=int, default=512)
  parser.add_argument("--gpu", type=int, default=15, help="GPU to use.")
  parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
  # model arguments
  parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden units in GNN.")
  parser.add_argument("--n_layers", type=int, default=2, help="Number of layers.")
  parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--alpha", type=float, default=0.15, help="Reset probability of PPR.")
  parser.add_argument("--epsilon",
                      type=float,
                      default=1e-4,
                      help="Convergence threshold for push based PPR computation.")
  parser.add_argument("--topk", type=int, default=50, help="The number of PPR neighbors to keep.")
  parser.add_argument("--rnd", type=int, default=0, help="Random seed.")

  args = parser.parse_args()
  return args


def main(args):
  # setup output dir
  if not args.output_dir.is_dir():
    args.output_dir.mkdir(parents=True)
  # random seed
  torch.manual_seed(args.rnd)

  logger.info("Load train data")
  train_set = GNNAttnDataset(args, mode="train")
  logger.info("Load dev data")
  dev_set = GNNAttnDataset(args, mode="dev")
  feat_dim = train_set.features.shape[1]
  n_classes = train_set.n_classes

  # device = torch.cuda.set_device(args.gpu)
  device = torch.device(f"cuda:{args.gpu}")

  logger.info("Create model")
  model = GNNAttn(feat_dim, n_classes, args.hidden_size, args.n_layers, args.dropout)
  trainer = GNNTrainer(
      model,
      dev_set,
      device=device,
      lr=args.lr,
      weight_decay=args.weight_decay,
      batch_size=args.batch_size,
      batch_mul_val=4,
      eval_steps=args.eval_steps,
      output_dir=args.output_dir,
  )

  logger.info("Start training")
  trainer.train(train_set, max_steps=args.max_steps)

  logger.info("Running prediction")

  train_set.use_all_document_nodes()
  pred_probs = trainer.predict(test_dataset=train_set)

  pred_file = Path(args.output_dir, f"gnn_preds_iter{args.iter_num}.npy")
  id_file = Path(args.output_dir, f"gnn_ids_iter{args.iter_num}.txt")
  np.save(pred_file, pred_probs)
  with open(id_file, "w") as f:
    for idx in range(len(train_set)):
      f.write(train_set.get_doc_id(idx) + "\n")
  logger.warning(f"Saved predicted probs to {pred_file}")
  logger.warning(f"Saved corresponding IDs to {id_file}")


if __name__ == "__main__":
  logging.basicConfig(
      format="[GNN] %(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  args = parse_args()
  main(args)
