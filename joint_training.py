import argparse
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def parse_args():
  parser = argparse.ArgumentParser()
  # data arguments
  parser.add_argument("--output_dir",
                      type=Path,
                      default=None)
  parser.add_argument("--overwrite_output_dir", action="store_true")
  parser.add_argument("--data_dir",
                      type=Path,
                      default=None,
                      help="Path to dataset.")
  parser.add_argument("--dataset",
                      type=str,
                      default="books",
                      help="Name of the dataset.")
  parser.add_argument("--gpu",
                      type=str,
                      default="0,1,2,3,4,5",
                      help="GPU to use")
  parser.add_argument("--cotrain_iter",
                      type=int,
                      default=3,
                      help="Number of co-training iteration")
  parser.add_argument("--master_addr",
                      type=str,
                      default="127.0.0.1",
                      help="Master node address for distributed training.")
  parser.add_argument("--master_port",
                      type=str,
                      default="10086",
                      help="Master node port for distributed training.")
  parser.add_argument("--random_seed", type=int, default=0)
  parser.add_argument("--exp_name",
                      type=str,
                      default=None,
                      help="Comments to experiment.")
  parser.add_argument("--wandb_project",
                      type=str,
                      default=None,
                      help="W&B project name.")
  parser.add_argument("--conf_threshold_text",
                      type=float,
                      default=0.5,
                      help="Confident threshold for BERT prediction.")
  parser.add_argument("--conf_threshold_graph",
                      type=float,
                      default=0.95,
                      help="Confident threshold for GNN prediction.")
  parser.add_argument("--topk",
                      type=int,
                      default=50,
                      help="Number of top examples to take for each co-training iteration.")
  parser.add_argument("--no_feat_share",
                      action="store_true",
                      help="Ablation: turn off feature sharing.")
  parser.add_argument("--fp16", action="store_true", help="Mixed precision training for BERT (GNN still use FP32)")
  # bert model arguments
  bert_parser = parser.add_argument_group("BERT arguments")
  bert_parser.add_argument("--bert_model_name_or_path",
                           type=Path,
                           default="bert-base-uncased",
                           help="Path to pretrained BERT model.")
  bert_parser.add_argument("--bert_learning_rate",
                           type=float,
                           default=2e-5,
                           help="BERT learning rate.")
  bert_parser.add_argument("--bert_cls_learning_rate",
                           type=float,
                           default=2e-5,
                           help="BERT cls layer learning rate.")
  bert_parser.add_argument("--bert_max_seq_length",
                           type=int,
                           default=128,
                           help="BERT maximum sequence length (maximum 512).")
  bert_parser.add_argument("--bert_per_device_train_batch_size",
                           type=int,
                           default=64,
                           help="Training batch size for BERT per device.")
  bert_parser.add_argument("--bert_max_steps",
                           type=int,
                           default=500,
                           help="BERT steps for each co-train iteration.")
  bert_parser.add_argument("--bert_eval_steps",
                           type=int,
                           default=100,
                           help="Number of steps per eval for BERT model.")
  bert_parser.add_argument("--bert_logging_steps", type=int, default=100, help="Number of steps per logging.")
  # gnn model arguments
  gnn_parser = parser.add_argument_group("GNN arguments")
  gnn_parser.add_argument("--gnn_max_steps",
                          type=int,
                          default=200,
                          help="GNN steps for each co-train iteration.")
  gnn_parser.add_argument("--gnn_eval_steps",
                          type=int,
                          default=10,
                          help="GNN steps for each logging and evaluation.")
  args = parser.parse_args()
  args.gpu = [int(d) for d in args.gpu.split(",")]

  # split args into GNN and BERT args
  general_args = None
  bert_args = None
  gnn_args = None
  for group in parser._action_groups:
    group_dict = {
        a.dest: getattr(args, a.dest, None) for a in group._group_actions
    }
    group_namespace = argparse.Namespace(**group_dict)
    if "optional" in group.title:
      general_args = group_namespace
    elif "BERT" in group.title:
      bert_args = group_namespace
    elif "GNN" in group.title:
      gnn_args = group_namespace
    else:
      logger.warning(f"Discard argument group {group.title}: {group_namespace}")

  # if 1 gpu given, use it for both gnn and bert
  # if more than 1 gpu given, use the first gpu for gnn, and the rest for bert
  if len(general_args.gpu) == 1:
    setattr(gnn_args, "gpu", str(general_args.gpu[0]))
    setattr(bert_args, "gpu", str(general_args.gpu[0]))
  elif len(general_args.gpu) > 1:
    setattr(gnn_args, "gpu", str(general_args.gpu[0]))
    setattr(bert_args, "gpu", ",".join([str(d) for d in general_args.gpu[1:]]))
  else:
    setattr(gnn_args, "gpu", None)
    setattr(bert_args, "gpu", None)

  # copy values from general args to bert and gnn args
  setattr(bert_args, "data_dir", general_args.data_dir)
  setattr(bert_args, "dataset", general_args.dataset)
  setattr(bert_args, "master_addr", general_args.master_addr)
  setattr(bert_args, "master_port", general_args.master_port)
  setattr(bert_args, "exp_name", general_args.exp_name)
  setattr(bert_args, "fp16", general_args.fp16)
  setattr(gnn_args, "data_dir", general_args.data_dir)
  setattr(gnn_args, "dataset", general_args.dataset)

  # handle output dir
  if general_args.output_dir.is_dir() and general_args.overwrite_output_dir:
    logger.warning(f"Overwriting output dir: {general_args.output_dir}")
    shutil.rmtree(general_args.output_dir)
    general_args.output_dir.mkdir()
  elif not general_args.output_dir.is_dir():
    logger.warning(f"Creating output dir {general_args.output_dir}")
    general_args.output_dir.mkdir()
  else:
    logger.error("Output dir exists, set --overwrite_output_dir to continue")
    raise RuntimeError("Output dir exists, set --overwrite_output_dir to continue")

  return general_args, bert_args, gnn_args


def args2list(args: argparse.Namespace, prefix=None) -> List[str]:
  """Convert argparse Namespace to list of keyword arguments."""
  kwargs = dict()
  for k, v in vars(args).items():
    if prefix and k.startswith(prefix):
      k = k[len(prefix):]
    if isinstance(v, list):
      v = ",".join([str(item) for item in v])
    kwargs["--" + str(k)] = str(v)
  cmd = list(chain(*kwargs.items()))
  return cmd


def train_gnn(args):
  env = os.environ
  kwarg_list = args2list(args, prefix="gnn_")
  cmd = ["python", "gnn/train_gnn.py", *kwarg_list]
  subprocess.call(cmd, env=env)


def train_bert(args):
  devices = args.gpu.split(",")
  n_devices = len(devices)
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = args.gpu
  master_addr = args.master_addr
  master_port = args.master_port
  del args.master_addr
  del args.master_port
  del args.gpu
  kwarg_list = args2list(args, prefix="bert_")
  cmd = [
      "python", "-m", "torch.distributed.launch", "--master_addr", master_addr,
      "--master_port", master_port, "--nproc_per_node",
      str(n_devices), "bert/train_bert.py", "--do_train", "--do_eval",
      "--do_predict", "--evaluation_strategy", "steps",
      "--save_total_limit", "2",
      "--overwrite_output_dir", *kwarg_list
  ]
  if args.fp16:
    cmd.append("--fp16")
  subprocess.call(cmd, env=env)


def main(general_args, bert_args, gnn_args):
  data_dir = general_args.data_dir
  dataset = general_args.dataset
  micro_f1_scores = []
  macro_f1_scores = []
  for it in range(general_args.cotrain_iter):
    logger.info(f"********** Co-Training Iter {it} ************")

    # set pseudo label file
    if it > 0:  # use generated seed for later iterations
      pseudo_label_file = Path(general_args.output_dir, f"pseudo_label.iter{it - 1}.csv")
    else:
      pseudo_label_file = None

    # set training parameters for BERT
    setattr(bert_args, "output_dir",
            Path(general_args.output_dir, f"bert_iter_{it}"))
    setattr(bert_args, "cache_dir", bert_args.output_dir)
    if pseudo_label_file:
      setattr(bert_args, "pseudo_label_file", pseudo_label_file)
    setattr(bert_args, "iter_num", it)

    # set training parameters for GNN
    setattr(gnn_args, "output_dir",
            Path(general_args.output_dir, f"gnn_iter_{it}"))
    if pseudo_label_file:
      setattr(gnn_args, "pseudo_label_file", pseudo_label_file)
    setattr(gnn_args, "iter_num", it)

    # feature sharing
    if it > 0 and (not general_args.no_feat_share):
      feature_dir = Path(general_args.output_dir, f"bert_iter_{it - 1}")
    else:
      feature_dir = Path(data_dir, dataset)
    setattr(gnn_args, "feature_dir", feature_dir)

    # train both modules
    gnn_job = mp.Process(target=train_gnn, args=(gnn_args,))
    bert_job = mp.Process(target=train_bert, args=(bert_args,))
    gnn_job.start()
    bert_job.start()
    gnn_job.join()
    bert_job.join()

    # get BERT performance
    performance_file = Path(general_args.output_dir, f"bert_iter_{it}", f"eval_results_{dataset}.txt")
    with performance_file.open("r") as f:
      lines = f.read()
    mi_f1 = float(re.search(r"test_ACC = (\d+\.\d*)", lines).group(1))
    ma_f1 = float(re.search(r"test_Ma-F1 = (\d+\.\d*)", lines).group(1))
    micro_f1_scores.append(mi_f1)
    macro_f1_scores.append(ma_f1)

    # take confident prediction
    env = os.environ
    new_seed_file = Path(general_args.output_dir, f"pseudo_label.iter{it}.csv")
    gnn_pred_file = Path(gnn_args.output_dir, f"gnn_preds_iter{it}.npy")
    gnn_id_file = Path(gnn_args.output_dir, f"gnn_ids_iter{it}.txt")
    bert_pred_file = Path(bert_args.output_dir, f"bert_preds_iter{it}.npy")
    bert_id_file = Path(bert_args.output_dir, f"bert_ids_iter{it}.txt")
    subprocess.call([
        "python", "get_pseudo_labels.py", "--data_dir", general_args.data_dir,
        "--dataset", general_args.dataset,
        "--output_file", new_seed_file, "--pred_file_graph", gnn_pred_file,
        "--id_file_graph", gnn_id_file, "--pred_file_text", bert_pred_file,
        "--id_file_text", bert_id_file, "--conf_threshold_text",
        str(general_args.conf_threshold_text), "--conf_threshold_graph",
        str(general_args.conf_threshold_graph), "--topk",
        str(general_args.topk)
    ],
                    env=env)
  logger.info("All done")
  logger.info("BERT Performance")
  with Path(general_args.output_dir, f"cotrain_results.txt").open("w") as f:
    for i, (mi_f1, ma_f1) in enumerate(zip(micro_f1_scores, macro_f1_scores)):
      logger.info(f"Iter {i},  Mi-F1  {mi_f1:.4f}  Ma-F1  {ma_f1:.4f}")
      f.write(f"Iter {i},  Mi-F1  {mi_f1:.4f}  Ma-F1  {ma_f1:.4f}\n")



if __name__ == "__main__":
  os.environ["OMP_NUM_THREADS"] = "1"
  logging.basicConfig(level=logging.INFO)
  general_args, bert_args, gnn_args = parse_args()
  if general_args.wandb_project:
    os.environ["WANDB_PROJECT"] = general_args.wandb_project
  else:
    os.environ["WANDB_DISABLED"] = "1"
  main(general_args, bert_args, gnn_args)
