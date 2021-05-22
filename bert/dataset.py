import logging
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import datasets as hf_datasets
import pandas as pd
import torch
import torch.distributed
from filelock import FileLock
from transformers import PreTrainedTokenizer

import utils

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
  data_dir: str = field(
      metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."})
  dataset: str = field(metadata={"help": "The input dataset."})
  length_limit: int = field(default=2000, metadata={"help": "Truncate number of train examples"})
  max_seq_length: int = field(
      default=128,
      metadata={
          "help": "The maximum total input sequence length after tokenization. Sequences longer "
                  "than this will be truncated, sequences shorter will be padded."
      },
  )
  overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
  negative_samples: str = field(default="",
                                metadata={
                                    "help": "If specified, use samples from the given file as negative samples,"
                                            "use sampling if not given"
                                })
  seed_limit: Optional[int] = field(default=None, metadata={"help": "Maximum number of seed per class"})  # TODO: remove this
  # seed_file: Optional[str] = field(default=None, metadata={"help": "If specified, use only seed file for training."})
  # ori_seed_file: Optional[str] = field(default=None, metadata={"help": "Ground truth seed documents."})
  pseudo_label_file: Optional[str] = field(default=None, metadata={"help": "Pseudo labels."})


class Split(Enum):
  train = "train"
  dev = "dev"
  test = "test"


def get_process_dataset_func(label_map, tokenizer, max_seq_length):

  def process_dataset(examples):
    # Tokenize the texts
    args = ((examples["text"],))
    result = tokenizer(*args, padding="do_not_pad", max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_map is not None and "label" in examples:
      result["label"] = [label_map[str(l)] for l in examples["label"]]
    return result

  return process_dataset


def load_training_data(
    data_dir: Path,
    dataset: str,
    tokenizer: PreTrainedTokenizer,
    pseudo_label_file: Optional[Path] = None,
    max_seq_length=128,
    with_dev=True,
    cache_dir: Optional[Path] = None,
    overwrite_cache=False,
) -> Tuple[hf_datasets.DatasetDict, Dict[str, int]]:
  """Load text data with minimal supervision. Returns train/dev labeled data and unlabeled
  data for inference."""

  # always load label map
  label_map = utils.load_label_map(data_dir, dataset)

  # load dataset features from cache or file
  # deal with cache
  use_cache = cache_dir is not None
  if use_cache and not Path(cache_dir).is_dir():
    Path(cache_dir).mkdir(parents=True)

  cache_file = Path(cache_dir, "train_dev.datasets.cache") if use_cache else None
  lock_file = cache_file.with_suffix(".lock") if use_cache else Path("/tmp/train_dev.dataset.lock")

  # Make sure only the first process in distributed training processes the dataset,
  # and the others will use the cache.
  with FileLock(lock_file):
    if use_cache and cache_file.is_file() and overwrite_cache:
      logger.info("Overwrite cache file!")
      cache_file.unlink()
    if use_cache and cache_file.is_dir() and overwrite_cache:
      logger.info("Overwrite cache dir!")
      shutil.rmtree(cache_file)
  try:
    torch.distributed.barrier()
  except:
    pass

  with FileLock(lock_file):
    if use_cache and (cache_file.is_file() or cache_file.is_dir()):
      logger.warning(f"######## Loading from cache {cache_file} ##########")
      s = time.time()
      datasets = hf_datasets.DatasetDict.load_from_disk(cache_file)
      e = time.time()
      logger.info(f"Time {e - s:.4f}")
    else:
      # load raw input data
      df_train_unlabeled = pd.read_csv(Path(data_dir, dataset, "train_docs.csv"), dtype="str").set_index("ID")
      df_train_labels = pd.read_csv(Path(data_dir, dataset, "train_labels.csv"), dtype="str").set_index("ID")
      if pseudo_label_file:
        df_pseudo_labels = pd.read_csv(pseudo_label_file, dtype="str").set_index("ID")
      df_dev = None
      if with_dev:
        try:
          df_dev = pd.read_csv(Path(data_dir, dataset, "dev_docs.csv"), dtype=str).set_index("ID")
        except FileNotFoundError:
          logger.warning("Try loading dev labels but not found!")

      # split and process data
      df = df_train_unlabeled.join(df_train_labels, rsuffix="_train", how="left")
      if pseudo_label_file:
        # if pseudo label file provided, use them as training set
        logger.info("Using pseudo label file")
        df = df_train_unlabeled.join(df_pseudo_labels, rsuffix="_pseudo", how="left")
        df_train = df[~df.label.isnull()]
      else:
        df_train = df[~df.label.isnull()]
      df_train_full = df_train_unlabeled
      # unlabeled set is always the original unlabeled set
      # which means it may overlap with pseudo labeled set
      df_unlabeled = df[df.label.isnull()]
      train_dataset = hf_datasets.Dataset.from_pandas(df_train, split="train")
      full_train_dataset = hf_datasets.Dataset.from_pandas(df_train_full, split="full_train")
      unlabeled_dataset = hf_datasets.Dataset.from_pandas(df_unlabeled[["text"]], split="unlabeled")

      process_func = get_process_dataset_func(label_map, tokenizer, max_seq_length)

      train_dataset = train_dataset.map(process_func, batched=True)
      full_train_dataset = full_train_dataset.map(process_func, batched=True)
      unlabeled_dataset = unlabeled_dataset.map(process_func, batched=True)

      dev_dataset = None
      if df_dev is not None:
        dev_dataset = hf_datasets.Dataset\
          .from_pandas(df_dev, split="dev")\
          .map(process_func,batched=True)

      datasets = hf_datasets.DatasetDict({
          "train": train_dataset,
          "full_train": full_train_dataset,
          "unlabeled": unlabeled_dataset,
      })
      if dev_dataset is not None:
        datasets["dev"] = dev_dataset

      if use_cache:
        logger.info(f"Saving dataset to cache {cache_file}")
        s = time.time()
        datasets.save_to_disk(cache_file)
        e = time.time()
        logger.info(f"Time {e - s:.4f}")

  # print info
  logger.info("Unlabeled dataset")
  logger.info(datasets["unlabeled"])
  logger.info("(Pseudo) Labeled dataset for training")
  logger.info(datasets["train"])
  logger.info("Full training document set")
  logger.info(datasets["full_train"])
  dev_dataset = datasets.get("dev", None)
  if with_dev and dev_dataset is not None:
    logger.info("Dev dataset")
    logger.info(dev_dataset)
  elif with_dev and dev_dataset is None:
    logger.warning("with_dev = True but dev dataset not found!")
  index = random.choice(range(len(datasets["unlabeled"])))
  logger.info(f"Sample {index} of the unlabeled set: {datasets['unlabeled'][index]}.")
  index = random.choice(range(len(datasets["train"])))
  logger.info(f"Sample {index} of the training set: {datasets['train'][index]}.")
  index = random.choice(range(len(datasets["full_train"])))
  logger.info(f"Sample {index} of the training set: {datasets['full_train'][index]}.")

  return datasets, label_map


def load_test_data(
    data_dir: Path,
    dataset: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length=128,
    with_labels=True,
    cache_dir: Optional[Path] = None,
    overwrite_cache=False,
) -> Tuple[hf_datasets.DatasetDict, Dict[str, int]]:
  """Load text data with minimal supervision. Returns train/dev labeled data and unlabeled
  data for inference."""

  # always load label map
  label_map = utils.load_label_map(data_dir, dataset)

  # load dataset features from cache or file
  # deal with cache
  use_cache = cache_dir is not None
  if use_cache and not Path(cache_dir).is_dir():
    Path(cache_dir).mkdir(parents=True)

  cache_file = Path(cache_dir, "test.datasets.cache") if use_cache else None
  lock_file = cache_file.with_suffix(".lock") if use_cache else Path("/tmp/test.dataset.lock")

  # Make sure only the first process in distributed training processes the dataset,
  # and the others will use the cache.
  with FileLock(lock_file):
    if use_cache and cache_file.is_file() and overwrite_cache:
      logger.info("Overwrite cache file!")
      cache_file.unlink()
    if use_cache and cache_file.is_dir() and overwrite_cache:
      logger.info("Overwrite cache dir!")
      shutil.rmtree(cache_file)
  try:
    torch.distributed.barrier()
  except:
    pass

  with FileLock(lock_file):
    if use_cache and (cache_file.is_file() or cache_file.is_dir()):
      logger.warning(f"######## Loading from cache {cache_file} ##########")
      s = time.time()
      test_dataset = hf_datasets.Dataset.load_from_disk(cache_file)
      e = time.time()
      logger.info(f"Time {e - s:.4f}")
    else:
      # load raw input data
      df_test = pd.read_csv(Path(data_dir, dataset, "test_docs.csv"), dtype="str").set_index("ID")

      if not with_labels and "label" in df_test.columns:
        del df_test["label"]

      # split and process data
      process_func = get_process_dataset_func(label_map, tokenizer, max_seq_length)
      test_dataset = hf_datasets.Dataset.from_pandas(df_test, split="test")
      test_dataset = test_dataset.map(process_func, batched=True)

      if use_cache:
        logger.info(f"Saving dataset to cache {cache_file}")
        s = time.time()
        test_dataset.save_to_disk(cache_file)
        e = time.time()
        logger.info(f"Time {e - s:.4f}")

  # print info
  logger.info("Test dataset")
  logger.info(test_dataset)

  return test_dataset, label_map


def load_labeled_seeds(  # TODO: remove this
    fname: Path,
    seed_limit: Optional[int] = None,
    with_probs=False,
) -> Union[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]]]:
  """Load seed ASINs (and probs) from file."""
  logger.info(f"Loading seeds from {fname}")
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
