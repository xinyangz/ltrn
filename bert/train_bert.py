"""Fine-tuning BERT model using seed examples from a large labeled dataset, and
run prediction on the rest of the 'unlabaled' examples in the dataset."""

import datetime
import gzip
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import sklearn.metrics
from scipy.special import softmax
from transformers import (BertConfig, BertTokenizerFast,
                          DataCollatorWithPadding, EvalPrediction,
                          HfArgumentParser, TrainingArguments, set_seed)
from transformers.integrations import is_wandb_available, WandbCallback

import utils
from dataset import DataArguments, load_test_data, load_training_data
from model import BertClassification
from trainer import CustomTrainer

logger = logging.getLogger(__name__)

if is_wandb_available():
  import wandb


@dataclass
class ModelArguments:
  """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
  model_name_or_path: str = field(
      metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
  config_name: Optional[str] = field(default=None,
                                     metadata={"help": "Pretrained config name or path if not the same as model_name"})
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
  cache_dir: Optional[str] = field(
      default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
  freeze_bert: bool = field(default=False, metadata={"help": "Freeze BERT parameters"})
  exp_name: Optional[str] = field(default=None, metadata={"help": "Comments to experiment."})


@dataclass
class CustomTrainingArgs(TrainingArguments):
  cls_learning_rate: Optional[float] = field(default=1e-2, metadata={"help": "Learning rate for classification layer"})
  iter_num: Optional[int] = field(default=0,
                                  metadata={"help": "Number of training iteration, used for prediction file output."})


def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:

  def compute_metrics_fn(p: EvalPrediction):
    if isinstance(p.predictions, tuple):
      predictions = p.predictions[0]
    else:
      predictions = p.predictions
    preds = np.argmax(predictions, axis=1)
    labels = p.label_ids
    metrics = {
        "ACC": sklearn.metrics.accuracy_score(labels, preds),
        "Ma-F1": sklearn.metrics.f1_score(labels, preds, average="macro"),
    }
    return metrics

  return compute_metrics_fn


def parse_args():
  """Parse arguments from either json configuration file or command line."""
  parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArgs))

  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
  else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and
      not training_args.overwrite_output_dir):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

  return model_args, data_args, training_args


def setup_logging(training_args):
  logging.basicConfig(
      format="[BERT] %(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO if training_args.local_rank in [-1, 0] else logging.ERROR,
  )
  logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
      training_args.local_rank,
      training_args.device,
      training_args.n_gpu,
      bool(training_args.local_rank != -1),
      training_args.fp16,
  )
  logger.info("Training/evaluation parameters %s", training_args)


def main():
  # turn off some unnecessary logs
  logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

  # parse command line arguments
  model_args, data_args, training_args = parse_args()

  # Setup wandb name
  unique_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  model_name = f"BERT-{unique_id}"
  os.environ["WANDB_NAME"] = model_name

  # Setup logging
  setup_logging(training_args)

  # Set seed
  set_seed(training_args.seed)

  label_map = utils.load_label_map(data_args.data_dir, data_args.dataset)
  num_labels = len(label_map)

  # Load pretrained model and tokenizer
  config = BertConfig.from_pretrained(
      model_args.model_name_or_path,
      num_labels=num_labels,
      cache_dir=model_args.cache_dir,
  )
  tokenizer = BertTokenizerFast.from_pretrained(
      model_args.model_name_or_path,
      cache_dir=model_args.cache_dir,
  )
  model = BertClassification.from_pretrained(
      model_args.model_name_or_path,
      config=config,
      cache_dir=model_args.cache_dir,
      freeze_bert=model_args.freeze_bert,
      label_smoothing=0.,
  )

  datasets, _ = load_training_data(data_args.data_dir,
                                   data_args.dataset,
                                   tokenizer,
                                   pseudo_label_file=data_args.pseudo_label_file,
                                   max_seq_length=data_args.max_seq_length,
                                   with_dev=True,
                                   cache_dir=model_args.cache_dir,
                                   overwrite_cache=data_args.overwrite_cache)
  test_dataset, _ = load_test_data(data_args.data_dir,
                                   data_args.dataset,
                                   tokenizer,
                                   max_seq_length=data_args.max_seq_length,
                                   with_labels=True,
                                   cache_dir=model_args.cache_dir,
                                   overwrite_cache=data_args.overwrite_cache)

  train_dataset = datasets.get("train")
  full_train_dataset = datasets.get("full_train")  # labeled + unlabeled
  dev_dataset = datasets.get("dev", None)
  unlabeled_dataset = datasets.get("unlabeled")  # full - real_train

  # set load best metric
  training_args.load_best_model_at_end = True
  training_args.metric_for_best_model = "ACC"
  compute_metrics_fn = build_compute_metrics_fn()
  trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=dev_dataset if training_args.do_eval else None,
      compute_metrics=compute_metrics_fn,
      tokenizer=tokenizer,
      data_collator=DataCollatorWithPadding(tokenizer),
  )

  # set wandb global logs
  # if is_wandb_available() and trainer.is_world_process_zero():
  #   wandb.config.update(model_args)
  #   wandb.config.update(data_args)

  # Training
  if training_args.do_train:
    logger.info("*** Train ***")
    trainer.train()

  # Evaluation
  if training_args.do_eval:
    logger.info("*** Evaluate ***")

    eval_result = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    # save and log eval results
    output_eval_file = os.path.join(training_args.output_dir,
                                    f"eval_results_{data_args.dataset}.txt")
    if trainer.is_world_process_zero():
      with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results on test set {} *****".format(data_args.dataset))
        writer.write(f"***** Eval results on test set {data_args.dataset} *****\n")
        for key, value in eval_result.items():
          logger.info("  %s = %s", key, value)
          writer.write("%s = %s\n" % (key, value))


  # Prediction
  if training_args.do_predict:
    logger.info("*** Predict ***")
    # embedding on full train dataset
    pred_embed_output = trainer.predict_and_embed(full_train_dataset)
    pred_probs = softmax(pred_embed_output.predictions, axis=1)
    embeddings = pred_embed_output.embeddings
    if trainer.is_world_process_zero():
      probs_file = Path(training_args.output_dir, f"bert_preds_iter{training_args.iter_num}.npy")
      ids_file = Path(training_args.output_dir, f"bert_ids_iter{training_args.iter_num}.txt")
      logger.warning(f"Saving prediction on full train set to {probs_file}")
      np.save(probs_file, pred_probs)
      logger.warning(f"Saving associated IDs to {ids_file}")
      with ids_file.open("w") as f:
        for doc_id in full_train_dataset["ID"]:
          f.write(doc_id + "\n")

      embeds_file = Path(training_args.output_dir, f"train.emb.tsv.gz")
      logger.warning(f"Saving embeddings to {embeds_file}")
      with gzip.open(embeds_file, "wt") as f:
        for i, doc_id in enumerate(full_train_dataset["ID"]):
          emb = embeddings[i]
          emb_str = ",".join([str(x) for x in emb])
          f.write(doc_id + "\t" + emb_str + "\n")

    # embedding on full dev dataset
    if dev_dataset is not None:
      pred_embed_output = trainer.predict_and_embed(dev_dataset)
      embeddings = pred_embed_output.embeddings
      if trainer.is_world_process_zero():
        embeds_file = Path(training_args.output_dir, f"dev.emb.tsv.gz")
        logger.warning(f"Saving embeddings to {embeds_file}")
        with gzip.open(embeds_file, "wt") as f:
          for i, doc_id in enumerate(dev_dataset["ID"]):
            emb = embeddings[i]
            emb_str = ",".join([str(x) for x in emb])
            f.write(doc_id + "\t" + emb_str + "\n")


if __name__ == "__main__":
  main()
