import logging
from typing import Dict, NamedTuple, Optional

import numpy as np
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.integrations import is_wandb_available
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

if is_wandb_available():
  import wandb
class PredictEmbedOutput(NamedTuple):
  predictions: np.ndarray
  label_ids: Optional[np.ndarray]
  metrics: Optional[Dict[str, float]]
  embeddings: np.ndarray

class CustomTrainer(Trainer):
  """Trainer which saves best model in training and allows setting different
  learning rate for BERT and classification layer."""

  def create_optimizer_and_scheduler(self, num_training_steps: int):
    """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or override this method in a subclass.
        """
    if self.optimizer is not None:
      return
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    # set different learning rate for BERT and classifier if specified
    if hasattr(self.args, "cls_learning_rate") \
      and self.args.cls_learning_rate != self.args.learning_rate:
      logger.warning("Setting different learning rate for BERT & CLS layer")
      cls_lr = self.args.cls_learning_rate
    else:
      cls_lr = self.args.learning_rate

    bert_params_decay = []
    bert_params_nodecay = []
    cls_params = []
    registered_params = set()
    for name, param in self.model.bert.named_parameters():
      if any(nd in name for nd in no_decay):
        bert_params_nodecay.append(param)
      else:
        bert_params_decay.append(param)
      registered_params.add(param)
    for name, param in self.model.classifier.named_parameters():
      cls_params.append(param)
      registered_params.add(param)

    # check that all parameters are registered
    for name, param in self.model.named_parameters():
      if "loss_fct" in name:
        continue
      assert param in registered_params, name

    optimizer_grouped_parameters = [
        {
            "params": bert_params_decay,
            "weight_decay": self.args.weight_decay,
            "lr": self.args.learning_rate
        },
        {
            "params": bert_params_nodecay,
            "weight_decay": 0.0,
            "lr": self.args.learning_rate
        },
        {
            "params": cls_params,
            "weight_decay": self.args.weight_decay,
            "lr": cls_lr
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=self.args.warmup_steps,
                                                num_training_steps=num_training_steps)
    self.optimizer = optimizer
    self.lr_scheduler = scheduler

  # def _setup_wandb(self):
  #   # TODO: use wandb callback
  #   if self.is_world_process_zero():
  #     wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"),
  #                config=vars(self.args),
  #                reinit=True)

  def predict_and_embed(self, embed_dataset: Dataset) -> PredictEmbedOutput:
    self.model.output_pooled_embeddings = True
    pred_outputs = self.predict(embed_dataset)
    prediction_embeddings, label_ids, metrics = pred_outputs
    predictions = prediction_embeddings[0]
    embeddings = prediction_embeddings[1]
    return PredictEmbedOutput(predictions=predictions, label_ids=label_ids, metrics=metrics, embeddings=embeddings)
