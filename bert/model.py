import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class LabelSmoothedCrossEntropyLoss(nn.Module):
  """This loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

  def __init__(self, n_classes, smoothing=0.0, dim=-1):
    super().__init__()
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.n_classes = n_classes
    self.dim = dim

  def forward(self, pred, target):
    pred = pred.log_softmax(dim=self.dim)
    with torch.no_grad():
      true_dist = torch.zeros_like(pred)
      true_dist.fill_(self.smoothing / (self.n_classes - 1))
      true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

    return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


@dataclass
class BertOutput(SequenceClassifierOutput):
  pooled_embeddings: Optional[torch.FloatTensor] = None


class BertClassification(BertForSequenceClassification):
  """A BERT classification model with label smoothing and an option
  to output pooled representation for embedding."""

  def __init__(self, config, label_smoothing=0., freeze_bert=False):
    super(BertClassification, self).__init__(config)
    self.label_smoothing = label_smoothing
    self.output_pooled_embeddings = False
    self.force_return_dict = False
    self.loss_fct = LabelSmoothedCrossEntropyLoss(self.num_labels, smoothing=label_smoothing)
    if freeze_bert:
      logger.warning("BERT parameters frozen")
      for para in self.bert.parameters():
        para.requires_grad = False

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      output_pooled_embedding=None,
      return_dict=None,
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # if self.force_return_dict:
    #   return_dict = True

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_representation = outputs[1]

    pooled_output = self.dropout(pooled_representation)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regression
        loss = self.loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      # add hidden states and attention if they are here
      output = (logits,) + outputs[2:]

      if self.output_pooled_embeddings or output_pooled_embedding:
        outputs = outputs + (pooled_representation,)
      return ((loss,) + output) if loss is not None else output

    return BertOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        pooled_embeddings=pooled_representation,
    )
