"""Load pre-trained BERT model and get ASIN embedding in an unsupervised way."""

import gzip
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plac
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertModel,
                          BertTokenizerFast, InputFeatures,
                          PreTrainedTokenizer)

logger = logging.getLogger(__name__)


class AsinEmbeddingDataset(Dataset):

  def __init__(self,
               csv_file,
               tokenizer: PreTrainedTokenizer,
               max_length: Optional[int] = None):
    self.df = pd.read_csv(csv_file)
    self.df.ID = self.df.ID.astype(str)
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.keys = self.df["ID"].tolist()

    self.features = self._text_to_features(self.df["text"].tolist())
    assert len(self.features) == len(self.keys)

  def _text_to_features(self, texts: List[str]):
    batch_encoding = self.tokenizer.batch_encode_plus(
        texts, max_length=self.max_length, pad_to_max_length=True)

    features = []
    for i in range(len(texts)):
      inputs = {k: batch_encoding[k][i] for k in batch_encoding}
      feature = InputFeatures(**inputs)
      features.append(feature)
    return features

  def __len__(self):
    return len(self.features)

  def __getitem__(self, index):
    return (self.keys[index], self.features[index])


def collate_batch(examples):
  asins, features = list(zip(*examples))
  first = features[0]

  # Special handling for labels.
  # Ensure that tensor is created with the correct type
  # (it should be automatically the case, but let's make sure of it.)
  if hasattr(first, "label") and first.label is not None:
    if type(first.label) is int:
      labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
      labels = torch.tensor([f.label for f in features], dtype=torch.float)
    batch = {"labels": labels}
  elif hasattr(first, "label_ids") and first.label_ids is not None:
    if type(first.label_ids[0]) is int:
      labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    else:
      labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
    batch = {"labels": labels}
  else:
    batch = {}

  # Handling of all other possible attributes.
  # Again, we will use the first element to figure out which key/values are not None for this model.
  for k, v in vars(first).items():
    if k not in ("label",
                 "label_ids") and v is not None and not isinstance(v, str):
      batch[k] = torch.tensor([getattr(f, k) for f in features],
                              dtype=torch.long)

  return asins, batch


class AsinEmbedding:

  def __init__(self, dataset, tokenizer: BertTokenizerFast, model: BertModel, batch_size=32):
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.model = model.cuda()
    self.batch_size = batch_size

  def embed(self, use_cls=False, use_emb=False) -> Tuple[str, List[np.ndarray]]:
    data_loader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_batch)

    preds = []
    all_token_ids = []
    all_asins = []
    nu = 0
    with torch.no_grad():
      for batch in tqdm(data_loader, desc="embedding prediction"):
        # nu += 1
        # if nu > 10:
        #   break
        asins, inputs = batch
        all_asins.extend(asins)
        all_token_ids.append(inputs["input_ids"].data.cpu())
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model(**inputs)[0]  # outputs[0] gives sequence embedding
        preds.append(outputs.data.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    all_token_ids = torch.cat(all_token_ids, dim=0).numpy()
    if use_emb:
      logger.info("Use linear layer output as sequence embedding")
      asin_embeds = preds
    elif use_cls:
      logger.info("Using CLS as sentence embedding")
      asin_embeds = self._take_cls_embedding(preds, all_token_ids)
    else:
      logger.info("Using pooled tokens as sentence embedding")
      asin_embeds = self._pool_seq_embedding(preds, all_token_ids)
    # asin_embeds = self._take_cls_embedding(preds, all_token_ids)
    return all_asins, asin_embeds

  def _pool_seq_embedding(self, preds, all_token_ids):
    """Pool token embeddings from the sequence to obtain sequence embedding.
    Drop special tokens, average out all other tokens."""
    seq_embeds = []
    for i in tqdm(range(preds.shape[0]), desc="pool sequence embedding"):
      embeds = []
      token_count = 0
      tokens = self.tokenizer.convert_ids_to_tokens(all_token_ids[i])
      seq_outputs = preds[i]
      for token, seq_output in zip(tokens, seq_outputs):
        if token == self.tokenizer.pad_token:
          break

        token_count += 1
        if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
          # ignore [CLS] or [SEP]
          continue

        # if token.startswith("##"):
        #   continue

        embeds.append(seq_output)
      embeds = np.vstack(embeds)
      seq_embeds.append(embeds.mean(0))
    return seq_embeds

  def _take_cls_embedding(self, preds, all_token_ids):
    seq_embeds = []
    for i in tqdm(range(preds.shape[0]), desc="pool sequence embedding"):
      seq_embeds.append(preds[i, 0, :])
    return seq_embeds


def main(data_dir, dataset, bert_model, split, cls: ("Use cls instead of pooling", "flag", "c"), emb: ("Use Linear layer output as embedding", "flag", "e"), output_file=None):
  tokenizer = BertTokenizerFast.from_pretrained(bert_model)
  if emb:
    model = BertForSequenceClassification.from_pretrained(bert_model)
  else:
    model = BertModel.from_pretrained(bert_model)
  asin_dataset = AsinEmbeddingDataset(Path(data_dir, dataset, f"{split}_docs.csv"),
                                 tokenizer=tokenizer,
                                 max_length=128)
  embed_executor = AsinEmbedding(asin_dataset, tokenizer, model)
  asins, asin_embeds = embed_executor.embed(use_cls=cls, use_emb=emb)

  if output_file is not None:
    with gzip.open(output_file, "wt") as f:
      for asin, emb in zip(asins, asin_embeds):
        emb_str = ",".join([str(x) for x in emb])
        f.write(asin + "\t" + emb_str + "\n")
  else:
    with gzip.open(Path(data_dir, dataset, f"{split}.emb.tsv.gz"), "wt") as f:
      for asin, emb in zip(asins, asin_embeds):
        emb_str = ",".join([str(x) for x in emb])
        f.write(asin + "\t" + emb_str + "\n")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  plac.call(main)
