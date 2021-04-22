import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import Dataset
from tqdm import tqdm

from pytorch_utils import matrix_to_torch

logger = logging.getLogger(__name__)


def run_batch(model, xbs, yb, optimizer, train, return_preds=False):

  # Set model to training mode
  if train:
    model.train()
  else:
    model.eval()

  # zero the parameter gradients
  if train:
    optimizer.zero_grad()

  # forward
  with torch.set_grad_enabled(train):
    if return_preds:
      pred, source_idx, ppr_scores, attn_weights, neighbor_idx = model(*xbs, return_rankings=return_preds)
    else:
      pred = model(*xbs)
    loss = F.cross_entropy(pred, yb)
    top1 = torch.argmax(pred, dim=1)
    ncorrect = torch.sum(top1 == yb)

    # backward + optimize only if in training phase
    if train:
      loss.backward()
      optimizer.step()

  if return_preds:
    # pred shape (batch_size, n_classes)
    # attn_weights shape (batch_size * neighbor_size,)
    return loss, ncorrect.item(), pred, source_idx, ppr_scores, attn_weights, neighbor_idx
  else:
    return loss, ncorrect.item()


def get_local_logits(model, attr_matrix, batch_size=10000):
  device = next(model.parameters()).device

  nnodes = attr_matrix.shape[0]
  logits = []
  with torch.set_grad_enabled(False):
    for i in range(0, nnodes, batch_size):
      batch_attr = matrix_to_torch(attr_matrix[i:i + batch_size]).to(device)
      logits.append(model(batch_attr).to('cpu').numpy())
  logits = np.row_stack(logits)
  return logits


class GNNTrainer:

  def __init__(self,
               model,
               eval_dataset: Dataset,
               device='cpu',
               lr=1e-3,
               weight_decay=1e-4,
               batch_size=512,
               batch_mul_val=4,
               eval_steps=50,
               output_dir=None,
               listen_interrupt=False):
    self.model = model.to(device)
    self.lr = lr
    self.weight_decay = weight_decay
    self.batch_size = batch_size
    self.batch_mul_val = batch_mul_val
    self.eval_steps = eval_steps
    # TODO(xinyaz): distributed training
    self.eval_dataset = eval_dataset
    self.best_acc = 0
    self.best_step = 0
    self.output_dir = output_dir
    self.listen_interrupt = listen_interrupt

  def train(self,
            dataset: Dataset,
            max_steps: int,
            model_path: Optional[str] = None):
    device = next(self.model.parameters()).device

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            batch_size=self.batch_size,
            drop_last=False),
        batch_size=None,
        num_workers=16,
        pin_memory=True,
    )
    step = 0
    best_loss = np.inf

    optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=self.lr,
                                 weight_decay=self.weight_decay)

    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}

    loss = 0
    ncorrect = 0
    nsamples = 0
    best_state = {}
    self.global_step = 0
    num_train_epochs = max_steps // len(train_loader) + 1
    try:
      for epoch in range(num_train_epochs):
        for xbs, yb in train_loader:
          self.global_step += 1
          xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

          loss_batch, ncorr_batch = run_batch(self.model,
                                              xbs,
                                              yb,
                                              optimizer,
                                              train=True)
          loss += loss_batch
          ncorrect += ncorr_batch
          nsamples += yb.shape[0]

          if self.global_step > max_steps:
            break

          step += 1
          if step % self.eval_steps == 0:
            # update train stats
            train_loss = loss / nsamples
            train_acc = ncorrect / nsamples

            loss_hist['train'].append(train_loss)
            acc_hist['train'].append(train_acc)

            if self.eval_dataset is not None:
              # update val stats
              rnd_idx = list(range(len(self.eval_dataset)))
              np.random.shuffle(rnd_idx)
              rnd_idx = rnd_idx[:self.batch_mul_val * self.batch_size]
              xbs, yb = self.eval_dataset[rnd_idx]
              xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)
              val_loss, val_ncorr = run_batch(self.model,
                                              xbs,
                                              yb,
                                              None,
                                              train=False)
              val_acc = val_ncorr / (self.batch_mul_val * self.batch_size)
              if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_step = step

              loss_hist['val'].append(val_loss)
              acc_hist['val'].append(val_acc)

              logger.info(
                  f"Epoch {epoch}, step {step}: train {train_loss:.5f}, val {val_loss:.5f}"
              )
              logger.info(f"Train acc {train_acc:.5f}, val acc {val_acc:.5f}")

              if val_loss < best_loss:
                best_loss = val_loss
                # best_epoch = epoch
                best_state = {
                    key: value.cpu()
                    for key, value in self.model.state_dict().items()
                }
              # early stop only if this variable is set to True
              # elif early_stop and epoch >= best_epoch + patience:
              #   model.load_state_dict(best_state)
              #   return epoch + 1, loss_hist, acc_hist
            else:
              logger.info(f"Epoch {epoch}, step {step}: train {train_loss:.5f}")
    except KeyboardInterrupt as e:
      if not self.listen_interrupt:
        raise e
    if self.eval_dataset is not None:
      self.model.load_state_dict(best_state)
      if self.output_dir:
        output_file = Path(self.output_dir, "best_model.pt")
        logger.warning(f"Saving best model to {output_file}")
        torch.save(best_state, output_file)
    print(f"Best test acc {self.best_acc:.4f} at step {self.best_step}")

  def full_inference(self, dataset: Dataset, return_rankings=False):
    """Run full inference."""
    device = next(self.model.parameters()).device

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            batch_size=self.batch_size,
            drop_last=False),
        batch_size=None,
        num_workers=16,
        pin_memory=True,
    )
    loss = 0
    ncorrect = 0
    nsamples = 0
    source_indices = None
    neighbor_indices = None
    ppr_scores = None
    attn_scores = None
    preds = None
    for xbs, yb in tqdm(train_loader, desc="Full inference"):
      xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

      loss_batch, ncorr_batch, preds_b, source_idx_b, ppr_scores_b, attn_scores_b, neighbor_idx_b = run_batch(
          self.model, xbs, yb, None, train=False, return_preds=True)
      if source_indices is None:
        source_indices = source_idx_b.cpu().numpy()
        neighbor_indices = neighbor_idx_b.cpu().numpy()
        ppr_scores = ppr_scores_b.cpu().numpy()
        attn_scores = attn_scores_b.cpu().numpy()
        preds = preds_b.cpu().numpy()
      else:
        source_indices=  np.concatenate([source_indices, source_idx_b.cpu().numpy()])
        neighbor_indices=  np.concatenate([neighbor_indices, neighbor_idx_b.cpu().numpy()])
        ppr_scores = np.concatenate([ppr_scores, ppr_scores_b.cpu().numpy()])
        attn_scores = np.concatenate([attn_scores, attn_scores_b.cpu().numpy()])
        preds = np.vstack([preds, preds_b.cpu().numpy()])
      loss += loss_batch
      ncorrect += ncorr_batch
      nsamples += yb.shape[0]
    print(f"Inference accuracy: {ncorrect / nsamples:.4f}")
    if return_rankings:
      return preds, source_indices, neighbor_indices, ppr_scores, attn_scores
    else:
      return preds

  def predict(self,
              test_dataset: Dataset,
              inf_fraction=1.0,
              alpha=0.15,
              nprop=2,
              ppr_normalization='sym',
              batch_size_logits=10000):
    """Run prediction with fast inference. First, run MLP on a subset
    of nodes, then propagate the hidden representation with PPR."""
    self.model.eval()

    adj_matrix = test_dataset.adj_matrix
    attr_matrix = test_dataset.features

    start = time.time()
    if inf_fraction < 1.0:
      idx_sub = np.random.choice(adj_matrix.shape[0],
                                 int(inf_fraction * adj_matrix.shape[0]),
                                 replace=False)
      idx_sub.sort()
      attr_sub = attr_matrix[idx_sub]
      logits_sub = get_local_logits(self.model.mlp, attr_sub, batch_size_logits)
      local_logits = np.zeros([adj_matrix.shape[0], logits_sub.shape[1]],
                              dtype=np.float32)
      local_logits[idx_sub] = logits_sub
    else:
      local_logits = get_local_logits(self.model.mlp, attr_matrix,
                                      batch_size_logits)
    time_logits = time.time() - start

    start = time.time()
    logits = local_logits.copy()

    if ppr_normalization == 'sym':
      # Assume undirected (symmetric) adjacency matrix
      deg = adj_matrix.sum(1).A1
      deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
      for _ in range(nprop):
        logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (
            deg_sqrt_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'col':
      deg_col = adj_matrix.sum(0).A1
      deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
      for _ in range(nprop):
        logits = (1 - alpha) * (
            adj_matrix @ (deg_col_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'row':
      deg_row = adj_matrix.sum(1).A1
      deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
      for _ in range(nprop):
        logits = deg_row_inv_alpha[:, None] * (
            adj_matrix @ logits) + alpha * local_logits
    else:
      raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
    predictions = softmax(logits, axis=1)
    time_propagation = time.time() - start

    logger.info(
        f"Time logits {time_logits}, time propagation {time_propagation}")

    keep_indices = test_dataset.doc_indices()
    return predictions
