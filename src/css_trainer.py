import torch
from tqdm import tqdm
from transformers import Trainer
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import Optional, Tuple, List, Union, Dict, Any


class HFDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset.remove_columns(
            ["offset_mapping", "example_id", "word_ids", "sequence_ids"]
        )

    def __getitem__(self, idx):
        return self.dset[idx]

    def __len__(self):
        return len(self.dset)


class CSSTrainer(Trainer):
    def __init__(
        self,
        *args,
        data_files=None,
        eval_examples=None,
        post_process_function=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_files = data_files
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = HFDataset(self.eval_dataset)
        sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            # num_workers=4,
        )
        self.model.eval()
        with torch.no_grad():
            pred_list = []
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                pred = self.model(**batch)[0]
                pred_list.append(pred.cpu())
        all_pred = torch.vstack(pred_list).numpy()
        pred_file = self.post_process_function(
            self.eval_examples, self.eval_dataset, all_pred, prefix="eval"
        )
        try:
            metrics = self.compute_metrics(pred_file, self.data_files["validation"])
        except:
            metrics = {}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        pred_dataset=None,
        pred_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "test",
    ):
        hf_dataset = HFDataset(pred_dataset)
        sampler = SequentialSampler(hf_dataset)
        hf_dataloader = DataLoader(
            dataset=hf_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            # num_workers=4,
        )
        self.model.eval()
        with torch.no_grad():
            pred_list = []
            for batch in tqdm(hf_dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                pred = self.model(**batch)[0]
                pred_list.append(pred.cpu())
        all_pred = torch.vstack(pred_list).numpy()
        pred_file = self.post_process_function(
            pred_examples, pred_dataset, all_pred, prefix="test"
        )

        metrics = {"placeholder": 0.0}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics
