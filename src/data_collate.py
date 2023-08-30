import torch
from transformers import PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Union, Dict, Any

"""
注意: Padder最大长度暂设为512, batch size最大为8
"""
# for padding
max_len = 512
max_batch_size = 8


class LabelMatricesPadder:
    """
        from: [yhcc/CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER)
    """

    def __init__(self, batch_size=max_batch_size, max_len=max_len, pad_val=0):
        self.pad_val = pad_val
        self.buffer = torch.full(
            (batch_size, max_len, max_len),
            fill_value=self.pad_val,
            dtype=torch.float,
        ).clone()

    def __call__(self, field):
        max_len = max([len(f) for f in field])
        buffer = self.buffer[: len(field), :max_len, :max_len].clone()
        for i, f in enumerate(field):
            buffer[i, : len(f), : len(f)] = torch.tensor(f)
        return buffer


class ScatterIndicesPadder:
    def __init__(self, pad_val=0, batch_size=max_batch_size, max_len=max_len):
        self.pad_val = pad_val
        self.buffer = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_val,
            dtype=torch.long,
        ).clone()

    def __call__(self, field):
        max_len = max([len(f) for f in field])
        buffer = self.buffer[: len(field), :max_len].clone()
        for i, f in enumerate(field):
            buffer[i, : len(f)] = torch.tensor(f)
        return buffer


class CSSDataCollator:
    """
        reference: transformers DataCollatorForTokenClassification
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_val: int = 0,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

        self.label_pad_fn = LabelMatricesPadder(pad_val=pad_val)
        self.scatter_indices_pad_fn = ScatterIndicesPadder()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        span_labels = (
            [feature["span_labels"] for feature in features]
            if "span_labels" in features[0].keys()
            else None
        )

        span_unmask = (
            [feature["span_unmask"] for feature in features]
            if "span_unmask" in features[0].keys()
            else None
        )

        scatters = [feature["scatter_indices"] for feature in features]

        iats = [
            {
                k: v
                for k, v in feature.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            iats,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["scatter_indices"] = self.scatter_indices_pad_fn(scatters)

        if span_labels is None:
            return batch

        batch_size = len(features)
        batch["span_labels"] = torch.cat(
            (
                torch.zeros([batch_size, 1]),
                self.label_pad_fn(span_labels).view(batch_size, -1),
            ),
            dim=1,
        )

        batch["span_unmask"] = torch.cat(
            (
                torch.ones([batch_size, 1]),
                self.label_pad_fn(span_unmask).view(batch_size, -1),
            ),
            dim=1,
        )

        return batch
