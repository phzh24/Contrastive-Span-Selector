import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Union, Dict
from dataclasses import dataclass


ACTIVATION_FUNCTION = nn.GELU()

EVAL_PADDER = torch.full(
    (
        8,
        512,
        512,
    ),
    fill_value=0,
    dtype=torch.float,
)


@dataclass
class CSSModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    span_pred: torch.FloatTensor = None


class CSSModelConfig(PretrainedConfig):
    def __init__(
        self,
        pretrained_model_name_or_path=None,
        cache_dir=None,
        revision="main",
        use_auth_token=False,
        hidden_dropout_prob=0.1,
        biaffine_head=12,
        cnn_depth=6,
        cnn_kernel_size=3,
        head_dropout_prob=0.1,
        init_temperature=1,
        span_loss_weight=0.5,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        self.revision = revision
        self.use_auth_token = use_auth_token
        self.hidden_dropout_prob = hidden_dropout_prob
        self.biaffine_head = biaffine_head
        self.cnn_depth = cnn_depth
        self.cnn_kernel_size = cnn_kernel_size
        self.head_dropout_prob = head_dropout_prob
        self.init_temperature = init_temperature
        self.span_loss_weight = span_loss_weight


class CSSModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            revision=config.revision,
            use_auth_token=config.use_auth_token,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.hf_model = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
        )

        self.span_repr_dim = hf_config.hidden_size // 2

        self.q_mlp = nn.Sequential(
            nn.Dropout(config.head_dropout_prob),
            nn.Linear(hf_config.hidden_size, self.span_repr_dim),
            ACTIVATION_FUNCTION,
        )
        self.cls_mlp = nn.Sequential(
            nn.Dropout(config.head_dropout_prob),
            nn.Linear(hf_config.hidden_size, self.span_repr_dim),
            ACTIVATION_FUNCTION,
        )

        self.s_mlp = nn.Sequential(
            nn.Dropout(config.head_dropout_prob),
            nn.Linear(hf_config.hidden_size, self.span_repr_dim),
            ACTIVATION_FUNCTION,
        )
        self.e_mlp = nn.Sequential(
            nn.Dropout(config.head_dropout_prob),
            nn.Linear(hf_config.hidden_size, self.span_repr_dim),
            ACTIVATION_FUNCTION,
        )

        if config.biaffine_head > 0:
            self.multi_head_biaffine = MultiHeadBiaffine(
                self.span_repr_dim, self.span_repr_dim, n_head=config.biaffine_head
            )

        self.W = torch.nn.Parameter(
            torch.empty(self.span_repr_dim, self.span_repr_dim * 2 + 2)
        )
        torch.nn.init.xavier_normal_(self.W.data)

        if config.cnn_depth > 0:
            self.cnn = MaskCNN(
                self.span_repr_dim,
                self.span_repr_dim,
                cnn_kernel_size=config.cnn_kernel_size,
                cnn_depth=config.cnn_depth,
            )

        self.dropout = nn.Dropout(config.head_dropout_prob)

        if config.init_temperature > 0:
            self.logit_scale = torch.nn.Parameter(
                torch.ones([]) * np.log(1 / config.init_temperature)
            )

        self.span_loss_weight = config.span_loss_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        scatter_indices=None,
        span_labels=None,
        span_unmask=None,
    ):
        if token_type_ids is not None:
            outputs = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        else:
            outputs = self.hf_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

        # (batch_size, seq_token_len, hidden_size)
        last_hidden_state = outputs["last_hidden_state"]
        cls_repr = self.cls_mlp(last_hidden_state[:, 0, :])

        # word_repr: (batch_size, seq_word_len, hidden_size), 1目的是拿掉CLS等
        word_repr = scatter(
            last_hidden_state, index=scatter_indices, dim=1, reduce="max"
        )  # scatter()会补0
        # lengths: 每句word长度, (batch_size,)
        lengths, _ = scatter_indices.max(dim=-1)
        lengths = lengths - 1
        batch_size, seq_word_length = lengths.shape[0], int(lengths.max())

        question = word_repr[:, 1]
        question_repr = self.q_mlp(question)

        # biaffine
        context = word_repr[:, 2:]
        start = self.s_mlp(context)
        end = self.e_mlp(context)
        if hasattr(self, "multi_head_biaffine"):
            multihead = self.multi_head_biaffine(start, end)

        start = torch.cat([start, torch.ones_like(start[..., :1])], dim=-1)
        end = torch.cat([end, torch.ones_like(end[..., :1])], dim=-1)
        affined_cat = torch.cat(
            [
                self.dropout(start)
                .unsqueeze(2)
                .expand(
                    -1, -1, end.size(1), -1
                ),  # -1 means not changing the size of that dimension; expand作用: 在维度为1的维度上扩展张量
                self.dropout(end).unsqueeze(1).expand(-1, start.size(1), -1, -1),
            ],
            dim=-1,
        )

        affine = torch.einsum("bmnh,kh->bkmn", affined_cat, self.W)
        # bsz x dim x L x L

        if hasattr(self, "multi_head_biaffine"):
            ba_repr = multihead + affine
        else:
            ba_repr = affine

        if hasattr(self, "cnn"):
            mask = self._mask(lengths)
            cnn_repr = self.cnn(ba_repr, mask)
            span_repr = ba_repr + cnn_repr
        else:
            span_repr = ba_repr

        cls_and_flat_span = torch.cat(
            (
                cls_repr.unsqueeze(1),
                span_repr.permute(0, 2, 3, 1).view(
                    batch_size, seq_word_length * seq_word_length, -1
                ),
            ),
            dim=1,
        )

        if hasattr(self, "logit_scale"):
            scaled_similarity = self.logit_scale.exp() * torch.einsum(
                "bh,blh->bl", question_repr, cls_and_flat_span
            )
        else:
            scaled_similarity = torch.einsum(
                "bh,blh->bl", question_repr, cls_and_flat_span
            )

        if self.training:
            assert scaled_similarity.shape == span_labels.shape
            loss = self._loss(scaled_similarity, span_labels, span_unmask)

            return CSSModelOutput(loss=loss)

        cls, ctx = scaled_similarity[:, 0], scaled_similarity[:, 1:]
        pred = (
            (ctx > cls.unsqueeze(-1).expand(-1, ctx.shape[-1]))
            .view(batch_size, seq_word_length, seq_word_length)
            .float()
        )
        if len(pred.shape) == 3:
            expand_pred = EVAL_PADDER[: pred.shape[0], ...].clone().to(pred)
            expand_pred[:, : pred.shape[1], : pred.shape[2]] = pred
        else:
            expand_pred = EVAL_PADDER[:1, ...].clone().to(pred)
            expand_pred[:, : pred.shape[0], : pred.shape[1]] = pred

        return CSSModelOutput(span_pred=expand_pred)

    def _mask(self, seq_len, max_len=None):
        """
        reference: [fastnlp/fastNLP: A Modularized and Extensible NLP Framework.](https://github.com/fastnlp/fastNLP)
        """
        max_len = int(max_len) if max_len is not None else int(seq_len.max())
        assert (
            seq_len.ndim == 1
        ), f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
        batch_size = seq_len.shape[0]
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len < seq_len.unsqueeze(1)  # bsz x length x length
        mask = (mask[:, None] * mask.unsqueeze(-1)).triu()  # None: 增加一个维度
        mask = mask[:, None].eq(0)
        return mask

    def _loss(self, logits, labels, unmask):
        """
        unmask: cls, pos, neg为1; 下三角, padding为0.
        """
        loss_list = []
        exp_logits = (logits - 1e9 * (unmask == 0)).exp()
        batch_size = logits.shape[0]
        for i in range(batch_size):
            _unmask = unmask[i].int()
            _labels = labels[i]
            sum_cls_neg_exp_logit = (
                exp_logits[i]
                .masked_select(torch.bitwise_and(_unmask == 1, _labels == 0).bool())
                .sum()
            )
            cls_exp_logit = exp_logits[i, 0]
            all_pos_exp_logit = exp_logits[i].masked_select(
                torch.bitwise_and(_unmask == 1, _labels == 1).bool()
            )
            if all_pos_exp_logit.numel() == 0:
                span_loss = None
            else:
                num_pos = all_pos_exp_logit.shape[0]
                pos_loss_list = []
                for j in range(num_pos):
                    pos_loss = all_pos_exp_logit[j] / (
                        sum_cls_neg_exp_logit + all_pos_exp_logit[j]
                    )
                    pos_loss_list.append(pos_loss.log())
                if len(pos_loss_list):
                    span_loss = torch.stack(pos_loss_list).mean()
                else:
                    span_loss = None

            threshold_loss = cls_exp_logit / sum_cls_neg_exp_logit
            threshold_loss = threshold_loss.log()

            if span_loss is not None and threshold_loss != 0:
                loss_list.append(
                    -(
                        self.span_loss_weight * span_loss
                        + (1 - self.span_loss_weight) * threshold_loss
                    )
                )
            elif span_loss is not None:
                loss_list.append(-span_loss)
            else:
                loss_list.append(-threshold_loss)

        return torch.vstack(loss_list).mean()


class MultiHeadBiaffine(nn.Module):
    """
    from: [yhcc/CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER)
    """

    def __init__(self, dim, out=None, n_head=4):
        super().__init__()
        assert dim % n_head == 0
        in_head_dim = dim // n_head
        out = dim if out is None else out
        assert out % n_head == 0
        out_head_dim = out // n_head
        self.n_head = n_head
        self.W = nn.Parameter(
            nn.init.xavier_normal_(
                torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)
            )
        )
        self.out_channels = out

    def forward(self, h, v):
        """
        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_channels
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum(
            "blhx,hoxy,bkhy->bholk", h, self.W, v
        )  # b: batch_size; l, k:len_seq; h: head_num; x, y: 每头的输入feature_size; d:每头输出特征纬度,
        w = w.reshape(bsz, self.out_channels, max_len, max_len)  # h和d结合变为out_dim
        return w


class MaskConv2d(nn.Module):
    """
    from: [yhcc/CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER)
    """

    def __init__(self, input_channels, output_channels, cnn_kernel_size, padding):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=cnn_kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        x = self.conv2d(x)
        return x


class MaskCNN(nn.Module):
    """
    from: [yhcc/CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER)
    """

    def __init__(self, input_channels, output_channels, cnn_kernel_size=3, cnn_depth=6):
        super(MaskCNN, self).__init__()

        layers = []
        for _ in range(cnn_depth - 1):
            layers.extend(
                [
                    MaskConv2d(
                        input_channels,
                        input_channels,
                        cnn_kernel_size=cnn_kernel_size,
                        padding=(cnn_kernel_size - 1) // 2,
                    ),
                    nn.LayerNorm((input_channels,)),
                    ACTIVATION_FUNCTION,
                ]
            )
        layers.append(
            MaskConv2d(
                input_channels,
                output_channels,
                cnn_kernel_size=cnn_kernel_size,
                padding=(cnn_kernel_size - 1) // 2,
            )
        )
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x  # 用作residual
        for layer in self.cnns:
            if isinstance(layer, MaskConv2d):
                x = layer(x, mask)
            elif isinstance(layer, nn.LayerNorm):
                x = x + _x
                x = layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                _x = x
            else:
                x = layer(x)
        return x
