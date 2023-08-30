import os
import json
import collections
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Optional, Tuple, List, Union, Dict, Any

import logging

logger = logging.getLogger(__name__)


def prepare_split_features(
    data_args, tokenizer, examples, split="train"
):  # examples为1000条数据
    if data_args.max_seq_length > tokenizer.model_max_length:
        raise Exception("data_args.max_seq_length > tokenizer.model_max_length")

    tokenized_examples = tokenizer(
        examples[data_args.question_column_name],
        examples[data_args.context_column_name],
        max_length=data_args.max_seq_length,
        stride=data_args.doc_stride,  # 步长参数, 切分文本时重叠token数目
        padding=False,
        truncation="only_second",
        return_overflowing_tokens=True,  # 是否切分文本块, 文本长度过长则需要切分
        return_offsets_mapping=True,  # 获取从token到原文的映射
        is_split_into_words=True,  #  序列为list of strings
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    if split == "train":
        tokenized_examples["span_labels"] = []
        tokenized_examples["threshold_labels"] = []
        tokenized_examples["span_unmask"] = []
        tokenized_examples["threshold_unmask"] = []
    tokenized_examples["example_id"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sequence_ids"] = []
    tokenized_examples["scatter_indices"] = []

    tokenized_examples["input_length"] = []

    # i: 文本块索引; sample_index: 原始句子索引
    for i, sample_index in enumerate(sample_mapping):

        sequence_ids = tokenized_examples.sequence_ids(i)

        tokenized_examples["input_length"].append(
            len(tokenized_examples["input_ids"][i])
        )

        # Start token index of the current span in the text.
        # 找到context对应的的第一个token的索引
        first_ctx_token_index = sequence_ids.index(1)  # included
        last_ctx_token_index = len(sequence_ids) - 2  # included, 最后一个元素为None

        # 获得token位置对应的原始词的索引, 特殊token对应None; question, context均从0开始, 截断后的文本索引按原文本
        word_ids = tokenized_examples.word_ids(i)

        # 构造scatter_indices: question, seq, cls对应0, context从1开始(包括被截断的)
        first_wordid = word_ids[first_ctx_token_index]  # included
        last_wordid = word_ids[last_ctx_token_index]  # included
        scatter_indices = (
            [0]
            + [1] * (first_ctx_token_index - 2)
            + [0]
            + [
                wordid - first_wordid + 2
                for wordid in word_ids[first_ctx_token_index:-1]
            ]
            + [0]
        )
        tokenized_examples["scatter_indices"].append(scatter_indices)

        ctx_word_ids = word_ids[first_ctx_token_index:-1]
        for i in range(1, len(ctx_word_ids)):
            if ctx_word_ids[i] - ctx_word_ids[i - 1] > 1:
                raise Exception("出现tokenize后消失的词")

        if split == "train":
            # 构造label矩阵
            label = examples[data_args.label_column_name][sample_index]
            label = label[first_wordid : last_wordid + 1]
            xys = []
            x = None
            for ind, la in enumerate(label + ["O"]):
                if la == "B" and x is None:
                    x = ind
                if la == "O" and x is not None:
                    xys.append((x, ind - 1))
                    x = None
            context_len = last_wordid - first_wordid + 1

            label_matrix = np.zeros((context_len, context_len), dtype=np.float64)
            tokenized_examples["threshold_labels"].append(deepcopy(label_matrix))
            label_matrix = np.full((context_len, context_len), 0)
            for x, y in xys:
                label_matrix[x][y] = 1
            tokenized_examples["span_labels"].append(label_matrix)

            unmask_matrix = np.full((context_len, context_len), 1)
            unmask_matrix = np.triu(unmask_matrix)
            tokenized_examples["span_unmask"].append(deepcopy(unmask_matrix))
            for x, y in xys:
                unmask_matrix[x][y] = 0
            tokenized_examples["threshold_unmask"].append(unmask_matrix)

        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["word_ids"].append(word_ids)
        tokenized_examples["sequence_ids"].append(sequence_ids)

    return tokenized_examples


def postprocess_predictions(
    examples,
    features,
    predictions,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,  # 调用时为‘eval’
    log_level: Optional[int] = logging.WARNING,
):
    all_scores = predictions

    example_id_to_index = {
        k: i for i, k in enumerate(examples["id"])
    }  # k: example_id, v: example index
    features_per_example = collections.defaultdict(
        list
    )  # k: example index, v: list of feature index
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_ids = []

    # Logging.
    logger.setLevel(log_level)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        whole_ctx_len = len(example["context"])
        whole_ctx_scores = np.zeros((whole_ctx_len, whole_ctx_len))

        feature_indices = features_per_example[example_index]
        
        for feature_index in feature_indices:
            sequence_ids = features[feature_index]["sequence_ids"]
            word_ids = features[feature_index]["word_ids"]
            scores = deepcopy(all_scores[feature_index])  # deepcopy

            first_ctx_token_index = sequence_ids.index(1)  # included
            last_ctx_token_index = len(sequence_ids) - 2  # included, 最后一个元素为None
            first_wordid = word_ids[first_ctx_token_index]  # included
            last_wordid = word_ids[last_ctx_token_index]  # included

            chunk_ctx_len = last_wordid - first_wordid + 1
            scores = scores[:chunk_ctx_len, :chunk_ctx_len]

            temp = np.zeros((whole_ctx_len, whole_ctx_len))
            temp[
                first_wordid : last_wordid + 1,
                first_wordid : last_wordid + 1,
            ] = scores
            whole_ctx_scores = np.maximum(whole_ctx_scores, temp)

        context = example["context"]
        predict_entities = get_entities_from_scores(whole_ctx_scores, context)
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = list(set(predictions))
        all_ids.append(example["id"])

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"{prefix}_predictions.json",
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    return prediction_file


def get_entities_from_scores(scores, context):
    shape = scores.shape
    l = len(context)
    assert shape[0] == l and shape[1] == l

    res = []
    for row in range(l):
        for col in range(row, l):
            if scores[row, col] == 1 and col >= row:
                res.append((" ".join(context[row : col + 1]), row, col))
    return res
