import os
import sys
import logging
import wandb
import torch
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Union, Dict, Any

from css_modeling import CSSModelConfig, CSSModel
from css_trainer import CSSTrainer
from data_process import (
    prepare_split_features,
    postprocess_predictions,
)
from data_collate import CSSDataCollator
from eval_script import multi_span_evaluate_from_file

"""
注意~~Padder最大: batch_size=8, max_len=512
"""

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    hidden_dropout_prob: float = field(default=0.1)
    biaffine_head: int = field(default=12)
    cnn_depth: int = field(default=6)
    cnn_kernel_size: int = field(default=3)
    head_dropout_prob: float = field(default=0.1)
    init_temperature: float = field(default=0.1)
    span_loss_weight: float = field(default=0.5)

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    def to_dict(self):
        return self.__dict__


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(default="")
    data_dir: Optional[str] = field(default="")
    train_file: Optional[str] = field(default="train.json")
    validation_file: Optional[str] = field(default="valid.json")
    test_file: Optional[str] = field(default="test.json")
    question_column_name: Optional[str] = field(default="question")
    context_column_name: Optional[str] = field(default="context")
    label_column_name: Optional[str] = field(default="label")
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_num_span: int = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    # important #
    max_seq_length: int = field(default=512)
    doc_stride: int = field(default=128)
    boundary_smoothing: bool = field(default=False)
    overwrite_cache: bool = field(
        default=False, metadata={"help": "whether to overwrite the cache"}
    )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    def to_dict(self):
        return self.__dict__


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if sys.argv[-1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb init
    if (
        training_args.report_to == "wandb"
        and training_args.do_train
        and training_args.local_rank < 1
    ):
        model_args_dict = model_args.to_dict()
        data_args_dict = data_args.to_dict()
        training_args_dict = training_args.to_dict()

        if (
            set(model_args_dict.keys())
            & set(data_args_dict.keys())
            & set(training_args_dict.keys())
        ):
            raise Exception("model_args_dict, data_args, training_args包含重复键")

        if not training_args.resume_from_checkpoint:
            wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                # name=training_args.run_name,
                name=training_args.output_dir.split("/")[-1],
                config={**model_args_dict, **data_args_dict, **training_args_dict},
            )
        else:
            last_run_id = training_args.training_args.run_name
            wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                id=last_run_id,
                resume="must",
            )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logfilename = os.path.join(training_args.output_dir, "train.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfilename, mode="w"),
        ],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"model_args: \n{model_args}")
    # logger.info(f"data_args: \n{data_args}")
    # logger.info(f"Training/evaluation parameters: \n{training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not training_args.do_predict:
        data_files = {
            "train": os.path.join(data_args.data_dir, data_args.train_file),
            "validation": os.path.join(data_args.data_dir, data_args.validation_file),
        }
    if training_args.do_predict:
        data_files = {"test": os.path.join(data_args.data_dir, data_args.test_file)}

    raw_datasets = load_dataset("json", field="data", data_files=data_files)

    pretrained_model_name_or_path = (
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path
    )
    if "roberta" in pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
    elif "albert" in pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
            # add_prefix_space=True,
        )
    elif "bert" in pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
            # add_prefix_space=True,
        )
    logger.info("===== Init the model =====")
    config = CSSModelConfig(
        pretrained_model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        biaffine_head=model_args.biaffine_head,
        cnn_depth=model_args.cnn_depth,
        cnn_kernel_size=model_args.cnn_kernel_size,
        head_dropout_prob=model_args.head_dropout_prob,
        init_temperature=model_args.init_temperature,
        span_loss_weight=model_args.span_loss_weight,
    )
    model = CSSModel(config)

    # # print model parameters
    if training_args.do_train and training_args.local_rank < 1:
        print("^" * 100)
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                print(name, num_params)
                total_params += num_params
        print(f"Total number of parameters: {total_params}")
        print("-" * 100)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this requirement"
        )

    if training_args.do_train:
        # shape: (5230, 7)
        train_examples = raw_datasets["train"]
        # ['id', 'type', 'question', 'context', 'num_span', 'label', 'structure']
        train_column_names = raw_datasets["train"].column_names
        if data_args.max_train_samples is not None:
            train_examples = train_examples.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_examples.map(
                lambda x: prepare_split_features(
                    data_args=data_args,
                    tokenizer=tokenizer,
                    examples=x,
                    split="train",
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        eval_column_names = raw_datasets["validation"].column_names
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_examples.map(
                lambda x: prepare_split_features(
                    data_args=data_args, tokenizer=tokenizer, examples=x, split="valid"
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_examples = raw_datasets["test"]
        test_column_names = raw_datasets["test"].column_names
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(
                range(data_args.max_predict_samples)
            )
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_examples.map(
                lambda x: prepare_split_features(
                    data_args=data_args, tokenizer=tokenizer, examples=x, split="test"
                ),
                batched=True,
                remove_columns=test_column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    def post_processing_function(examples, features, predictions, prefix="eval"):
        pred_file = postprocess_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=prefix,
        )
        return pred_file

    data_collator = CSSDataCollator(tokenizer=tokenizer)

    trainer = CSSTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        data_files=data_files,  # for quick evaluation
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        post_process_function=post_processing_function,
        compute_metrics=multi_span_evaluate_from_file,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        # trainer.save_state()
    else:
        model.load_state_dict(
            torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
        )
        # model.load_state_dict(torch.load("pytorch_model.bin"))
        trainer.model = model
        logger.info('======\nloading ckpt from "pytorch_model.bin"\n======')

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_examples)
        trainer.log_metrics(split="eval", metrics=metrics)
        trainer.save_metrics(split="eval", metrics=metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        metrics = trainer.predict(predict_dataset, predict_examples)
        metrics["predict_samples"] = len(predict_examples)
        trainer.log_metrics(split="predict", metrics=metrics)
        trainer.save_metrics(split="predict", metrics=metrics)


if __name__ == "__main__":
    main()
