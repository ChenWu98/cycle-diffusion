import os
from pathlib import Path
import time
import datetime
import json
import re
import logging
import warnings
import random
import math
import collections.abc
import shutil
from typing import Dict, Union, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers.optimization import AdamW, Adafactor, get_scheduler
from transformers.trainer_pt_utils import (
    get_parameter_names,
    reissue_pt_warnings,
    ShardSampler,
    torch_pad_and_concatenate,
    numpy_pad_and_concatenate,
)
from transformers.trainer import TrainerState
from transformers.trainer_utils import (
    IntervalStrategy,
    denumpify_detensorize,
)
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"
PREFIX_CHECKPOINT_DIR = "checkpoint"


def distributed_concat(tensor: Union[Tuple, List, torch.tensor], num_total_examples: Optional[int] = None):
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: distributed_concat(v, num_total_examples) for k, v in tensor.items()})
        elif tensor is None:
            return None
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        output_tensors = [t if len(t.shape) > 0 else t[None] for t in output_tensors]
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, dict):
        assert set(tensors.keys()) == set(new_tensors.keys())
        return type(tensors)({k: nested_concat(tensors[k], new_tensors[k], padding_index=padding_index) for k in tensors.keys()})
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif tensors is None:
        return None
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def nested_cpu(tensors):
    "CPU `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, dict):
        return type(tensors)({k: distributed_concat(v) for k, v in tensors.items()})
    elif tensors is None:
        return None
    return tensors.cpu()


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    elif isinstance(tensors, dict):
        return type(tensors)({k: nested_truncate(v, limit) for k, v in tensors.items()})
    elif tensors is None:
        return None
    return tensors[:limit]


def _secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """

    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}/runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}/samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}/steps_per_second"] = round(steps_per_second, 3)
    return result


class Trainer:
    optimizer = None
    scheduler = None
    state = None

    def __init__(self,
                 args,
                 model,
                 compute_metrics,
                 train_dataset,
                 eval_dataset,
                 visualizer,
                 wandb_run_dir=None,
                 ):

        # force device and distributed setup init explicitly
        logging.info(f'Rank {args.local_rank} device = {args.device}')

        self.args = args
        self.output_interval = 50
        self.model = model
        self.compute_metrics = compute_metrics
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.visualizer = visualizer
        self.wandb_run_dir = wandb_run_dir

        # Build training state tracker.
        self.state = TrainerState()

        # CUDA and distributed training.
        self.model = self.model.to(args.device)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=self.args.ddp_find_unused_parameters,
        )

        if self.args.verbose and self.is_world_process_zero():
            print(self.model)

            # Setup output directory.
            if self.args.overwrite_output_dir:
                shutil.rmtree(self.args.output_dir)
            os.makedirs(self.args.output_dir, exist_ok=True)
        dist.barrier()

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        return get_scheduler(
            self.args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [],
                "weight_decay": 0.0,
            },
        ]
        for n, p in self.model.named_parameters():
            if 'gan_wrapper' in n:
                continue
            if n in decay_parameters and p.requires_grad:
                optimizer_grouped_parameters[0]["params"].append(p)
                if self.args.verbose and self.is_world_process_zero():
                    print('Trainable (w/ weight decay):', n)
            elif n not in decay_parameters and p.requires_grad:
                optimizer_grouped_parameters[1]["params"].append(p)
                if self.args.verbose and self.is_world_process_zero():
                    print('Trainable (w/o weight decay):', n)

        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=self.args.seed,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_sampler = ShardSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_processes=self.args.world_size,
            process_index=self.args.process_index,
        )

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """

        test_sampler = ShardSampler(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_processes=self.args.world_size,
            process_index=self.args.process_index,
        )

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if not self.is_world_process_zero():
            return

        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        # wandb
        wandb.log(logs)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
        return self.args.process_index == 0

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        torch.save(self.model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def save_state(self):
        """
        Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

        Under distributed environment this is done only for a process with rank 0.
        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, "trainer_state.json")
        self.state.save_to_json(path)

    def _sorted_checkpoints(
            self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
                self.state.best_model_checkpoint is not None
                and self.args.save_total_limit == 1
                and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint)

    def _save_checkpoint(self, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)

        # deepspeed.save_checkpoint above saves model/optim/sched
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        with warnings.catch_warnings(record=True) as caught_warnings:
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval/"):
                metric_to_check = f"eval/{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state(),
        }

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.local_rank}.pth"))

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _maybe_log_save_evaluate(self,
                                 weighted_loss,
                                 losses,
                                 epoch_end,
                                 ):
        args, state = self.args, self.state

        should_log, should_evaluate, should_save = False, False, False

        # Log?
        if state.global_step == 1 and args.logging_first_step:
            should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            should_log = True
        if args.logging_strategy == IntervalStrategy.EPOCH and epoch_end:
            should_log = True

        # Evaluate?
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step % args.eval_steps == 0:
            should_evaluate = True
            if args.load_best_model_at_end:
                should_save = True
        if args.evaluation_strategy == IntervalStrategy.EPOCH and epoch_end:
            should_evaluate = True

        # Save?
        if (
                args.save_strategy == IntervalStrategy.STEPS
                and args.save_steps > 0
                and state.global_step % args.save_steps == 0
        ):
            should_save = True
        if args.save_strategy == IntervalStrategy.EPOCH and epoch_end:
            should_save = True

        # Log.
        if should_log:
            logs = {
                name: loss.mean(0).item()
                for name, loss in losses.items()
            }

            logs["weighted_loss"] = weighted_loss.item()
            logs["learning_rate"] = self.scheduler.get_last_lr()[0]

            self.log(logs)

        # Evaluate.
        metrics = None
        if should_evaluate:
            metrics = self.evaluate()

        # Save.
        if should_save:
            self._save_checkpoint(metrics=metrics)

    def visualize(self, images, description):
        if not self.is_world_process_zero():
            return

        save_dir = self.args.output_dir
        self.visualizer.visualize(
            images=images,
            model=self.model.module,
            description=description,
            save_dir=save_dir,
            step=self.state.global_step,
        )

    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (:obj:`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            elif type(metrics_copy[k]) == float:
                metrics_copy[k] = round(v, 4)

        return metrics_copy

    def log_metrics(self, split, metrics):
        """
        Log metrics in a specially formatted way

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predictmetrics: metrics dict

        Notes on memory reports:

        In order to get memory usage report you need to install ``psutil``. You can do that with ``pip install psutil``.

        Now when this method is run, you will see a report that will include: ::

            init_mem_cpu_alloc_delta   =     1301MB
            init_mem_cpu_peaked_delta  =      154MB
            init_mem_gpu_alloc_delta   =      230MB
            init_mem_gpu_peaked_delta  =        0MB
            train_mem_cpu_alloc_delta  =     1345MB
            train_mem_cpu_peaked_delta =        0MB
            train_mem_gpu_alloc_delta  =      693MB
            train_mem_gpu_peaked_delta =        7MB

        **Understanding the reports:**

        - the first segment, e.g., ``train__``, tells you which stage the metrics are for. Reports starting with ``init_``
          will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
          ``__init__`` will be reported along with the ``eval_`` metrics.
        - the third segment, is either ``cpu`` or ``gpu``, tells you whether it's the general RAM or the gpu0 memory
          metric.
        - ``*_alloc_delta`` - is the difference in the used/allocated memory counter between the end and the start of the
          stage - it can be negative if a function released more memory than it allocated.
        - ``*_peaked_delta`` - is any extra memory that was consumed and then freed - relative to the current allocated
          memory counter - it is never negative. When you look at the metrics of any stage you add up ``alloc_delta`` +
          ``peaked_delta`` and you know how much memory was needed to complete that stage.

        The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
        main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
        use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
        memory than the rest since it stores the gradient and optimizer states for all participating GPUS. Perhaps in the
        future these reports will evolve to measure those too.

        The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
        memory shared with other processes. It is important to note that it does not include swapped out memory, so the
        reports could be imprecise.

        The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
        that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
        reality. Using ``tracemalloc`` would have reported the exact peak memory, but it doesn't report memory allocations
        outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
        was dropped in favor of the memory sampling approach, which reads the current process memory usage.

        The GPU allocated and peak memory reporting is done with ``torch.cuda.memory_allocated()`` and
        ``torch.cuda.max_memory_allocated()``. This metric reports only "deltas" for pytorch-specific allocations, as
        ``torch.cuda`` memory management system doesn't track any memory allocated outside of pytorch. For example, the
        very first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

        Note that this tracker doesn't account for memory allocations outside of :class:`~transformers.Trainer`'s
        ``__init__``, ``train``, ``evaluate`` and ``predict`` calls.

        Because ``evaluation`` calls may happen during ``train``, we can't handle nested invocations because
        ``torch.cuda.max_memory_allocated`` is a single counter, so if it gets reset by a nested eval call, ``train``'s
        tracker will report incorrect info. If this `pytorch issue <https://github.com/pytorch/pytorch/issues/16266>`__
        gets resolved it will be possible to change this class to be re-entrant. Until then we will only track the outer
        level of ``train``, ``evaluate`` and ``predict`` methods. Which means that if ``eval`` is called during ``train``,
        it's the latter that will account for its memory usage and that of the former.

        This also means that if any other tool that is used along the :class:`~transformers.Trainer` calls
        ``torch.cuda.reset_peak_memory_stats``, the gpu peak memory stats could be invalid. And the
        :class:`~transformers.Trainer` will disrupt the normal behavior of any such tools that rely on calling
        ``torch.cuda.reset_peak_memory_stats`` themselves.

        For best performance you may want to consider turning the memory profiling off for production runs.
        """
        if not self.is_world_process_zero():
            return

        print(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

    def save_metrics(self, split, metrics, combined=True):
        """
        Save metrics into a json file for that split, e.g. ``train_results.json``.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``, ``all``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Creates combined metrics by updating ``all_results.json`` with metrics of this call

        To understand the metrics please read the docstring of :meth:`~transformers.Trainer.log_metrics`. The only
        difference is that raw unformatted numbers are saved in the current method.

        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)

    def _prepare_inputs(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

    def training_step(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        # Training mode.
        self.model.train()

        # CUDA.
        self._prepare_inputs(inputs)

        # Forward.
        images, weighted_loss, losses = self.model(**inputs)

        weighted_loss = weighted_loss.mean(0)

        if self.args.gradient_accumulation_steps > 1:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            weighted_loss = weighted_loss / self.args.gradient_accumulation_steps

        # Backward.
        weighted_loss.backward()

        return images, weighted_loss, losses

    def prediction_step(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        """
        self._prepare_inputs(inputs)

        with torch.no_grad():
            images, weighted_loss, losses = self.model(**inputs)

        return images, weighted_loss, losses

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            metric_key_prefix: str = "eval",
    ) -> Tuple[Dict[str, float], int]:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {len(dataloader.dataset)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        self.model.eval()

        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        prediction_outputs_host = None
        # losses/preds/labels on CPU (final containers)
        all_prediction_outputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            # Prediction step
            prediction_outputs = self.prediction_step(inputs)

            # Update containers on host
            if prediction_outputs is not None:
                prediction_outputs = distributed_concat(prediction_outputs)
            prediction_outputs_host = (
                prediction_outputs if prediction_outputs_host is None else
                nested_concat(prediction_outputs_host, prediction_outputs, padding_index=-100)
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if prediction_outputs_host is not None:
                    prediction_outputs = nested_cpu(prediction_outputs_host)
                    all_prediction_outputs = (
                        prediction_outputs if all_prediction_outputs is None else
                        nested_concat(all_prediction_outputs, prediction_outputs, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                prediction_outputs_host = None

        # Gather all remaining tensors and put them back on the CPU
        if prediction_outputs_host is not None:
            prediction_outputs = nested_cpu(prediction_outputs_host)
            all_prediction_outputs = (
                prediction_outputs if all_prediction_outputs is None else
                nested_concat(all_prediction_outputs, prediction_outputs, padding_index=-100)
            )

        # Number of samples
        num_samples = len(eval_dataset)

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_prediction_outputs is not None:
            all_prediction_outputs = nested_truncate(all_prediction_outputs, num_samples)

        images, weighted_loss, losses = all_prediction_outputs

        # Metrics!
        if self.is_world_process_zero():
            if self.compute_metrics and all_prediction_outputs:
                metrics = self.compute_metrics(images,
                                               self.model.module,
                                               weighted_loss,
                                               losses,
                                               dataset=eval_dataset,
                                               split=metric_key_prefix,
                                               )
            else:
                metrics = {}

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            # Prefix all keys with metric_key_prefix + '/'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}/"):
                    metrics[f"{metric_key_prefix}/{key}"] = metrics.pop(key)

            # Weighted loss and losses
            metrics[f"{metric_key_prefix}/weighted_loss"] = weighted_loss.mean(0).item()
            for key, value in losses.items():
                metrics[f"{metric_key_prefix}/{key}"] = value.mean(0).item()

            # Save images.
            self.visualize(images, description)
        else:
            metrics = None

        return metrics, num_samples

    def train(self):
        args = self.args

        # Build train dataloader.
        print('In train')
        train_dataloader = self.get_train_dataloader()
        print('Train loader successfully built')
        # Set up training control variables.
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        assert isinstance(self.train_dataset, collections.abc.Sized)
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = len(self.train_dataset) * args.num_train_epochs

        # Build optimizer and scheduler.
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler(num_training_steps=max_steps)

        # Train!
        num_examples = len(self.train_dataset)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.model.zero_grad()

        for epoch in range(num_train_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            steps_in_epoch = len(train_dataloader)

            for step, inputs in tqdm(enumerate(train_dataloader)):
                _, weighted_loss, losses = self.training_step(inputs)

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        (step + 1) == steps_in_epoch <= args.gradient_accumulation_steps
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(self.model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            self.model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()

                    self.model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    epoch_end = (step + 1) == steps_in_epoch
                    self._maybe_log_save_evaluate(weighted_loss, losses, epoch_end)

        # Finished training.
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        self.log(metrics)

        return metrics

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        metrics, num_samples = eval_loop(
            eval_dataloader,
            description="eval",
            metric_key_prefix=metric_key_prefix,
        )

        if self.is_world_process_zero():
            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=num_samples,
                    num_steps=math.ceil(num_samples / total_batch_size),
                )
            )

            self.log(metrics)

        return metrics

    def predict(
            self,
            test_dataset: Dataset,
            metric_key_prefix: str = "test",
    ) -> Dict[str, float]:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        metrics, num_samples = eval_loop(
            test_dataloader,
            description="test",
            metric_key_prefix=metric_key_prefix,
        )

        if self.is_world_process_zero():
            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=num_samples,
                    num_steps=math.ceil(num_samples / total_batch_size),
                )
            )

            self.log(metrics)

        return metrics
