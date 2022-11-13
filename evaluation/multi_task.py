import os
import numpy as np
import torch
from utils.program_utils import get_evaluator
from utils.config_utils import get_config


class Evaluator(object):
    def __init__(self, meta_args):
        self.meta_args = meta_args

    def evaluate(self, images, model, weighted_loss, losses, dataset, split):
        assert split in ['eval', 'test']
        assert len(weighted_loss) == len(dataset) == len(dataset.data)
        num_examples = len(dataset)
        assert all(len(v) == num_examples for k, v in losses.items())
        if isinstance(images, torch.Tensor):
            assert images.shape[0] == num_examples
        elif isinstance(images, (list, tuple)):
            assert (
                    all(_images.shape[0] == num_examples for _images in images)
                    or all(_images is None for _images in images)
            )
        elif images is None:
            pass
        else:
            raise TypeError()

        # Gather evaluation data for each task.
        name2eval_kwargs = dict()
        for i in range(num_examples):
            name = dataset.data[i]['name']
            if name not in name2eval_kwargs:
                name2eval_kwargs[name] = {
                    "images": [],
                    "model": model,
                    "weighted_loss": [],
                    "losses": {
                        k: [] for k in losses.keys()
                    },
                    "data": [],
                }
            if isinstance(images, torch.Tensor):
                name2eval_kwargs[name]['images'].append(images[i])
            elif isinstance(images, (list, tuple)):
                name2eval_kwargs[name]['images'].append(
                    tuple(_images[i] if _images is not None else None for _images in images)
                )
            elif images is None:
                name2eval_kwargs[name]['images'].append(None)
            else:
                raise TypeError()

            name2eval_kwargs[name]['weighted_loss'].append(weighted_loss[i])
            for k, v in losses.items():
                name2eval_kwargs[name]['losses'][k].append(v[i])
            name2eval_kwargs[name]['data'].append(dataset.data[i])

        # Evaluate each task.
        summary = dict()
        for name, eval_kwargs in name2eval_kwargs.items():
            arg_path = getattr(self.meta_args.arg_paths, name)
            args = get_config(arg_path)
            evaluator = get_evaluator(args.evaluation.evaluator_program)(args, self.meta_args)
            summary_tmp = evaluator.evaluate(**eval_kwargs, split=split)
            for key, metric in summary_tmp.items():
                summary[f'{name}/{key}'] = metric

        if len(summary) > 0:
            summary['avr'] = float(np.mean([float(v) for k, v in summary.items()]))
        return summary
