# Created by Chen Henry Wu
import importlib


def get_model(model):
    return importlib.import_module('model.{}'.format(model)).Model


def get_preprocessor(preprocess_program):
    return importlib.import_module('preprocess.{}'.format(preprocess_program)).Preprocessor


def get_evaluator(evaluator_program):
    return importlib.import_module('evaluation.{}'.format(evaluator_program)).Evaluator


def get_visualizer(visualizer_program):
    return importlib.import_module('visualization.{}'.format(visualizer_program)).Visualizer