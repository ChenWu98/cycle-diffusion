# Created by Chen Henry Wu
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):

    cfg: str = None

    verbose: bool = field(
        default=False,
    )



