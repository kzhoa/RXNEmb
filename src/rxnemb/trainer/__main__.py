import argparse
import os
from pathlib import Path

from .classifier import ClassifierTrainer
from .utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--local-rank", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config_json)
    if str(config.task).lower() != "classification":
        raise ValueError("RXNEmb trainer only supports classification.")

    if args.local_rank is not None:
        config.others.local_rank = args.local_rank
    elif "LOCAL_RANK" in os.environ:
        config.others.local_rank = int(os.environ["LOCAL_RANK"])

    trainer = ClassifierTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
