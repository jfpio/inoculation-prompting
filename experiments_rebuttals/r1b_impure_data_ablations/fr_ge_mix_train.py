"""
Experiment: Does inoculation outperform doing nothing if the classifier is somewhat noisy? 

Minimal experiment: Inoculate the full mixture of French and German.
"""

import asyncio
from pathlib import Path

from fr_ge_configs import build_datasets, list_configs

from ip.experiments import train_main
from ip.experiments.utils import setup_experiment_dirs

async def main():
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "training_data"
    build_datasets(data_dir)
    configs = list_configs(data_dir, groups=["no-inoc", "french-inoc", "fully-spanish-inoc"])
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())