from pathlib import Path
import os
import yaml


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get(
            "CRYPTOML_CONFIG_PATH",
            "src/main/resources/application.yaml",
        )
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
