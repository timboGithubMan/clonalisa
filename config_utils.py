import json
import re
import os
from pathlib import Path
from datetime import datetime

CONFIG_PATH = Path(__file__).resolve().parent / 'clonalisa_config.json'

DEFAULT_CONFIG = {
    "regex": {
        "file": r"(?P<well>[A-Za-z]+\d+)_pos(?P<position>\d+)(?:_(?P<channel>[^_]+))?(?:_Z(?P<z_index>\d+))?(?:_s(?P<step>\d+))?",
        "time_from_folder": r".*_(?P<date>\d{8})_(?P<time>\d{6})$",
    },
    "time_source": "folder",
    "model_history": [
        {
            "path": str(Path('omnipose_models/10x_NPC_nclasses_2_nchan_3_dim_2_2024_03_29_02_03_10.875324_epoch_960')),
            "flow_threshold": 0.4,
            "mask_threshold": 0.0,
            "z_indices": [1, 2, 0],
        }
    ],
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)


def parse_filename(filename, cfg=None):
    if cfg is None:
        cfg = load_config()
    pattern = cfg.get('regex', {}).get('file', DEFAULT_CONFIG['regex']['file'])
    m = re.match(pattern, Path(filename).stem)
    if not m:
        return None, None, None, None, None
    groups = m.groupdict()
    well = groups.get('well')
    position = groups.get('position')
    channel = groups.get('channel')
    z_index = groups.get('z_index')
    step = groups.get('step')
    if z_index is not None:
        try:
            z_index = int(z_index)
        except ValueError:
            z_index = None
    return well, position, channel, z_index, step


def extract_time_from_folder(folder_name, cfg=None):
    if cfg is None:
        cfg = load_config()
    pattern = cfg.get('regex', {}).get('time_from_folder', DEFAULT_CONFIG['regex']['time_from_folder'])
    m = re.match(pattern, folder_name)
    if not m:
        return None
    gd = m.groupdict()
    if 'date' in gd and 'time' in gd:
        dt_str = f"{gd['date']} {gd['time']}"
        return datetime.strptime(dt_str, '%Y%m%d %H%M%S')
    return None


def get_image_time(image_path, cfg=None):
    if cfg is None:
        cfg = load_config()
    source = cfg.get('time_source', 'folder')
    if source == 'date_created':
        try:
            ts = Path(image_path).stat().st_birthtime
            return datetime.fromtimestamp(ts)
        except Exception:
            return None
    # fallback to folder name
    return extract_time_from_folder(Path(image_path).parent.name, cfg)
