import os
import json
import pathlib
from random import shuffle

from rvc.configs.config import Config

config = Config()
current_directory = os.getcwd()


def generate_config(rvc_version, sampling_rate, model_path):
    if rvc_version == "v1" or sampling_rate == "40000":
        config_path = f"v1/{sampling_rate}.json"
    else:
        config_path = f"v2/{sampling_rate}.json"
    config_save_path = os.path.join(model_path, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")


def generate_filelist(f0_method, model_path, rvc_version, sampling_rate):
    gt_wavs_dir = f"{model_path}/0_gt_wavs"
    feature_dir = (
        f"{model_path}/3_feature256"
        if rvc_version == "v1"
        else f"{model_path}/3_feature768"
    )
    if f0_method:
        f0_dir = f"{model_path}/2a_f0"
        f0nsf_dir = f"{model_path}/2b-f0nsf"
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    options = []
    for name in names:
        if f0_method:
            options.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
            )
        else:
            options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")
    fea_dim = 256 if rvc_version == "v1" else 768
    if f0_method:
        for _ in range(2):
            options.append(
                f"{current_directory}/logs/mute/0_gt_wavs/mute{sampling_rate}.wav|{current_directory}/logs/mute/3_feature{fea_dim}/mute.npy|{current_directory}/logs/mute/2a_f0/mute.wav.npy|{current_directory}/logs/mute/2b-f0nsf/mute.wav.npy|0"
            )
    else:
        for _ in range(2):
            options.append(
                f"{current_directory}/logs/mute/0_gt_wavs/mute{sampling_rate}.wav|{current_directory}/logs/mute/3_feature{fea_dim}/mute.npy|0"
            )
    shuffle(options)
    with open(f"{model_path}/filelist.txt", "w") as f:
        f.write("\n".join(options))
