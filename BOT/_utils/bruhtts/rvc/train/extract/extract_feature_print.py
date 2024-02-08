import os
import sys
import tqdm
import torch
import torch.nn.functional as F
import fairseq
import soundfile as sf
import numpy as np


device = sys.argv[1]
n_parts = int(sys.argv[2])
i_part = int(sys.argv[3])

if len(sys.argv) == 7:
    exp_dir, version, is_half = sys.argv[4], sys.argv[5], sys.argv[6]
else:
    i_gpu, exp_dir = sys.argv[4], sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version, is_half = sys.argv[6], sys.argv[7]


def forward_dml(ctx, x, scale):
    ctx.scale = scale
    res = x.clone().detach()
    return res


fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

model_path = "hubert_base.pt"

wav_path = f"{exp_dir}/1_16k_wavs"
out_path = f"{exp_dir}/3_feature256" if version == "v1" else f"{exp_dir}/3_feature768"
os.makedirs(out_path, exist_ok=True)


def read_wave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav)
    feats = feats.half() if is_half else feats.float()
    feats = feats.mean(-1) if feats.dim() == 2 else feats
    feats = feats.view(1, -1)
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats


print("Starting feature extraction...")
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
if device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(os.listdir(wav_path))[i_part::n_parts]
n = max(1, len(todo) // 10)

if len(todo) == 0:
    print(
        "An error occurred in the feature extraction, make sure you have provided the audios correctly."
    )
else:
    print(f"{len(todo)}")
    with tqdm.tqdm(total=len(todo)) as pbar:
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_file_path = os.path.join(wav_path, file)
                    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))

                    if os.path.exists(out_file_path):
                        continue

                    feats = read_wave(wav_file_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0])
                            if version == "v1"
                            else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_file_path, feats, allow_pickle=False)
                    else:
                        print(f"{file} - contains nan")
                    pbar.set_description(f"Processing {file} {feats.shape}")
            except Exception as error:
                print(error)
            pbar.update(1)

    print("Feature extraction completed successfully!")
