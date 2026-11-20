"""Экспорт звуковой модели drone_classifier_aug_mixed.pt в ONNX (запускается на Kaggle).

Результаты кладутся в /kaggle/working: drone_audio.onnx, audio_golden.npz.
"""
import io
import sys
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out")
OUT.mkdir(parents=True, exist_ok=True)

import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "huggingface_hub", "soundfile", "pyarrow"])

import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.transforms as T
from huggingface_hub import hf_hub_download

REPO = "AntoineNaccache/drone-audio-detector"
sys.path.insert(0, str(Path(hf_hub_download(REPO, "model.py")).parent))
from model import DroneClassifier  # noqa: E402

ckpt = hf_hub_download(REPO, "drone_classifier_aug_mixed.pt")
state = torch.load(ckpt, map_location="cpu", weights_only=False)
if "model_state_dict" in state:
    state = state["model_state_dict"]

model = DroneClassifier()
model_keys = set(model.state_dict().keys())
extra = sorted(set(state) - model_keys)
print("Лишние ключи чекпойнта (игнорируются):", extra)
# strict=True: ни один ключ модели не должен отсутствовать или не совпасть по форме.
# Лишние ключи (старый joint-чекпойнт с separator) отбрасываем явно.
state = {k: v for k, v in state.items() if k in model_keys}
model.load_state_dict(state, strict=True)
model.eval()
print("Веса загружены со strict=True, параметров:", sum(p.numel() for p in model.parameters()))


def load_test_wave() -> np.ndarray:
    try:
        import pyarrow.parquet as pq

        p = hf_hub_download("geronimobasso/drone-audio-detection-samples",
                            "data/train-00004-of-00039.parquet", repo_type="dataset")
        tbl = pq.read_table(p).slice(0, 2000).to_pylist()
        print("Колонки датасета:", list(tbl[0].keys()))
        lab_key = next(k for k in tbl[0] if k != "audio")
        for row in tbl:
            if int(row[lab_key]) == 1:
                a = row["audio"]
                data, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(1)
                assert sr == 16000, sr
                if len(data) >= 16000 and np.abs(data[:16000]).max() > 1e-3:
                    print("Тестовый сигнал: клип из датасета, метка дрона")
                    return data[:16000].astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print("Датасет недоступен, синтез сигнала:", repr(e))
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    w = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 12)) * 0.2
    return (w + rng.normal(0, 0.02, 16000)).astype(np.float32)


wave = load_test_wave()
mel = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=64, f_min=50, f_max=5500)
log_mel = (T.AmplitudeToDB()(mel(torch.from_numpy(wave)[None])) + 40) / 40
x = log_mel.unsqueeze(0).float()  # (1,1,64,101)
assert tuple(x.shape) == (1, 1, 64, 101), x.shape
with torch.no_grad():
    prob = torch.sigmoid(model(x)).item()
print("PyTorch prob:", prob)

onnx_path = OUT / "drone_audio.onnx"
torch.onnx.export(
    model, x, str(onnx_path), opset_version=17,
    input_names=["log_mel"], output_names=["logit"],
    dynamic_axes={"log_mel": {3: "T"}}, dynamo=False,
)
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
logit = sess.run(None, {"log_mel": x.numpy()})[0]
onnx_prob = float(1 / (1 + np.exp(-logit.reshape(-1)[0])))
print("ONNX prob:", onnx_prob)
assert abs(onnx_prob - prob) < 1e-4, (onnx_prob, prob)
# динамическая ось: другая длина тоже работает
sess.run(None, {"log_mel": np.zeros((1, 1, 64, 51), np.float32)})

np.savez(OUT / "audio_golden.npz", wave=wave.astype(np.float32),
         log_mel=x.numpy().astype(np.float32), prob=np.float32(prob))
print("OK")
