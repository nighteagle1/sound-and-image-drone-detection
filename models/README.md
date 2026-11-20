# Модели

Оба файла лежат в репозитории; пути задаются в `config.yaml`.

## `drone_audio.onnx` — звук

- Источник: [AntoineNaccache/drone-audio-detector](https://huggingface.co/AntoineNaccache/drone-audio-detector),
  чекпойнт `drone_classifier_aug_mixed.pt`. Лицензия MIT.
- Получен конвертацией на Kaggle скриптом `tools/kaggle_export/export_audio.py`
  (веса загружены со `strict=True`, 1 486 113 параметров; FP32, opset 17).
  Вероятность ONNX совпала с PyTorch (допуск 1e-4) на тестовом клипе.
- Вход: `log_mel`, форма `(1, 1, 64, T)` float32, ось `T` динамическая (для 1 с `T = 101`).
- Выход: `logit`, форма `(1, 1)`; вероятность = sigmoid(logit).
- Признаки: моно, 16 кГц, Hann 512, hop 160, 64 mel, 50–5500 Гц, `(AmplitudeToDB + 40) / 40`.
- Эталон для тестов: `tests/data/audio_golden.npz` (`wave`, `log_mel`, `prob`).

## `drone_yolov5.onnx` — изображение

- Источник: [engdarwish/drone-detection-yolov5](https://huggingface.co/engdarwish/drone-detection-yolov5),
  файл `best.onnx`. Лицензия MIT.
- Вход: `images`, форма `(1, 3, 416, 416)` float32, RGB, значения 0..1, NCHW (размер фиксированный).
- Выход: `output0`, форма `(1, 10647, 6)` = `(1, N, 5 + nc)`: `cx, cy, w, h, obj, cls0`
  (центр и размеры в пикселях входа 416×416). Число классов `nc = 1` (дрон).
- Оценка = `obj * cls0`.

Заявленные авторами метрики (F1 0.9996 для звука, mAP50 99% для изображения) получены на их данных
и на показе не гарантируются.
