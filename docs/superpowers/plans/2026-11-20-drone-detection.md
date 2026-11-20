# Обнаружение дронов по звуку и изображению — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Веб-приложение на Python, которое в браузере обнаруживает дронов по звуку (микрофон или файл) и по изображению (камера или видеофайл) двумя независимыми режимами; запуск одним `run.sh` на Raspberry Pi и `run.bat` на Windows.

**Architecture:** FastAPI-сервер сам захватывает микрофон (sounddevice) и камеру (OpenCV) и гоняет кадры и звук через ONNX Runtime. Браузер получает видео как MJPEG и результаты как SSE. Одновременно работает один режим, модель другого не загружена. Звуковые признаки считаются на NumPy.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, onnxruntime, numpy, scipy, opencv-python-headless, sounddevice, soundfile, pyyaml, pytest, httpx (для TestClient). Ванильный JS без сборки.

**Spec:** `docs/superpowers/specs/2026-11-20-drone-detection-design.md`

## Global Constraints

- Python 3.11+; платформы: Raspberry Pi 4/5 с 64-bit Bookworm и Windows 10/11.
- На устройстве только ONNX Runtime: PyTorch и torchaudio не устанавливаются (они нужны только в Kaggle-скрипте экспорта).
- Модели лежат в репозитории: `models/drone_audio.onnx`, `models/drone_yolov5.onnx`; пути задаются в `config.yaml`.
- Звук: моно, 16 кГц, окно 1 с, шаг 0,5 с (перекрытие 50%), Hann-окно 512, hop 160, 64 mel, 50–5500 Гц, нормировка `(AmplitudeToDB + 40) / 40`, выход модели — логит, вероятность = sigmoid.
- Индикатор звука: включение после двух окон подряд выше порога, выключение после двух ниже; порог по умолчанию 0.5. На графике сырая вероятность за 30 с.
- Изображение: один порог уверенности (ползунок), NMS IoU 0.45.
- Сервер слушает `127.0.0.1`; `--host 0.0.0.0` включает доступ из сети с предупреждением.
- `run.sh` при нехватке системных пакетов выводит команду `sudo apt install ...` и сам `sudo` не запускает.
- Одновременно работает один режим; смена вкладки = Стоп текущего режима и выгрузка модели.
- Интерфейс и сообщения об ошибках на русском.
- Коммиты: сообщение на русском, без упоминания Claude и без строк Co-Authored-By; `git push` не делать.

## Review Focus

Входы, которых спецификация не описывает, но которые встретятся на показе:

1. Отсутствует файл модели: `/api/status` сообщает недоступность режима, «Старт» возвращает понятную ошибку, сервер не падает (Task 8).
2. Нет микрофона или камеры, либо камера занята: `SourceError` с русским текстом, показывается на странице (Tasks 5, 7, 8).
3. Тишина (нули) и аудиофайл короче 1 с: признаки конечны (без NaN), короткий файл даёт ровно один дополненный нулями фрагмент (Tasks 3, 5).
4. Битый, пустой или не тот формат загруженного файла, а также слишком большой файл: ошибка при «Старте», не падение потока (Tasks 5, 7, 8).
5. Повторный «Старт» при работе, «Стоп» без запуска, «Старт» другого режима поверх работающего: идемпотентно или 409 с текстом (Task 8).

## Структура файлов

```
config.yaml                       пути моделей, порог по умолчанию, порт
requirements.txt, requirements-dev.txt
run.sh, run.bat
models/drone_audio.onnx, models/drone_yolov5.onnx, models/README.md
tools/kaggle_export/{export_audio.py,kernel-metadata.json}
src/dronedetect/
  __init__.py, __main__.py        CLI: --host, --port, --no-browser
  errors.py                       DroneError, ModelNotFoundError, SourceError
  config.py                       загрузка config.yaml
  audio/{features.py,detector.py,sources.py}
  vision/{detector.py,sources.py}
  session.py                      DetectionSession, EventHub
  server.py                       FastAPI app
  static/{index.html,app.js,style.css}
tests/ (+ tests/data/audio_golden.npz)
```

---

### Task 1: Каркас проекта, конфиг, ошибки

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `config.yaml`, `.gitignore`, `src/dronedetect/{__init__,errors,config}.py`, `tests/test_config.py`, `pytest.ini` (`pythonpath = src`)

**Interfaces:**
- Produces: `errors.DroneError(message: str)`, `errors.ModelNotFoundError(DroneError)`, `errors.SourceError(DroneError)` — у всех атрибут `.message` (русский текст для UI);
  `config.AppConfig` (dataclass: `audio_model: Path`, `vision_model: Path`, `audio_threshold: float = 0.5`, `vision_threshold: float = 0.25`, `port: int = 8000`);
  `config.load_config(path: Path | None = None) -> AppConfig` — пути в `config.yaml` относительны корню репозитория.

- [ ] **Step 1: Тест** `test_load_config_resolves_relative_paths` (пути становятся абсолютными относительно каталога файла конфига), `test_defaults_when_keys_missing` (`audio_threshold == 0.5`, `port == 8000`).
- [ ] **Step 2:** `pytest tests/test_config.py -v` — FAIL (нет модуля).
- [ ] **Step 3:** Реализовать `errors.py`, `config.py`; в `requirements.txt` прописать пакеты из Tech Stack (без pytest/httpx), в `requirements-dev.txt` — `-r requirements.txt`, `pytest`, `httpx`; в `.gitignore` — `.venv/`, `__pycache__/`, `uploads/`.
- [ ] **Step 4:** `pytest -v` — PASS.
- [ ] **Step 5:** Коммит: `git commit -m "Каркас проекта: конфиг и типы ошибок"`.

---

### Task 2: Модели — экспорт звуковой в ONNX на Kaggle и загрузка YOLO (точка риска, делается первой)

**Files:**
- Create: `tools/kaggle_export/export_audio.py`, `tools/kaggle_export/kernel-metadata.json`, `models/drone_audio.onnx`, `models/drone_yolov5.onnx`, `models/README.md`, `tests/data/audio_golden.npz`

**Interfaces:**
- Produces: `models/drone_audio.onnx` — вход `(1,1,64,T)` float32 с динамической осью `T`, выход `(1,1)` логит; `tests/data/audio_golden.npz` с ключами `wave` `(16000,) float32` (реальная запись дрона или фрагмент, не нули), `log_mel` `(1,1,64,101) float32` (из torchaudio по формуле спецификации), `prob` `float32` (sigmoid логита PyTorch); `models/README.md` — фактические имя входа, размер входа и формат выхода YOLO (`(1,N,5+nc)`), число классов, порядок координат.

- [ ] **Step 1:** Написать `export_audio.py` для Kaggle: скачать `model.py`, `config.yaml`, `drone_classifier_aug_mixed.pt` из `AntoineNaccache/drone-audio-detector` (huggingface_hub), загрузить веса по README модели (ключ `model_state_dict`, если есть) **с `strict=True`**, чтобы несовпадение ключей было ошибкой; взять тестовый сигнал (синтез шума или клип из датасета `geronimobasso/drone-audio-detection-samples`), посчитать torchaudio `log_mel` и `prob`, экспортировать `torch.onnx.export` (opset 17, динамическая ось T), прогнать onnxruntime и `assert abs(onnx_prob - prob) < 1e-4`, сохранить `drone_audio.onnx` и `audio_golden.npz`.
- [ ] **Step 2:** `kernel-metadata.json` (internet включён, GPU не нужен, `code_file` = скрипт, id = `<kaggle username из kaggle config view>/drone-audio-onnx-export`). Запуск: `kaggle kernels push -p tools/kaggle_export`; ожидание: `kaggle kernels status <id>` до `complete`; результат: `kaggle kernels output <id> -p tools/kaggle_export/out`. Если у аккаунта нет интернета в ядрах — загрузить три файла как приватный датасет (`kaggle datasets create`) и читать их оттуда.
- [ ] **Step 3:** Скопировать результаты в `models/` и `tests/data/`; `strict=True` упал или параметры не сошлись — остановиться и сообщить пользователю (по спецификации тогда выпускаем сначала режим «Изображение»).
- [ ] **Step 4:** Скачать `https://huggingface.co/engdarwish/drone-detection-yolov5/resolve/main/best.onnx` в `models/drone_yolov5.onnx`; через onnxruntime распечатать входы и выходы и записать их в `models/README.md` вместе с источниками и лицензиями MIT.
- [ ] **Step 5:** Проверка: оба файла открываются `onnxruntime.InferenceSession(path)`; `audio_golden.npz` содержит все три ключа.
- [ ] **Step 6:** Коммит: `git commit -m "Добавлены ONNX-модели и скрипт экспорта звуковой модели на Kaggle"`.

---

### Task 3: Звуковые признаки на NumPy

**Files:**
- Create: `src/dronedetect/audio/__init__.py`, `src/dronedetect/audio/features.py`, `tests/test_features.py`

**Interfaces:**
- Produces: `features.log_mel(wave: np.ndarray) -> np.ndarray` — вход float32 моно 16 кГц, любая длина ≥ 1; выход float32 `(1,1,64,T)`, для 16000 отсчётов `T == 101`.

- [ ] **Step 1: Тесты:** `test_shape_for_one_second` (`(1,1,64,101)`); `test_silence_is_finite_and_floor` (нули → все значения конечны и равны `-1.5`, т. е. `(10*log10(1e-10)+40)/40`); `test_matches_torchaudio_golden` (из `tests/data/audio_golden.npz`: `np.allclose(log_mel(wave), golden["log_mel"], atol=1e-3, rtol=1e-3)`).
- [ ] **Step 2:** `pytest tests/test_features.py -v` — FAIL.
- [ ] **Step 3:** Реализовать как `torchaudio.transforms.MelSpectrogram` по умолчанию: `center=True`, отражающее дополнение, периодическое Hann-окно 512, power=2, HTK-шкала, без нормировки фильтров, `f_min=50`, `f_max=5500`; затем `10*log10(max(x, 1e-10))`, `(db + 40) / 40`. Банк фильтров считается один раз при импорте.
- [ ] **Step 4:** `pytest tests/test_features.py -v` — PASS (при расхождении с golden сначала сверить дополнение и окно).
- [ ] **Step 5:** Коммит: `git commit -m "Звуковые признаки (log-mel) на NumPy, сверенные с torchaudio"`.

---

### Task 4: Звуковой детектор и сглаживание

**Files:**
- Create: `src/dronedetect/audio/detector.py`, `tests/test_audio_detector.py`

**Interfaces:**
- Consumes: `features.log_mel`, `errors.ModelNotFoundError`, `config.AppConfig`.
- Produces: `AudioDetector(model_path: Path)` — бросает `ModelNotFoundError("Файл звуковой модели не найден: ...")`; `.predict(chunk: np.ndarray) -> float` (chunk — ровно 16000 float32, иначе `ValueError`; результат — sigmoid, 0..1); `.close()`;
  `DetectionSmoother(on_after: int = 2, off_after: int = 2)` с `.update(prob: float, threshold: float) -> bool` и `.reset()`.

- [ ] **Step 1: Тесты:** `test_predict_matches_pytorch_golden` (`abs(det.predict(golden["wave"]) - golden["prob"]) < 1e-3`); `test_predict_silence_in_range` (нули → число в [0,1], без исключения); `test_predict_wrong_length_raises`; `test_missing_model_raises` (`ModelNotFoundError`); `test_smoother_turns_on_after_two` (0.9 → False, 0.9 → True); `test_smoother_turns_off_after_two` (после включения 0.1 → True, 0.1 → False); `test_smoother_single_dip_ignored`.
- [ ] **Step 2:** `pytest tests/test_audio_detector.py -v` — FAIL.
- [ ] **Step 3:** Реализовать; `InferenceSession` с `CPUExecutionProvider`, имя входа брать из `get_inputs()[0].name`, один поток обработки.
- [ ] **Step 4:** `pytest tests/test_audio_detector.py -v` — PASS.
- [ ] **Step 5:** Коммит: `git commit -m "Звуковой детектор на ONNX Runtime и сглаживание индикатора"`.

---

### Task 5: Источники звука (файл, микрофон)

**Files:**
- Create: `src/dronedetect/audio/sources.py`, `tests/test_audio_sources.py`

**Interfaces:**
- Consumes: `errors.SourceError`.
- Produces: `Chunker(size: int = 16000, hop: int = 8000)` с `.push(samples: np.ndarray) -> list[np.ndarray]`;
  `AudioSource` (протокол): `.chunks() -> Iterator[np.ndarray]` (каждый элемент float32, 16000 отсчётов) и `.close()`;
  `FileAudioSource(path: Path, realtime: bool = True)` — открывается и декодируется в конструкторе (`SourceError("Не удалось прочитать аудиофайл")` при пустом или битом файле); приводит к моно и 16 кГц (`scipy.signal.resample_poly`); файл короче 1 с даёт один фрагмент, дополненный нулями; при `realtime=True` ждёт 0,5 с между фрагментами;
  `MicAudioSource(device_id: int | None)` — `SourceError("Микрофон не найден")`, если нет устройств ввода; `list_microphones() -> list[Device]`, где `Device(id: int, name: str)` — dataclass, определённый в `audio/sources.py` и импортируемый из `vision/sources.py`.

- [ ] **Step 1: Тесты:** `test_chunker_overlap` (push 16000 → 1 фрагмент, затем 8000 → 1, затем 4000 → 0); `test_file_source_stereo_44k_yields_mono_16k_chunks` (WAV, 2 с, 2 канала, 44100 Гц, `realtime=False` → фрагменты по 16000, 1-D); `test_short_file_single_padded_chunk` (0,3 с → ровно 1 фрагмент, длина 16000); `test_empty_file_raises`, `test_garbage_bytes_raises` (`SourceError`); `test_mic_no_devices_raises` (monkeypatch `sounddevice.query_devices` → пустой список).
- [ ] **Step 2:** `pytest tests/test_audio_sources.py -v` — FAIL.
- [ ] **Step 3:** Реализовать: файл читать через `soundfile`; микрофон — `sounddevice.InputStream(samplerate=16000, channels=1, dtype="float32")` с очередью, `chunks()` блокируется на очереди и завершается после `close()`.
- [ ] **Step 4:** `pytest tests/test_audio_sources.py -v` — PASS.
- [ ] **Step 5:** Коммит: `git commit -m "Источники звука: файл и микрофон"`.

---

### Task 6: Детектор изображения (YOLOv5 ONNX)

**Files:**
- Create: `src/dronedetect/vision/__init__.py`, `src/dronedetect/vision/detector.py`, `tests/test_vision_detector.py`

**Interfaces:**
- Consumes: `models/README.md` (вход и формат выхода), `errors.ModelNotFoundError`.
- Produces: `Detection(x1: float, y1: float, x2: float, y2: float, score: float)` (dataclass, координаты в пикселях исходного кадра);
  `letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]` (картинка `size×size`, коэффициент масштаба, отступы `(dw, dh)`);
  `postprocess(output: np.ndarray, ratio: float, pad: tuple[float, float], conf: float, iou: float = 0.45) -> list[Detection]` — `output` `(1,N,5+nc)`, оценка = `obj * max(cls)`;
  `YoloDetector(model_path: Path)` с `.detect(frame_bgr: np.ndarray, conf: float) -> list[Detection]` и `.draw(frame_bgr, detections) -> np.ndarray`.

- [ ] **Step 1: Тесты:** `test_letterbox_keeps_aspect` (480×640 → `(640,640,3)`, ratio 1.0, pad `(0.0, 80.0)`); `test_postprocess_filters_by_conf` (синтетический выход с боксами оценки 0.9 и 0.1 при `conf=0.25` → 1 бокс); `test_postprocess_nms_merges_overlap` (два почти одинаковых бокса → 1); `test_postprocess_maps_back_to_original` (бокс по центру letterbox-кадра ↔ корректные пиксели исходного кадра с учётом pad и ratio); `test_detect_on_blank_frame_returns_list` (реальная модель, чёрный кадр 480×640 → `list`); `test_missing_model_raises`.
- [ ] **Step 2:** `pytest tests/test_vision_detector.py -v` — FAIL.
- [ ] **Step 3:** Реализовать; размер входа брать из формы входа сессии (если динамический — 640); BGR→RGB, `/255`, NCHW float32; NMS на NumPy; если число классов > 1 и в README указан класс дрона — учитывать только его.
- [ ] **Step 4:** `pytest tests/test_vision_detector.py -v` — PASS.
- [ ] **Step 5:** Коммит: `git commit -m "Детектор изображения на ONNX Runtime (YOLOv5)"`.

---

### Task 7: Источники видео (камера, файл)

**Files:**
- Create: `src/dronedetect/vision/sources.py`, `tests/test_video_sources.py`

**Interfaces:**
- Consumes: `errors.SourceError`, `Device` из `audio.sources`.
- Produces: `VideoSource` (протокол): `.frames() -> Iterator[np.ndarray]` (BGR uint8) и `.close()`;
  `CameraSource(index: int)` — `SourceError("Камера не найдена или занята")`; `list_cameras(max_index: int = 4) -> list[Device]`;
  `FileVideoSource(path: Path, realtime: bool = True)` — открывается в конструкторе (`SourceError("Не удалось прочитать видеофайл")`), при `realtime=True` соблюдает исходный FPS (запасное значение 25), в конце файла итератор завершается.

- [ ] **Step 1: Тесты:** `test_file_source_yields_all_frames` (сгенерировать 10 кадров 64×48 в `.avi` MJPG во `tmp_path`, `realtime=False` → 10 кадров формы `(48,64,3)`); `test_garbage_file_raises`; `test_camera_missing_raises` (monkeypatch `cv2.VideoCapture`, `isOpened() == False`); `test_list_cameras_returns_only_opened`.
- [ ] **Step 2:** `pytest tests/test_video_sources.py -v` — FAIL.
- [ ] **Step 3:** Реализовать; для камеры на Windows использовать `cv2.CAP_DSHOW`.
- [ ] **Step 4:** `pytest tests/test_video_sources.py -v` — PASS.
- [ ] **Step 5:** Коммит: `git commit -m "Источники видео: камера и файл"`.

---

### Task 8: Сессия и HTTP API

**Files:**
- Create: `src/dronedetect/session.py`, `src/dronedetect/server.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: всё из Tasks 1–7.
- Produces: `EventHub` — `.publish(event: dict)`, `.subscribe() -> queue.Queue`, `.unsubscribe(q)`; `DetectionSession(config: AppConfig, hub: EventHub)` — `.start(mode: Literal["audio","vision"], source: dict, threshold: float | None)` (бросает `DroneError`; `ConflictError(DroneError)` если уже запущено), `.stop()` (идемпотентно), `.set_threshold(value: float)`, `.status() -> dict`, `.latest_jpeg() -> bytes | None`; фабрика `create_app(config: AppConfig) -> FastAPI`.
- HTTP: `GET /api/status` → `{audio_available, vision_available, running, mode}`; `GET /api/devices` → `{microphones:[{id,name}], cameras:[{id,name}]}`; `POST /api/upload` (multipart `file`, лимит 200 МБ → 413) → `{file_id}`, файлы во временном каталоге; `POST /api/start` `{mode, source:{kind:"device"|"file", device_id?, file_id?}, threshold?}`; `POST /api/stop`; `POST /api/threshold` `{value}`; `GET /api/events` (SSE); `GET /api/video.mjpg`. Ошибки: JSON `{"error": "<русский текст>"}` с кодом 400 (`DroneError`), 409 (`ConflictError`). События: звук `{"type":"audio","t":float,"prob":float,"detected":bool}`; изображение `{"type":"vision","fps":float,"count":int,"detected":bool}`; конец файла `{"type":"ended"}`; ошибка в потоке `{"type":"error","message":str}`.

- [ ] **Step 1: Тесты (`TestClient`, конфиг на временные пути):** `test_status_reports_missing_models` (оба `*_available == False`, статус 200); `test_start_without_model_returns_400_with_text` (в `error` есть «модель»); `test_start_audio_with_golden_wav_file` (загрузить WAV через `/api/upload`, старт, `hub` получает событие `audio` с `prob` в [0,1]; затем `/api/stop`); `test_start_twice_returns_409`; `test_stop_when_idle_is_ok` (200, `running == False`); `test_start_other_mode_while_running_returns_409`; `test_upload_garbage_then_start_returns_400` (ошибка читается при старте, процесс сервера жив, `/api/status` отвечает); `test_upload_too_large_returns_413` (монкипатчить лимит).
- [ ] **Step 2:** `pytest tests/test_api.py -v` — FAIL.
- [ ] **Step 3:** Реализовать: рабочий поток на сессию; источник и детектор создаются синхронно в `start` (ошибки сразу в ответ), цикл в потоке публикует события через `hub`; для изображения кодировать кадр с рамками в JPEG и хранить последний; `stop` закрывает источник, ждёт поток, выгружает модель; при ошибке внутри цикла — событие `error` и остановка. SSE и MJPEG реализовать через `StreamingResponse`; MJPEG-эндпоинт завершается при остановке сессии.
- [ ] **Step 4:** `pytest -v` — все тесты PASS.
- [ ] **Step 5:** Коммит: `git commit -m "Сессия обнаружения и HTTP API"`.

---

### Task 9: Страница в браузере

**Files:**
- Create: `src/dronedetect/static/index.html`, `app.js`, `style.css`; Modify: `src/dronedetect/server.py` (отдача `static/` на `/`)

**Interfaces:**
- Consumes: HTTP API из Task 8.

- [ ] **Step 1:** Тест `test_index_served` в `tests/test_api.py`: `GET /` → 200, в теле есть «Звук» и «Изображение».
- [ ] **Step 2:** `pytest tests/test_api.py::test_index_served -v` — FAIL.
- [ ] **Step 3:** Реализовать страницу по эскизу спецификации: вкладки «Звук» / «Изображение»; выбор источника (устройство из `/api/devices` или файл через `/api/upload`); кнопки «Старт»/«Стоп»; ползунок порога (звук 0.5, изображение 0.25), отправка на `/api/threshold` при изменении; в режиме «Звук» — крупный индикатор «ДРОН ОБНАРУЖЕН / дрон не обнаружен», вероятность и график за 30 с на `<canvas>`; в режиме «Изображение» — `<img src="/api/video.mjpg">` и «FPS», «Найдено: N»; блок ошибок с текстом из `error`; недоступный режим (по `/api/status`) — вкладка отключена с пояснением. Смена вкладки сначала вызывает `/api/stop`. Без внешних CDN и шрифтов (работает офлайн).
- [ ] **Step 4:** `pytest -v` — PASS; ручная проверка: `python -m dronedetect`, открыть страницу во встроенном браузере, прогнать `tests/data`-WAV через вкладку «Звук» (индикатор и график двигаются), загрузить любое видео во вкладке «Изображение» (рамки и FPS видны), проверить узкий экран (≈375 px) и сообщение при остановке.
- [ ] **Step 5:** Коммит: `git commit -m "Веб-интерфейс: вкладки звука и изображения"`.

---

### Task 10: Запуск (CLI, run.sh, run.bat) и README

**Files:**
- Create: `src/dronedetect/__main__.py`, `run.sh`, `run.bat`, `README.md`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `server.create_app`, `config.load_config`.
- Produces: `__main__.parse_args(argv: list[str]) -> argparse.Namespace` с `--host` (по умолчанию `127.0.0.1`), `--port` (по умолчанию из конфига), `--no-browser`; `__main__.main(argv=None) -> None` — запускает uvicorn, через секунду после старта открывает `http://127.0.0.1:<port>` в браузере (`webbrowser`), при `--host 0.0.0.0` печатает предупреждение об открытом доступе к камере и микрофону.

- [ ] **Step 1: Тесты:** `test_default_host_is_localhost`; `test_host_flag_parsed`; `test_no_browser_flag`.
- [ ] **Step 2:** `pytest tests/test_cli.py -v` — FAIL.
- [ ] **Step 3:** Реализовать `__main__.py`. `run.sh`: проверить `python3` ≥ 3.11; создать `.venv` при отсутствии; `pip install -r requirements.txt`; проверить `ldconfig -p | grep -q portaudio` и `libsndfile`, при отсутствии напечатать `sudo apt install libportaudio2 libsndfile1` и завершиться с кодом 1; запустить `python -m dronedetect "$@"`. `run.bat` — то же на `cmd` (`py -3.11`, `.venv\Scripts\python`). `run.sh` пометить исполняемым (`git update-index --chmod=+x`).
- [ ] **Step 4:** `pytest -v` — PASS; проверить `run.bat` на этой Windows-машине с нуля (удалить `.venv`, запустить, открылась страница), `bash -n run.sh` без ошибок.
- [ ] **Step 5:** README на русском: требования, запуск на Pi и Windows, источники и лицензии моделей, подсказки для показа (порог, расстояние до микрофона, ограничения звуковой модели), как заменить модель, как повторить экспорт на Kaggle.
- [ ] **Step 6:** Коммит: `git commit -m "Скрипты запуска, CLI и README"`. Проверку на реальной Raspberry Pi выполняет пользователь (`git clone`, `./run.sh`); результаты и замечания фиксировать отдельным коммитом.
