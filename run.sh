#!/usr/bin/env bash
# Запуск обнаружения дронов на Raspberry Pi / Linux: ./run.sh [--host 0.0.0.0] [--port N] [--no-browser]
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Нужен Python 3.11 или новее (python3)." >&2
    exit 1
fi

missing=""
ldconfig -p 2>/dev/null | grep -q portaudio || missing="$missing libportaudio2"
ldconfig -p 2>/dev/null | grep -q libsndfile || missing="$missing libsndfile1"
if [ -n "$missing" ]; then
    echo "Не хватает системных пакетов. Выполни:" >&2
    echo "  sudo apt install$missing" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m dronedetect "$@"
