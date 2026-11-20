from __future__ import annotations

import argparse
import threading
import webbrowser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dronedetect", description="Обнаружение дронов по звуку и изображению")
    p.add_argument("--host", default="127.0.0.1", help="адрес сервера (по умолчанию 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="порт (по умолчанию из config.yaml)")
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from .config import load_config
    from .server import create_app

    args = parse_args(argv)
    config = load_config()
    port = args.port or config.port
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("ВНИМАНИЕ: сервер доступен по сети. Страница без авторизации открывает "
              "камеру и микрофон этого устройства любому в сети.")
    url = f"http://127.0.0.1:{port}"
    print(f"Открой {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(config), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
