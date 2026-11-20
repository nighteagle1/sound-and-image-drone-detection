from dronedetect.__main__ import parse_args


def test_default_host_is_localhost():
    args = parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port is None and args.no_browser is False


def test_host_flag_parsed():
    args = parse_args(["--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0" and args.port == 9000


def test_no_browser_flag():
    assert parse_args(["--no-browser"]).no_browser is True
