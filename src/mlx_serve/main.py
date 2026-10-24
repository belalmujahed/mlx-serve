import argparse
from collections.abc import Sequence

import uvicorn

from .config import config
from .inference import app

CLIENT_HELP = """\
Claude Code setup (shows these models in its /model picker):
  export ANTHROPIC_BASE_URL=http://{host}:{port}
  export ANTHROPIC_AUTH_TOKEN=local
  export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
Switch models from the picker, `/model org/name`, or POST /v1/model.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-serve",
        description="Local Claude Messages API-compatible server for MLX models.",
        epilog=CLIENT_HELP.format(host="127.0.0.1", port=8888),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="Hugging Face repo id to load at startup")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="start without a model; load on first request or POST /v1/model",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config.apply_overrides(
        MODEL_NAME=args.model,
        PRELOAD=False if args.no_preload else None,
        HOST=args.host,
        PORT=args.port,
        TRUST_REMOTE_CODE=args.trust_remote_code,
        VERBOSE=args.verbose,
    )
    print(CLIENT_HELP.format(host=config.HOST, port=config.PORT))
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
