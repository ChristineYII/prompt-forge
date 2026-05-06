import argparse
import os
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the prompt-forge server.")
    parser.add_argument(
        "--model-config",
        default="config.json",
        metavar="PATH",
        help="Path to model role config JSON (default: config.json)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["MODEL_CONFIG_PATH"] = args.model_config
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
