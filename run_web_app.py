"""Launch the local PRODUCTIZATION UX-3 decision experience."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from webapp.server import BonsaiWebApplication, create_server


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Bonsai Decision Agent local UX-3 web application")
    parser.add_argument("--host", default="127.0.0.1", help="Local bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Local bind port (default: 8765)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Application data directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    app = BonsaiWebApplication(project_root=PROJECT_ROOT, data_dir=args.data_dir)
    server = create_server(app, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"Bonsai Decision Agent UX-3 running at http://{host}:{port}")
    print("Connection API keys are stored as plaintext in data/secrets.json; protect filesystem access.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UX-3 server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
