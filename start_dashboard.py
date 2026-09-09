"""Launch the Regime Engine dashboard (FastAPI backend + Vite dev server)."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def kill_port(port: int):
    import socket
    import psutil
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            try:
                psutil.Process(conn.pid).kill()
                print(f"  killed process {conn.pid} on port {port}")
            except Exception:
                pass


def main():
    try:
        import psutil
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil

    print("Starting Regime Engine dashboard...")

    # Stop anything on the target ports
    kill_port(8000)
    kill_port(5173)
    time.sleep(1)

    # Backend
    print("  [1/2] Starting FastAPI backend on :8000")
    backend = subprocess.Popen(
        [sys.executable, "api/run.py"],
        cwd=str(ROOT),
        stdout=open(ROOT / "api_server.log", "w"),
        stderr=open(ROOT / "api_server_err.log", "w"),
    )

    # Wait for backend
    for _ in range(40):
        if backend.poll() is not None:
            print("  ERROR: backend exited early")
            sys.exit(1)
        try:
            import socket
            s = socket.create_connection(("127.0.0.1", 8000), timeout=1)
            s.close()
            break
        except OSError:
            time.sleep(0.5)
    else:
        print("  ERROR: backend did not start in time")
        sys.exit(1)
    print("  backend ready")

    # Frontend
    print("  [2/2] Starting Vite dev server on :5173")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(FRONTEND),
        shell=True,
        stdout=open(FRONTEND / "dev_server.log", "w"),
        stderr=open(FRONTEND / "dev_server_err.log", "w"),
    )

    print("\n  Dashboard: http://localhost:5173")
    print("  API:       http://localhost:8000/docs")
    print("\n  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in (backend, frontend):
            p.terminate()
        for p in (backend, frontend):
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
