"""
run.py – Single unified launcher for BNS Legal Intelligence Multi-Agent RAG.

Runs the FastAPI backend and React frontend TOGETHER on a single port (http://127.0.0.1:8000)
and automatically launches your default web browser.

Usage:
    python run.py
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess
import uvicorn

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FRONTEND_DIST = os.path.join(FRONTEND_DIR, "dist")


def ensure_frontend_build():
    """Ensure React production build exists; if not, build it automatically."""
    if not os.path.exists(FRONTEND_DIST) or not os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        print("[Launcher] React build not found. Building frontend now...")
        cmd = "npm run build" if sys.platform != "win32" else "npm.cmd run build"
        res = subprocess.run(cmd, cwd=FRONTEND_DIR, shell=True)
        if res.returncode != 0:
            print("[Launcher] Warning: Frontend build failed. Will continue in API-only mode.")
        else:
            print("[Launcher] Frontend build completed successfully.")


def open_browser(url: str):
    """Open default web browser after server starts."""
    import urllib.request
    print("\n[Launcher] Waiting for server to initialize...")
    while True:
        try:
            # Try to fetch the health endpoint
            res = urllib.request.urlopen(f"{url}/api/health", timeout=1)
            if res.getcode() == 200:
                break
        except Exception:
            time.sleep(0.5)
            
    print(f"\n[Launcher] 🚀 Opening application in browser: {url}")
    webbrowser.open(url)


def main():
    print("=" * 60)
    print("   ⚖️  BNS Legal Intelligence · Multi-Agent RAG")
    print("   Unified Full-Stack Launcher (React + FastAPI)")
    print("=" * 60)

    # 1. Verify frontend build
    ensure_frontend_build()

    url = "http://127.0.0.1:8000"
    print(f"\n[Launcher] Starting unified server at {url}")
    print("[Launcher] Backend APIs + React Frontend are running together on Port 8000.")
    print("=" * 60)

    # 2. Open browser in separate daemon thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # 3. Launch Uvicorn server
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
