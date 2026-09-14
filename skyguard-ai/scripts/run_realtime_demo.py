import subprocess
import time
import requests
import sys
import argparse
from pathlib import Path

def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            time.sleep(1)
    return False

def run_demo(speed: float):
    print("Starting FastAPI Backend Server...")
    
    # We use subprocess.Popen to start it in the background
    # Set cwd to the project root
    root_dir = Path(__file__).resolve().parent.parent
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(root_dir)
    )
    
    print("Waiting for server to become healthy...")
    if not wait_for_server("http://127.0.0.1:8000/api/health"):
        print("Server failed to start in time.")
        server_process.terminate()
        sys.exit(1)
        
    print("Server is healthy! Launching Real-Time Simulator...")
    
    simulator_process = subprocess.Popen(
        [sys.executable, "scripts/replay_dataset.py", "--speed", str(speed)],
        cwd=str(root_dir)
    )
    
    try:
        # Wait for simulator to finish
        simulator_process.wait()
        print("Demo completed successfully!")
    except KeyboardInterrupt:
        print("Interrupt received, shutting down...")
    finally:
        print("Terminating server...")
        server_process.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=100.0, help="Replay speed multiplier")
    args = parser.parse_args()
    
    run_demo(args.speed)
