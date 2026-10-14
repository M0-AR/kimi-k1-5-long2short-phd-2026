"""Run all experiments in order. Verified end-to-end entrypoint."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
ORDER = [
    "01_overthinking_curve.py",
    "02_long2short_methods.py",
    "03_data_mix_audit.py",
    "04_live_market_reasoning.py",
    "05_hidden_patterns.py",
]

def main():
    for f in ORDER:
        print(f"\n===== {f} =====")
        r = subprocess.run([sys.executable, str(HERE / f)], check=False)
        if r.returncode != 0:
            print(f"FAILED: {f} (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\nAll experiments passed. See benchmarks/*.json + *.png")

if __name__ == "__main__":
    main()
