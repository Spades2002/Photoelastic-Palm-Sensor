"""Run every thesis plotting script in sequence."""
from pathlib import Path
import subprocess
import sys

HERE = Path(r"D:\\ERP\\Graphs")
SCRIPTS = [
    "01_fz_predicted_vs_ground_truth.py",
    "02_fz_mae_architecture_comparison.py",
    "03_contact_localisation_2d.py",
    "04_two_stage_depth_prediction.py",
    "05_fz_absolute_error_vs_depth.py",
    "06_fz_absolute_error_distributions.py",
]

for script in SCRIPTS:
    print(f"\nRunning {script}...")
    subprocess.run([sys.executable, str(HERE / script)], check=True)

print("\nAll figures generated successfully.")
