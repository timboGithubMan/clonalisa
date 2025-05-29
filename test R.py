import subprocess, sys, pathlib

res = subprocess.run(["Rscript", pathlib.Path("interaction.R"), r"2024_03_29_02_03_10.875324_epoch_960\2024_03_29_02_03_10.875324_epoch_960_all_data.csv"], capture_output=True, text=True)

print(res.stdout)        # R’s standard output
if res.returncode != 0:
    print(res.stderr, file=sys.stderr)