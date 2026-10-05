# TEM Automation

Vendor-independent automation experiments for transmission electron microscopes.

The experiment code depends only on the `Microscope` protocol. Vendor-specific
APIs are isolated in adapters. The first experiment is two-pass HAADF autofocus
and the first hardware adapter targets Nion uSim.

## Install in the Nion Swift development environment

```bat
conda activate nionswift-dev
cd /d D:\Development\tem-automation
python -m pip install -e .
```

Run the tests without additional test dependencies:

```bat
python -m unittest discover -s tests -v
```

## Run in Nion Swift

Start Nion Swift and uSim, open the Python Console, and run:

```python
from tem_automation.runners.nion_usim_autofocus import run

result = run(
    search_half_range_nm=200.0,
    coarse_points=9,
    fine_points=7,
    fov_nm=100.0,
    image_size_px=256,
    dwell_time_us=1.0,
)
```

The microscope is left at `result.best_defocus_m` after a successful run. The
returned result contains every tested defocus and its focus score.
