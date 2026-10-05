# TEM Automation

The runtime code is deliberately divided into only three layers:

```text
src/tem_automation/
├── adapters/
│   ├── base.py
│   └── nion_usim.py
├── algorithms/
│   └── autofocus.py
└── scripts/
    └── usim_haadf_autofocus.py
```

- `adapters`: defines a small vendor-independent microscope API, stores the
  current acquisition configuration, and translates calls to a vendor API.
- `algorithms`: owns algorithm-specific defaults and calls only the
  vendor-independent microscope API.
- `scripts`: initializes an instrument and lists the high-level experiment
  actions to execute.

The `tests` directory is development support and is not part of the runtime
architecture.

## Install in the Nion Swift development environment

```bat
conda activate nionswift-dev
cd /d D:\Development\tem-automation
python -m pip install -e .
```

## Test

```bat
python -m unittest discover -s tests -v
```

## Run in Nion Swift

Start Nion Swift and uSim, then choose `File > Scripts...` and add:

```text
D:\Development\tem-automation\src\tem_automation\scripts\usim_haadf_autofocus.py
```

Double-click the script to run HAADF autofocus. The script contains only the
instrument initialization and the high-level `autofocus(microscope)` action.
