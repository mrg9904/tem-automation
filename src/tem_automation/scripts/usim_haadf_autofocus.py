from tem_automation.adapters.nion_usim import NionUSimAdapter
from tem_automation.algorithms.autofocus import autofocus


# Instrument initialization parameters only.
FOV_NM = 100.0


def script_main(api_broker):
    """Entry point called automatically by Nion Swift."""
    api = api_broker.get_api(version="~1.0")
    with NionUSimAdapter(api, fov_nm=FOV_NM) as microscope:
        autofocus(microscope)
