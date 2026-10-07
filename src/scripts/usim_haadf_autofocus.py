from adapters.nion_usim import NionUSimAdapter
from algorithms.autofocus import autofocus

def script_main(api_broker):
    """Entry point called automatically by Nion Swift."""
    api = api_broker.get_api(version="~1.0")
    with NionUSimAdapter(api) as microscope:
        autofocus(microscope)
