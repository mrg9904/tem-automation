from datetime import datetime
from pathlib import Path
from uuid import uuid4

from scripts import __file__ as scripts_package_file

from adapters.nion_usim import NionUSimAdapter
from algorithms.FindParticles import find_particles


def script_main(api_broker):
    """Save current-FoV arrays and annotated HAADF in a separate run folder."""
    api = api_broker.get_api(version="~1.0")
    output_root = Path(scripts_package_file).resolve().parent / "find_particles_results"
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    output_directory = output_root / run_name
    output_directory.mkdir(parents=True, exist_ok=False)
    output_path = output_directory / "find_particles.npz"
    with NionUSimAdapter(api, image_size_px=512) as microscope:
        result = find_particles(microscope, output_path=output_path)
    result.save_images(output_directory)
    print(f"Found {len(result.particles)} particles. Results: {output_directory}")
