from tem_automation.runners.nion_usim_autofocus import run


# Parameters for the HAADF autofocus experiment.
SEARCH_HALF_RANGE_NM = 200.0
COARSE_POINTS = 9
FINE_POINTS = 7

FOV_NM = 100.0
IMAGE_SIZE_PX = 256
DWELL_TIME_US = 1.0

SETTLE_TIME_S = 0.0
FRAMES_PER_POSITION = 1


def script_main(api_broker):
    """Entry point called automatically by Nion Swift."""

    api = api_broker.get_api(
        version="~1.0",
    )

    print("Starting uSim HAADF autofocus...")

    result = run(
        api=api,
        search_half_range_nm=SEARCH_HALF_RANGE_NM,
        coarse_points=COARSE_POINTS,
        fine_points=FINE_POINTS,
        fov_nm=FOV_NM,
        image_size_px=IMAGE_SIZE_PX,
        dwell_time_us=DWELL_TIME_US,
        settle_time_s=SETTLE_TIME_S,
        frames_per_position=FRAMES_PER_POSITION,
    )

    print("")
    print("HAADF autofocus completed.")
    print(
        "Original defocus: "
        f"{result.original_defocus_m * 1e9:.2f} nm"
    )
    print(
        "Best defocus: "
        f"{result.best_defocus_m * 1e9:.2f} nm"
    )
    print(
        "Best focus score: "
        f"{result.best_score:.6f}"
    )

    print("")
    print("Focus measurements:")

    for measurement in result.measurements:
        print(
            f"{measurement.defocus_m * 1e9:9.2f} nm    "
            f"{measurement.score:.6f}"
        )