"""Selection helpers used to evaluate Anemoi/Zarr reader strategies."""


def coalesced_ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Return inclusive-exclusive ranges covering sorted unique indices."""
    if not indices:
        return []
    ordered = sorted(set(indices))
    ranges = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    return ranges
