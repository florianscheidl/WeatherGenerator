from weathergen.datasets.anemoi_channel_selection import coalesced_ranges


def test_coalesced_ranges_sorts_deduplicates_and_groups_adjacent_indices() -> None:
    assert coalesced_ranges([8, 2, 3, 3, 5, 7]) == [(2, 4), (5, 6), (7, 9)]


def test_coalesced_ranges_handles_empty_selection() -> None:
    assert coalesced_ranges([]) == []
