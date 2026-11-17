"""Diff accounting."""

from skillcheck.metrics import diff_trees, line_delta


def test_aligned_edit_counts_added_and_removed_lines() -> None:
    stats = diff_trees(
        {"a.txt": b"a\nb\nc\n"},
        {"a.txt": b"a\nB\nc\nd\n"},
    )
    assert stats.files == 1
    assert stats.lines_added == 2
    assert stats.lines_removed == 1


def test_new_and_deleted_files() -> None:
    added = diff_trees({}, {"a.txt": b"x\ny\n"})
    assert (added.files, added.lines_added, added.lines_removed) == (1, 2, 0)
    removed = diff_trees({"a.txt": b"x\ny\n"}, {})
    assert (removed.files, removed.lines_added, removed.lines_removed) == (1, 0, 2)


def test_unchanged_tree_is_zero() -> None:
    stats = diff_trees({"a.txt": b"same\n"}, {"a.txt": b"same\n"})
    assert stats.files == 0
    assert stats.lines_added == 0
    assert stats.lines_removed == 0


def test_large_file_uses_newline_count() -> None:
    before = b"a\n" * 100_001
    after = b"a\n" * 100_002
    assert len(before) > 200_000
    assert line_delta(before, after) == (1, 0)
    assert line_delta(after, before) == (0, 1)
