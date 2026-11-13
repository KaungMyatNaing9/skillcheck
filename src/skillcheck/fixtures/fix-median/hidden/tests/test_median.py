import unittest

from stats_util import median


class MedianTests(unittest.TestCase):
    def test_odd(self) -> None:
        self.assertEqual(median([1, 3, 2]), 2)

    def test_even(self) -> None:
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_single(self) -> None:
        self.assertEqual(median([5]), 5)

    def test_pair(self) -> None:
        self.assertEqual(median([1, 100]), 50.5)

    def test_empty(self) -> None:
        with self.assertRaises(ValueError):
            median([])
