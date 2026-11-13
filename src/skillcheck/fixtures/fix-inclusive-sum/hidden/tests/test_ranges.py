import unittest

from ranges import sum_inclusive


class SumTests(unittest.TestCase):
    def test_span(self) -> None:
        self.assertEqual(sum_inclusive(1, 3), 6)

    def test_single(self) -> None:
        self.assertEqual(sum_inclusive(5, 5), 5)

    def test_negative(self) -> None:
        self.assertEqual(sum_inclusive(-2, 2), 0)

    def test_descending_empty(self) -> None:
        self.assertEqual(sum_inclusive(3, 1), 0)
