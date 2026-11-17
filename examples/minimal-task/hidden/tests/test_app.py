import unittest

from app import add


class AddTests(unittest.TestCase):
    def test_positive(self) -> None:
        self.assertEqual(add(2, 3), 5)

    def test_negative(self) -> None:
        self.assertEqual(add(-1, 1), 0)
