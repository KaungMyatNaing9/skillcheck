import unittest

from slug import slugify


class SlugTests(unittest.TestCase):
    def test_words(self) -> None:
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_whitespace(self) -> None:
        self.assertEqual(slugify("  a  b  "), "a-b")

    def test_existing_hyphen(self) -> None:
        self.assertEqual(slugify("Already-Ok"), "already-ok")

    def test_punctuation(self) -> None:
        self.assertEqual(slugify("C++ is fun!"), "c-is-fun")
