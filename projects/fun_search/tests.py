import unittest

from utils import flatten_dict


class BigTest(unittest.TestCase):
    def xtest_dummy(self):
        self.assertEqual(True, False)

    def test_flatten_dict(self):
        d = {
            "a": 0,
            "letters": {
                "b": 1,
                "c": 2,
                "letters": {
                    "d": 3,
                    "e": 4,
                }
            }
        }
        target_d = {
            "a": 0,
            "letters/b": 1,
            "letters/c": 2,
            "letters/letters/d": 3,
            "letters/letters/e": 4,
        }
        pred_d = flatten_dict(d)
        self.assertEqual(target_d, pred_d)


if __name__ == "__main__":
    unittest.main()