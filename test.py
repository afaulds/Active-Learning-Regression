import unittest
from sgd_linear import SGDLinear
import numpy as np


class TestAL(unittest.TestCase):

    def test_x(self):
        predictor = SGDLinear()
        predictor.fit(np.array([[3, 2], [4, 8]]), np.array([[7], [2]]))
        x = predictor.predict(np.array([[3, 2]]))
        print(x[0])
        x = predictor.predict(np.array([[4, 8]]))
        print(x[0])

        predictor.fit(np.array([[3, 2], [4, 8]]), np.array([[7], [2]]))
        x = predictor.predict(np.array([[3, 2]]))
        print(x[0])
        x = predictor.predict(np.array([[4, 8]]))
        print(x[0])

        predictor.fit(np.array([[3, 2], [4, 8]]), np.array([[7], [2]]))
        x = predictor.predict(np.array([[3, 2]]))
        print(x[0])
        x = predictor.predict(np.array([[4, 8]]))
        print(x[0])

    def test_predictor(self):
        return
        predictor = SGDLinear()
        predictor.num_epochs = 1
        predictor.fit(np.array([[3, 2]]), np.array([[7]]))
        x = predictor.predict(np.array([[3, 2]]))
        self.assertEqual(x[0], 4.9)
        predictor.fit(np.array([[3, 2]]), np.array([[7]]))
        x = predictor.predict(np.array([[3, 2]]))
        self.assertEqual(x[0], 6.37)
        predictor.fit(np.array([[3, 2]]), np.array([[7]]))
        x = predictor.predict(np.array([[3, 2]]))
        self.assertEqual(x[0], 6.811)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()