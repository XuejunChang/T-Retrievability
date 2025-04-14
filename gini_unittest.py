import unittest

import numpy as np

import gini

def calc(arr):
    print('-' * 10)
    print(arr)
    oldest = gini.gini_oldest(arr)
    print("gini_oldest:",oldest)

    formal = gini.gini_formal(arr)
    print("gini_formal:", formal)

    # converted = gini.gini_alternative(arr)
    # print("gini_converted:", converted)
    #
    # simplified = gini.gini_alternative_convt(arr)
    # print("gini_simplified:",simplified)
    # print('-' * 10)

class TestStringMethods(unittest.TestCase):
    def test_1(self):
        arr = np.ones(10)
        calc(arr)

    def test_2(self):
        arr = -1 *  np.ones(10)
        calc(arr)

    def test_3(self):
        arr = np.ones(10) + 1
        calc(arr)

    def test_4(self):
        arr = [1, 1, 1, 1, 1]
        calc(arr)

    def test_5(self):
        arr = [0, 0, 0, 0, 0]
        calc(arr)

    def test_6(self):
        arr = [1, 3, 2]
        calc(arr)

    def test_7(self):
        arr = [1, 3, 2, 5, 4]
        calc(arr)

    def test_8(self):
        arr = [10, 20, 30, 40, 50]
        calc(arr)

    def test_9(self):
        arr = np.random.randint(-200, 200, size=(10,))
        calc(arr)

    def test_a(self):
        arr = np.zeros(1)
        arr = np.append(arr, [1])
        calc(arr)

    def test_b(self):
        arr = np.zeros(1)
        arr = np.append(arr, [1]).astype(int)
        calc(arr)

    def test_c(self):
        arr = np.zeros(10)
        arr = np.append(arr, [1])
        calc(arr)

    def test_d(self):
        arr = np.zeros(5)
        arr = np.append(arr, [1,1,1,1,1])
        calc(arr)

    def test_e(self):
        arr = np.zeros(50)
        arr = np.append(arr, [1])
        calc(arr)

    def test_f(self):
        arr = np.random.randint(1, 200, size=(10,))
        calc(arr)
        arr = -1 * arr
        calc(arr)

    def test_g(self):
        arr = np.zeros(50)
        arr = np.append(arr, [1]).astype(int)
        calc(arr)

    def test_g(self):
        arr = np.random.randint(-200, 200, size=(101093,))
        calc(arr)


if __name__ == '__main__':
    unittest.main()
