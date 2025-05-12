import unittest

import numpy as np
import gini
import fair_utils

def calc(arr):
    print('-' * 10)
    print(arr)
    old11 = fair_utils.old_gini(arr,intv=11)
    print("old_gini_interval_10",old11)

    old101 = fair_utils.old_gini(arr,intv=101)
    print("old_gini_interval_100",old101)

    old1001 = fair_utils.old_gini(arr,intv=1001)
    print("old_gini_interval_1000",old1001)

    basic_gini = fair_utils.raw_gini(arr)
    print("basic_gini:", basic_gini)

    fast_gini = fair_utils.curve_gini(arr)
    print("fast_gini:", fast_gini)

    arr = dict(enumerate(arr))
    java_gini = gini.compute_gini(arr)
    print("java_gini:", java_gini)
    #
    # simplified = fair_utils.gini_alternative_convt(arr)
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

    # def test_g(self):
    #     arr = np.random.randint(-200, 200, size=(101093,))
    #     calc(arr)


if __name__ == '__main__':
    unittest.main()
