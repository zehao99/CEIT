from unittest import TestCase
from MyEIT.util.utilities import quicksort, Comp, PointStack
from MyEIT.models.mesh import MeshObj, ccw
import numpy as np
import matplotlib.pyplot as plt


class Test(TestCase):

    def test_quicksort(self):
        array = [[1, 1]]
        for i in range(5000):
            array.append([1, np.random.rand() * 9999999999999999999999999997])
        array = np.array(array)
        comp = Comp([-0.05, -0.05])
        Test.assertEqual(self, comp.compare([1, 0], [0, 1]) < 0, True)
        quicksort(array, comp)
        x1 = array[0][1]
        for i, node in enumerate(array):
            if i == 0:
                continue
            Test.assertGreaterEqual(self, node[1], x1)
            x1 = node[1]

    def test_ccw(self):
        print(ccw([0.048, -0.008, 0.0], [-0.08, -0.032, 1.0], [0.034, -0.014, 2000]))

    def test_stack(self):
        s = PointStack(np.array([[0, 1, 1]]))
        s.push(np.array([0, 1, 100]))
        Test.assertEqual(self, s.peek()[2], np.array([0, 1, 100])[2])
        Test.assertEqual(self, s.pop()[2], np.array([0, 1, 100])[2])
        Test.assertEqual(self, s.pop()[2], np.array([0, 1, 1])[2])
        Test.assertEqual(self, s.isEmpty(), True)

    def test_get_perimeter(self):
        mesh = MeshObj()
        ans = mesh.get_perimeter()
        points_x = []
        points_y = []
        for n in ans:
            points_x.append(mesh.nodes[int(n)][0])
            points_y.append(mesh.nodes[int(n)][1])
