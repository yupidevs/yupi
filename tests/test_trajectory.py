import yupi
import unittest
import os

class TrajectoryTest(unittest.TestCase):

    def test_creation(self):
        try:
            yupi.Trajectory(x=[1.0, 2.0], y=[2.0, 3.0])
        except Exception as e:
            self.fail(f'Trajectory creation fails. Exeption: {e}')

        try:
            yupi.Trajectory(
                x=[1.0, 2.0],
                y=[2.0, 3.0],
                z=[1.0, 3.0],
                t=[0.0, 1.0],
                theta=[0.0,0.0],
                id='test'
            )
        except Exception as e:
            self.fail(f'Trajectory creation fails. Exeption: {e}')
        

    def test_iteration(self):
        t1 = yupi.Trajectory(x=[1.0, 2.0], y=[2.0, 3.0])
        tps = [(tp.x, tp.y) for tp in t1]

        self.assertGreater(len(tps), 0)
        self.assertEqual(tps, [(1,2),(2,3)])

    def _test_save(self):
        t1 = yupi.Trajectory(x=[1.0,2.0,3.0])

        # Wrong trajectory file extension at saving
        with self.assertRaises(ValueError):
            t1.save('t1', file_type='abc')

        # Saving json
        try:
            t1.save('t1', file_type='json')
        except Exception as e:
            self.fail(f'Trajectory json save fails. Exeption: {e}')

        # Saving csv
        try:
            t1.save('t1', file_type='csv')
        except Exception as e:
            self.fail(f'Trajectory csv save fails. Exeption: {e}')
        

    def test_save_and_load(self):
        self._test_save()

        # Wrong trajectory file extension at loading
        with self.assertRaises(ValueError):
            t1 = yupi.Trajectory.load('t1.abc')

        # Loading json
        try:
            t1 = yupi.Trajectory.load('t1.json')
        except Exception as e:
            self.fail(f'Trajectory json load fails. Exeption: {e}')

        # Loading csv
        try:
            t1 = yupi.Trajectory.load('t1.csv')
        except Exception as e:
            self.fail(f'Trajectory csv load fails. Exeption: {e}')
    
    @classmethod
    def tearDownClass(cls):
        os.remove('t1.json')
        os.remove('t1.csv')

if __name__ == '__main__':
    unittest.main()
