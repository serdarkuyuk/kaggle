import os
import pandas as pd


class Paths:
    """Helper to put submission files in the right folder"""

    def __init__(self):
        self._data = None
        self._submissions = None
        self._test_set = None
        return

    @property
    def data(self):
        """The path to the data set"""
        if self._data is None:
            self._data = "../data/"
            if not os.path.isdir(self._data):
                os.mkdir(self._data)
        return self._data

    @property
    def submissions(self):
        """Path to the submissions"""
        if self._submissions is None:
            self._submissions = os.path.join(self.data, "submissions/")
            if not os.path.isdir(self._submissions):
                os.mkdir(self._submissions)
        return self._submissions

    @property
    def test_set(self):
        """path to the test-set data"""
        if self._test_set is None:
            self._test_set = os.path.join(self.data, "test_pairs.csv")
        return self._test_set

    def submit(self, filename):
        """Add the filename to the path

        Args:
         filename (str): name to add to the submissions folder

        Returns:
         str: path to the file in the submissions folder
        """
        return os.path.join(self.submissions, filename)


class TestSet:
    """Loads the test-set data

    Args:
     paths: object with the path to the test-set
    """

    def __init__(self, paths=Paths):
        self.paths = paths()
        self._data = None
        return

    @property
    def data(self):
        """the test-set data

        Returns:
         `pandas.DataFrame`: the test-set data
        """
        if self._data is None:
            self._data = pd.read_csv(self.paths.test_set)
        return self._data


ser = Paths()
ser.submissions
