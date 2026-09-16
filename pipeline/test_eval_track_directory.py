"""Test directory discovery and missing references without simulator dependencies."""
import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np


class TrackDirectoryTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).with_name('eval.py')
        names = {'resolve_track_files', 'normalized_time', 'geometric_mean', 'track_set_metrics'}
        tree = ast.parse(source.read_text())
        self.ns = {'Path': Path, 'np': np, 'MAP_FILES': ['default.yaml'],
                   'TRACK_TIME_REFERENCES': {'known': 10.0}}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                     and n.name in names], type_ignores=[]), str(source), 'exec'), self.ns)

    def test_discovery(self):
        resolve = self.ns['resolve_track_files']
        self.assertEqual(resolve(None), ['default.yaml'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                resolve(directory)
            for name in ('b.yaml', 'a.yml', 'ignore.txt'):
                (root / name).touch()
            (root / 'nested').mkdir()
            (root / 'nested' / 'hidden.yaml').touch()
            self.assertEqual([Path(p).name for p in resolve(directory)], ['a.yml', 'b.yaml'])
            (root / 'a.yaml').touch()
            with self.assertRaises(ValueError):
                resolve(directory)

    def test_missing_references_and_empty_group(self):
        metrics = self.ns['track_set_metrics']
        self.assertEqual(metrics({})['n_tracks'], 0)
        result = metrics({'unknown': (True, 20.0), 'known': (True, 10.0)})
        self.assertEqual(result['success_rate'], 1.0)
        self.assertIsNone(result['normalized_time_gmean'])
        self.assertAlmostEqual(metrics({'known': (True, 20.0)})['normalized_time_gmean'], 2.0)


if __name__ == '__main__':
    unittest.main()
