"""Optional local replay against recorded pre-refactor HAADF fingerprints."""
from pathlib import Path
import hashlib
import json
import unittest
import numpy as np
from algorithms.FindROI import ROIReferenceBank, find_roi_in_image


class SavedROIRegressionTest(unittest.TestCase):
    def test_saved_images_match_pre_refactor_outputs_exactly(self):
        project=Path(__file__).resolve().parents[1]
        baseline=project/'tests/edge_review_results/refactor_baseline.json'
        reference_path=project/'src/scripts/roi_reference.npz'
        if not baseline.is_file() or not reference_path.is_file():
            self.skipTest('Local microscopy captures/reference are not committed')
        saved=json.loads(baseline.read_text())
        reference=ROIReferenceBank.load(reference_path)
        self.assertEqual(saved['model_id'],reference.model_id)
        root=project/'src/scripts/particle_edge_workflow_results/20261008_164305_658648_dc9ab569/particle_0002'
        for expected in saved['results']:
            with self.subTest(box=expected['box_id']):
                with np.load(root/expected['box_id']/'haadf.npz',allow_pickle=False) as data:
                    result=find_roi_in_image(data['image'],fov_m=float(data['fov_m']),reference=reference)
                for field,digest in expected['digests'].items():
                    self.assertEqual(hashlib.sha256(getattr(result,field).tobytes()).hexdigest(),digest,field)
                self.assertEqual(list(result.descriptions),expected['descriptions'])
