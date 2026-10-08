from pathlib import Path
import tempfile
import unittest
import numpy as np
from algorithms.FindROI import FindROIConfig, ROIReferenceBank, find_roi_in_image


def image(seed, anomaly=False):
    y,x=np.mgrid[:100,:100]
    data=(x>=50).astype(float)
    if anomaly:
        data[((x-46)**2+(y-50)**2)<7**2]=1
    return data+np.random.default_rng(seed).normal(0,.005,data.shape)


class FindROITest(unittest.TestCase):
    config=FindROIConfig(analysis_pixel_size_nm=.25,patch_size_nm=5,stride_nm=1,max_reference_patches=2048)

    def reference(self):
        normal=[(image(i),25e-9) for i in range(3)]
        bank=ROIReferenceBank(self.config).fit(normal)
        bank.calibrate([(image(5),25e-9),(image(6),25e-9)],quantile=1,margin=1.1)
        return bank

    def test_localized_candidate_has_coordinates_scores_and_saved_annotation(self):
        bank=self.reference()
        result=find_roi_in_image(image(10,True),fov_m=25e-9,reference=bank)
        self.assertGreater(len(result.candidates),0)
        nearest=result.candidates[np.argmin((result.candidates['center_x_px']-46)**2+(result.candidates['center_y_px']-50)**2)]
        self.assertLess(abs(nearest['center_y_px']-50),15)
        self.assertGreater(nearest['score_max'],result.threshold)
        self.assertAlmostEqual(nearest['offset_x_m'],(nearest['center_x_px']+.5-50)*.25e-9,delta=1e-18)
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            for name in ('find_roi.npz','haadf.png','haadf_roi.png','roi_score_map.png'):
                self.assertTrue((Path(directory)/name).is_file())
            with np.load(Path(directory)/'find_roi.npz',allow_pickle=False) as d:
                np.testing.assert_array_equal(d['candidates'],result.candidates)

    def test_normal_reference_roundtrip_and_update_requires_recalibration(self):
        bank=self.reference()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.npz';bank.save(path)
            loaded=ROIReferenceBank.load(path)
            normal=find_roi_in_image(image(8),fov_m=25e-9,reference=loaded)
            self.assertEqual(len(normal.candidates),0)
            loaded.update([(image(20),25e-9)])
            self.assertIsNone(loaded.threshold)
            self.assertLessEqual(len(loaded.features),loaded.config.max_reference_patches)
            with self.assertRaisesRegex(ValueError,'Calibrate'):
                find_roi_in_image(image(8),fov_m=25e-9,reference=loaded)

    def test_unfitted_reference_and_invalid_images_are_rejected(self):
        bank=ROIReferenceBank(self.config)
        with self.assertRaises(ValueError):
            find_roi_in_image(image(1),fov_m=25e-9,reference=bank)
        with self.assertRaises(ValueError):
            bank.fit([(np.full((20,20),np.nan),25e-9)])

    def test_approved_corner_is_normal_but_new_protrusion_remains_anomalous(self):
        y,x=np.mgrid[:100,:100]
        def corner(seed):
            return ((x>=50)&(y>=50)).astype(float)+np.random.default_rng(seed).normal(0,.005,x.shape)
        bank=ROIReferenceBank(self.config).fit([(image(0),25e-9),(corner(1),25e-9),(corner(2),25e-9)])
        bank.calibrate([(image(5),25e-9),(corner(6),25e-9)],quantile=1,margin=1.05)
        normal=find_roi_in_image(corner(7),fov_m=25e-9,reference=bank)
        self.assertEqual(len(normal.candidates),0)
        abnormal=find_roi_in_image(image(10,True),fov_m=25e-9,reference=bank)
        self.assertGreater(len(abnormal.candidates),0)

    def test_mosaic_physical_placement_overlap_and_roi_labels(self):
        from algorithms.FindROI import save_roi_mosaic, ROI_DTYPE
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.savez(root/'left.npz',image=np.full((10,10),2.),fov_m=10e-9)
            np.savez(root/'right.npz',image=np.full((10,10),6.),fov_m=10e-9)
            candidate=np.zeros(1,dtype=ROI_DTYPE)
            candidate['id']=1;candidate['x0_px']=4;candidate['y0_px']=4
            candidate['x1_px']=6;candidate['y1_px']=6;candidate['score_max']=3
            np.savez(root/'roi.npz',candidates=candidate,descriptions=np.array(['shape variation']))
            info=save_roi_mosaic([
                {'capture_path':root/'right.npz','center_m':(5e-9,0),'box_id':2,'roi_path':root/'roi.npz'},
                {'capture_path':root/'left.npz','center_m':(0,0),'box_id':1}],root/'mosaic')
            self.assertEqual(info['tile_count'],2)
            np.testing.assert_allclose(info['candidates'][0]['center_m'],[5e-9,0])
            self.assertEqual(info['candidates'][0]['description'],'shape variation')
            with np.load(root/'mosaic/haadf_mosaic.npz') as d:
                self.assertAlmostEqual(float(d['image'][5,2]),2)
                self.assertAlmostEqual(float(d['image'][5,7]),4)
                self.assertAlmostEqual(float(d['image'][5,12]),6)
            self.assertTrue((root/'mosaic/haadf_mosaic_roi.png').is_file())
            small=save_roi_mosaic([{'capture_path':root/'left.npz','center_m':(0,0),'box_id':1}],root/'small',max_side_px=5)
            self.assertLessEqual(max(small['shape_px']),5)

    def test_normal_focus_variation_and_nonround_local_change(self):
        from scipy import ndimage
        bank=self.reference()
        normal=ndimage.gaussian_filter(image(30),2.0)
        result=find_roi_in_image(normal,fov_m=25e-9,reference=bank)
        self.assertEqual(len(result.candidates),0)
        defect=normal.copy()
        defect[45:55,65:75]=.25
        result=find_roi_in_image(defect,fov_m=25e-9,reference=bank)
        self.assertGreater(len(result.candidates),0)
        self.assertTrue(any(abs(roi['center_x_px']-70)<15 and abs(roi['center_y_px']-50)<15
                            for roi in result.candidates))
        self.assertEqual(result.reference_score_map.shape,result.image.shape)
        self.assertEqual(result.context_baseline_map.shape,result.image.shape)

    def test_context_suppresses_shared_shift_but_keeps_strong_broad_evidence(self):
        from algorithms.FindROI import _context_scores
        features=np.zeros((100,34));features[:,25]=.9;features[:,26]=.03
        ys=xs=np.arange(10)*10.
        thresholds=np.full(100,.1)
        shared=np.full(100,.5)
        corrected,_=_context_scores(features,shared,ys,xs,self.config,thresholds)
        self.assertTrue(np.all(corrected<.1))
        shared[55]=1.
        corrected,_=_context_scores(features,shared,ys,xs,self.config,thresholds)
        self.assertGreater(corrected[55],.1)
        broad=np.full(100,3.)
        corrected,_=_context_scores(features,broad,ys,xs,self.config,thresholds)
        self.assertTrue(np.all(corrected>thresholds))

    def test_cached_reference_index_rebuilds_when_weights_change(self):
        bank=ROIReferenceBank(self.config)
        bank.features=np.zeros((1,34));bank.center=np.zeros(34)
        bank.scale=np.ones(34);bank.weights=np.ones(34)
        query=np.ones((1,34))
        self.assertAlmostEqual(float(bank.query_features(query)[0][0]),1.)
        first=bank._index
        bank.query_features(query)
        self.assertIs(bank._index,first)
        bank.weights[:]=2.
        self.assertAlmostEqual(float(bank.query_features(query)[0][0]),2.)
        self.assertIsNot(bank._index,first)

    def test_detection_extracts_features_once_and_uses_shared_search(self):
        from unittest import mock
        from algorithms.roi import detection
        bank=self.reference()
        with mock.patch.object(detection,'_features',wraps=detection._features) as extract, mock.patch.object(bank,'query_features',wraps=bank.query_features) as query:
            result=find_roi_in_image(image(10,True),fov_m=25e-9,reference=bank)
        self.assertGreater(len(result.candidates),0)
        self.assertEqual(extract.call_count,1)
        self.assertEqual(query.call_count,1)

    def test_lazy_mosaic_decodes_each_image_once_and_composition_has_no_file_io(self):
        from algorithms.FindROI import compose_roi_mosaic, MosaicTile
        calls=[]
        def load():
            calls.append(1)
            return np.full((10,10),2.)
        result=compose_roi_mosaic([MosaicTile(load,10e-9,(0.,0.),1,shape_px=(10,10))])
        self.assertEqual(len(calls),1)
        np.testing.assert_allclose(result.image,2.)
        self.assertEqual(result.metadata['tile_count'],1)
