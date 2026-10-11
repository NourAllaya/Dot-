import tempfile
from pathlib import Path
import cv2
import numpy as np
from perception import Detector
from stereo_pair import process


def test_real_model():
    d = Detector()
    assert d.names == {0:'rock',1:'rock_shadow'}
    assert (d.height,d.width) == (640,640)
    assert isinstance(d.detect(np.zeros((240,320,3),np.uint8)),list)


def test_stereo_metric_depth():
    # Rectified synthetic texture at disparity 16: Z = 250 * 0.2 / 16 = 3.125 m.
    rng = np.random.default_rng(9)
    gray = rng.integers(0,256,(240,320),dtype=np.uint8)
    left = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    right = np.zeros_like(left)
    right[:,:-16] = left[:,16:]
    class Boxes:
        def detect(self,image):
            return [dict(cls='rock',x=180,y=70,w=90,h=100)]
    with tempfile.TemporaryDirectory() as tmp:
        k = np.array([[250.,0,160],[0,250,120],[0,0,1]])
        path = Path(tmp)/'cal.npz'
        np.savez(path,K1=k,K2=k,D1=np.zeros(5),D2=np.zeros(5),R=np.eye(3),
                 T=np.array([-0.2,0,0]),image_size=np.array([320,240]))
        _,dets = process(left,right,path,Boxes())
        assert abs(dets[0]['depth_m']-3.125)<0.05,dets
        _,empty = process(np.zeros_like(left),np.zeros_like(right),path,Boxes())
        assert empty[0]['depth_m'] is None,empty

if __name__=='__main__':
    test_real_model();print('PASS actual ONNX inference')
    test_stereo_metric_depth();print('PASS stereo metric depth and unavailable handling')
