"""Detect on a rectified LEFT image; estimate depth from calibrated stereo.
Calibration NPZ: K1,D1,K2,D2,R,T,image_size. T must be in metres.
Distance is median optical-axis depth of valid pixels in the box, not crater depth.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from perception import Detector


def process(left, right, calibration, detector):
    if left is None or right is None or left.shape != right.shape:
        raise ValueError('Provide readable left/right images of identical dimensions')
    h, w = left.shape[:2]
    with np.load(calibration) as c:
        if tuple(c['image_size']) != (w, h):
            raise ValueError('Image resolution must match calibration image_size [width,height]')
        if np.linalg.norm(c['T']) <= 0:
            raise ValueError('Calibration baseline must be nonzero, in metres')
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            c['K1'], c['D1'], c['K2'], c['D2'], (w,h), c['R'], c['T'].astype(np.float64).reshape(3,1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if p2[0,3] >= 0 or abs(p2[1,3]) > 1e-6:
            raise ValueError('Requires horizontal stereo with right camera to the right of left camera')
        images = []
        for image, k, d, r, p in ((left,c['K1'],c['D1'],r1,p1),(right,c['K2'],c['D2'],r2,p2)):
            m1,m2 = cv2.initUndistortRectifyMap(k,d,r,p,(w,h),cv2.CV_32FC1)
            images.append(cv2.remap(image,m1,m2,cv2.INTER_LINEAR))
    num = min(128, ((w // 2) // 16)*16)
    if num < 16:
        raise ValueError('Images are too narrow for stereo matching')
    matcher = cv2.StereoSGBM_create(minDisparity=0,numDisparities=num,blockSize=5,
        P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=100,speckleRange=2)
    disparity = matcher.compute(*[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]).astype(np.float32)/16
    points = cv2.reprojectImageTo3D(disparity,q)
    valid = (disparity > 0) & np.isfinite(points).all(axis=2) & (points[:,:,2] > 0)
    roi_mask = np.zeros((h,w),bool)
    x,y,rw,rh = cv2.getValidDisparityROI(roi1,roi2,0,num,5)
    roi_mask[y:y+rh,x:x+rw] = True
    valid &= roi_mask
    detections = detector.detect(images[0])
    annotated = images[0].copy()
    for d in detections:
        x,y,bw,bh = d['x'],d['y'],d['w'],d['h']
        mask = valid[y:y+bh,x:x+bw]
        depths = points[y:y+bh,x:x+bw,2][mask]
        fraction = float(mask.mean()) if mask.size else 0.0
        distance = float(np.median(depths)) if len(depths) >= 30 and fraction >= 0.2 else None
        d.update(depth_m=distance,valid_depth_fraction=fraction,
                 depth_method='median_box_surface_optical_axis')
        label = f"{d['cls']} {distance:.2f} m" if distance is not None else f"{d['cls']} distance unavailable"
        cv2.rectangle(annotated,(x,y),(x+bw,y+bh),(0,255,255),2)
        cv2.putText(annotated,label,(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
    return annotated,detections


def main():
    ap=argparse.ArgumentParser()
    for name in ('left','right','calibration'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--model',default=str(Path(__file__).parent/'models'/'best.onnx'))
    ap.add_argument('--output',default='out/stereo')
    a=ap.parse_args()
    image,dets=process(cv2.imread(a.left),cv2.imread(a.right),a.calibration,Detector(a.model))
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output)+'.png',image):
        raise OSError('Could not save annotated image')
    Path(str(output)+'.json').write_text(json.dumps(dets,indent=2))
    print(json.dumps(dets,indent=2))

if __name__=='__main__':
    main()
