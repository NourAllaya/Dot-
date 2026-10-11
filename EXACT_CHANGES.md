# Exact changes for the uploaded C.R.A.T.E.R. project

## What was found
The archive has no remote detection API. `perception.py` used local colour segmentation. `ground.py` serves `/api/state` and `/api/truth` to the dashboard; keep these endpoints. `rover.py` calls `perception.detect(frame)`, then `localize()` and sends compact UDP telemetry.

Your actual `best.onnx` is YOLO26n, input float32 `[1,3,640,640]`, output `[1,6,8400]`, end2end=False. Classes are **0: rock; 1: rock_shadow**. It cannot detect craters or ice. Train/export weights with those classes to add them; do not rename rock_shadow to crater.

## Files to replace/add
- REPLACE `perception.py`: ONNX Runtime CPU inference, BGR to RGB, letterbox, normalization, decode raw xywh plus class scores, class-specific NMS, restore original image coordinates. Preserves the existing detection dictionary and 0–15 telemetry confidence. `detect()` omits rock_shadow because the existing radio protocol has no shadow class.
- ADD `models/best.onnx`: the exact uploaded weights, included in this package.
- ADD `requirements.txt`: numpy, OpenCV headless, ONNX Runtime. No PyTorch needed for this inference path.
- ADD `stereo_pair.py`: independent calibrated stereo image command; detection on the rectified left image, SGBM disparity, metric reprojection, distance annotation and JSON.
- ADD `test_onnx_stereo.py`: actual-model smoke test, synthetic stereo metric test, missing-depth test.
- KEEP `common.py`, `localize.py`, `rover.py`, `telemetry.py`, `ground.py`, `dashboard.html`, `odometry.py`, `sim.py`, `link.py`, `run_demo.py` unchanged.

## Install and run (Windows terminal, inside crater)
```powershell
python -m pip install -r requirements.txt
python test_onnx_stereo.py
python run_demo.py --duration 60 --hold
```
Open http://localhost:8000. The generated scene is synthetic, whereas these weights were trained on a different image distribution. Empty detections are possible. The previous `tests.py` end-to-end assertions about colour-segmentation precision, hazard counts and 30 FPS do not validate these new weights; they need a labeled held-out real dataset and measured hardware timing. Packet and projection tests are unchanged.

## Run on TWO images with distances
```powershell
python stereo_pair.py --left data/left.png --right data/right.png --calibration data/stereo_calibration.npz --output out/pair
```
Creates `out/pair.png` (boxes and distance labels) and `out/pair.json` (class, box, score, depth_m, valid_depth_fraction). Coordinates refer to the RECTIFIED left image. The model runs once on the left image; the right image supplies stereo correspondence, not another set of box centers.

## Required real camera calibration
`stereo_calibration.npz` must contain:
- `K1`, `K2`: 3x3 camera intrinsic matrices.
- `D1`, `D2`: lens distortion vectors.
- `R`: 3x3 rotation from left camera coordinates to right camera coordinates.
- `T`: length-3 translation from left to right coordinates, **in metres**; for parallel cameras with the right camera physically to the right, T is [-baseline_m,0,0].
- `image_size`: [width,height] matching both input images.

Generate these from synchronized checkerboard image pairs using camera and stereo calibration. Do not copy the simulator's CAM constants to a real rig. No real calibration or real stereo pair was supplied, so none is fabricated here.

The implementation supports horizontal stereo with positive disparity. Images must show the same scene from the calibrated left/right cameras, preferably simultaneously. Arbitrary images or two frames from a moving camera do not give metric distance without known camera motion/scale and a separate triangulation pipeline.

`depth_m` is a robust median optical-axis surface depth inside a detection box, not distance to the crater center/rim, crater excavation depth, or a guaranteed obstacle distance. Background inside boxes can bias it. Texture, occlusion, shadows and matching errors affect it. Insufficient valid correspondence yields null and “distance unavailable”. Validate on your real rig against measured distances before using it for navigation.

## Stereo versus rover mapping
This patch adds a reviewable two-image command; it does not invent a stereo rover camera. The rover demo still renders ONE synthetic image and uses `localize.py`'s flat-ground projection with simulator calibration. To use stereo in the rover, supply synchronized camera capture, calibrated camera-to-rover extrinsics, and a localization adapter transforming measured camera XYZ to rover coordinates before `rover_to_world()`. HazardMap needs world x/y/r and conf, not just depth_m. That integration cannot be correctly completed from the supplied monocular simulator alone.

## Validation performed
Actual ONNX session load and inference; verified model classes/input/output; known-depth synthetic stereo (3.125 m) and blank-image rejection; existing packet roundtrip, all single-bit CRC rejection, ACK roundtrip and projection tests; short rover/dashboard demo. No real-image detection accuracy or real-camera distance accuracy is claimed.
