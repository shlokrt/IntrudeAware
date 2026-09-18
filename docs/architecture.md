# IntrudeAware Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
Video Input -> Preprocessing -> Background Subtraction -> Contour Detection
                                      |                    |
                                      v                    v
                              Optical Flow           Centroid Tracker
                                      \                    /
                                       \                  /
                                        v                v
                                         Event Analysis
                                               |
                                               v
                                    Visualization + Metrics
                                               |
                                               v
                                      Annotated Video
```

## Components

- `preprocessing/`: frame cleanup and enhancement.
- `detection/`: moving-region detection from foreground masks.
- `tracking/`: persistent object IDs and trails.
- `motion/`: background subtraction and optical flow.
- `analysis/`: event rules and statistics.
- `visualization/`: overlays and chart data.
- `utils/`: logging and video I/O.
- `app.py`: Streamlit application orchestration.


### YOLO configuration

The UI exposes YOLO26 model scale selection and inference device selection. The detector returns the same `Detection` objects as before, so the existing centroid tracker, restricted-zone event engine, dashboard, and overlay remain downstream components and do not change their interfaces.
