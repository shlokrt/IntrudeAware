# IntrudeAware Workflow

1. User uploads a video.
2. The application opens the stream and reads frames sequentially.
3. Each frame is resized and enhanced.
4. MOG2 estimates the foreground mask.
5. Morphological operations clean the mask.
6. Contours are converted into moving-object detections.
7. The centroid tracker assigns persistent IDs.
8. Lucas–Kanade optical flow estimates sparse motion.
9. The event detector checks whether a tracked object enters the restricted region.
10. The frame is annotated and written to an output video.
11. Summary metrics and event records are shown in the dashboard.


### Adaptive YOLO scheduling

The semantic detector supports adaptive frame skipping. The configured maximum skip is used for calm scenes, while the EWMA-smoothed centroid speed or restricted-zone occupancy shortens the desired inference interval. A short refresh cooldown enforces a minimum gap between YOLO runs so noisy motion does not cause one-frame-on/one-frame-off oscillation. Intermediate frames still pass through the existing tracker, optical-flow stage, restricted-zone event engine, and overlay.
