# IntrudeAware — Project Statement

## Problem statement

Manual monitoring of continuous video streams is repetitive and makes it difficult to consistently record object movement and events. IntrudeAware provides a modular computer-vision pipeline that converts a video stream into annotated frames, object tracks, motion measurements, and rule-based alerts.

## Scope

The MVP focuses on uploaded video files. It performs preprocessing, motion-based object detection, object tracking, optical-flow analysis, restricted-zone monitoring, and visualization. It is a prototype for academic evaluation, not a safety-critical surveillance product.

## Target users

- Computer Vision students
- Academic demonstrators
- Researchers prototyping video-analysis pipelines
- Developers learning classical video analytics

## High-level features

- Video upload and processing
- Image enhancement and edge detection
- Moving-object detection
- Object tracking with persistent IDs
- Optical-flow visualization
- Restricted-zone event detection
- Metrics and event summary
- Annotated video output
