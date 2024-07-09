# 👁️🤖🖥️ IC - Pattern Extraction from Binary Images

## About the project

This projects use different approaches to extract patterns (noteheads and beams) from binary images, in this case music sheets.
The algorithms used to extract the patterns are: YOLOv8 (You Only Look Once), Ensemble between YOLOv8 and SAM (Segment Anything) and 
Cartesian Genetic Programming (CGP).


## Scripts for training

1. YOLOv8: yolov8_segmentation_ic.ipynb
2. YOLOv8 + SAM: YOLO_training.ipynb
3. CGP: cartesian-gp.ipynb

<strong>obs:</strong> whithin the jupyter notebooks there are instructions of how to train the algorithms and models.

## Scripts for running

1. YOLOv8 e YOLOv8 + SAM: main.py (and pipeline.py)
2. CGP: main_cgp.py (and CGP.py)

## Acknowledgements

- [Pedro Malandrin Klesse](https://www.github.com/Klesse)
- Emerson Carlos Pedrino
