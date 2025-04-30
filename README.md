# Smart Focus AI Project

## Introduction


## Projet Structure

- **env/**: Virtual env.
- **notebooks/**: Notebooks for data exploration.
- **src/**: Main script to run pipeline & Source code (gaze & blink module, objetc & action module, display & overlay, logic).
- **models/**: extern rss (YOLO, ect.).
- **tests/**: Tests & Debug tests & Unit tests.
- **requirements.txt**: List of dependencies.
- **setup.py**: Packaging configuration script.
- **.gitignore**: Ignore venv, __pycache__, etc.
<!-- - **streamlit.py**: Streamlit application for interactive display. -->


## Libraries

* **mediapipe** → FaceMesh & iris.
* **numpy** → For mathematical operations and numerical array processing.
* **matplotlib** → For creating classic visualizations (charts, scatter plots, etc.).
* **ultralytics** → YOLOv8 for object detection.
* **torch & torchvision** → PyTorch motor (for ultralytics).
* **scikit-learn** → ML classifier.
* **pandas** → Logs, analysis, graph.
* **jupyterlab** → For developping and testing your analysis interactively in notebooks.
* **imutils** → OpenCV utilitaires.
* **pytestd** → For unit tests.
* **onnxruntime** → YOLO.
* **pynput** → Keyboard/Typing captation(≥ Py 3.10).
<!-- * **fpdf** → For generating PDF reports. -->
* **opencv-python** → Captur & video treatment.
* **pytest** → .
* **streamlit-webrtc** → .
* **streamlit** (optional) → For creating an interactive website.
* **pyobjc-framework-AVFoundation** (optional) → Force macOS à afficher la popup Caméra pour ce binaire.

To install Py, run :
```bash
brew install python@3.10
```

To install all these independencies, run :
```bash
pip install opencv-python mediapipe numpy ultralytics torch torchvision scikit-learn pandas matplotlib imutils onnxruntime pynput pytest streamlit streamlit-webrtc pyobjc-framework-AVFoundation
```

## Usage
<!-- Run to complete the pipeline and generate the visualizations :  -->

1. Install the idependencies :
```bash
   pip install -r requirements.txt
```
2. Run the main script :
```bash
    python main.py
```

3. Run the Streamlit application :
```bash
   streamlit run src/streamlit_app.py
```

## Features

- Generation or loading of a simulated dataset.
- Data preparation and cleaning (feature engineering, normalization).
- Anomaly detection using Isolation Forest (and potentially other algorithms).
- Static visualizations with Matplotlib / Seaborn.
- Interactive visualizations with Plotly.
- An interactive dashboard with Streamlit.
- Generation of a detailed PDF report.


## Tests

Unit tests are located in the `tests/` folder. To run them, execute : :
```bash
python -m unittest discover tests
```

## Setup

From the root project, instal the package in development mode by running :
```bash
pip install -e 
```

Copyright (c) [2025] GeeksterLab