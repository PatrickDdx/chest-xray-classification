
# Chest X-ray Diagnosis and Grad-CAM Visualizer

This project provides an interactive **Streamlit application** for visualizing deep learning predictions and **Grad-CAM heatmaps** on chest radiographs from the **NIH ChestX-ray14** dataset. A fine-tuned **ResNet-50** multi-label classifier is used to detect thoracic pathologies and highlight relevant image regions that contributed to each prediction.

Users can upload a chest X-ray image and:
- Receive predicted disease labels with confidence scores
- View Grad-CAM explanations for each class
- Interpret the model’s focus across a visual grid layout

---

## Example Output

![cam-grid-example](app/assets/example_image.png)

---

## Features

- Upload chest X-ray images in PNG or JPEG format
- Predict probabilities for 14 common thoracic conditions
- Visual Grad-CAM overlays for class-specific attention
- 3×5 grid display with original image and heatmaps
- Clean, responsive interface (Streamlit-based)
- Designed for local use or online deployment

---

## Model Overview

| Property        | Value                     |
|----------------|---------------------------|
| Architecture    | ResNet-50                 |
| Classification  | Multi-label (14 classes)  |
| Loss Function   | BCEWithLogitsLoss         |
| Prediction Threshold | 0.25                |
| Dataset         | NIH ChestX-ray14          |

---

## Project Structure

```
chestxray_cam_app/
├── app/
│   ├── streamlit_app.py
│   ├── model_utils.py
│   ├── cam_utils.py
│   ├── transforms.py
│   └── assets/
│       └── example_image.png
├── models/
│   ├── best_model.pth
│   └── mlb_classes.pkl
├── training/
│   ├── train.py
│   └── training_notebook.ipynb
├── requirements.txt
└── README.md

```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/PatrickDdx/chest-xray-classification.git
cd chest-xray-classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the App

```bash
streamlit run app/streamlit_app.py
```

---

## Requirements

Key dependencies:

- `torch`, `torchvision`
- `streamlit`
- `pillow`
- `numpy`
- `matplotlib`
- `joblib`
- `pytorch-grad-cam`
- `scikit-learn` (for label binarization)

All packages can be installed via the provided `requirements.txt`.

---

## Model Weights

To run the app:

- Place your trained model file `best_model.pth` in the `models/` directory.
- Include the label binarizer (`mlb_classes.pkl`) in the same folder.

Update file paths in `streamlit_app.py` if your structure differs.

---

## Training the Model

The model was trained on the NIH ChestX-ray14 dataset using a fine-tuned ResNet-50 architecture.

- [`training/train.py`](training/train.py): Standalone script for model training
- [`training/training_notebook.ipynb`](training/training_notebook.ipynb): Jupyter notebook for exploratory development and visualization

Due to the dataset’s size and licensing, raw images are not included in this repository. You can download the ChestX-ray14 dataset from the [official NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC), then follow the instructions in the notebook to preprocess and train locally.

Update dataset paths in the notebook or script as needed before training.

Example training command:

```bash
python training/train.py \
  --csv archive/Data_Entry_2017.csv \
  --image_dir archive/all_images \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --workers 4 \
  --patience 5 
```

---

## Author

**Patrick Linke**  
GitHub: [@PatrickDdx](https://github.com/PatrickDdx)

---

## Disclaimer

This application is intended for **educational and research purposes only**. It is **not** certified for clinical use or diagnostic decision-making.

---

## Acknowledgments

- **NIH Clinical Center** for the ChestX-ray14 dataset  
  Sample chest X-ray images used in this project are from the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC), provided by the U.S. National Library of Medicine and the National Institutes of Health Clinical Center.  
  These images are included for **non-commercial, educational demonstration purposes only** and are **not** intended for clinical use.
- Open-source contributors to **PyTorch**, **Streamlit**, and **Grad-CAM** libraries

---

## Feedback & Contributions

Contributions, issues, and feedback are welcome. If you find the project useful, please consider starring ⭐ the repository or sharing it with others interested in medical AI and interpretability.
