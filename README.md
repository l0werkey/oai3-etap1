# OAI3 - Etap 1

A collection of 5 independent machine learning challenges covering computer vision, audio classification, NLP, and satellite image segmentation. Each task is self-contained in a Jupyter notebook with its own conda environment.

## Tasks

### 1. Filtry konwolucyjne (Convolutional Filters)
Learn a single universal 10×10 convolutional kernel to restore corrupted images using PyTorch's autograd.

**Directory:** `1_filtry_konwolucyjne/`

### 2. Klasyfikacja wieloetykietowa (Multi-label Classification)
Classify clothing images with multiple labels across 10 categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

- **Data:** 6,318 training / 702 validation / 780 test samples (168×168 px)
- **Metric:** Macro F1 score (0 pts at F1 ≤ 0.57, 100 pts at F1 ≥ 0.87)
- **Time limit:** 2.5 minutes on GPU

**Directory:** `2_klasyfikacja_wieloetykietowa/`

### 3. Szept czy krzyk? (Whisper or Scream)
Classify audio clips into 3 classes: normal speech (0), scream (1), whisper (2).

- **Data:** 2,400 training samples
- **Libraries:** librosa, torchaudio, scikit-learn

**Directory:** `3_szept_czy_krzyk/`

### 4. Zmiany semantyczne słów (Semantic Word Changes)
Binary classifier detecting whether word meanings shifted between 1900 and 1990 using pre-trained Word2Vec embeddings from both time periods.

- **Metric:** Balanced Accuracy (0 pts at ≤ 0.70, 100 pts at ≥ 0.87)
- **Constraints:** CPU-only, no internet access, 5-minute time limit
- **Allowed libraries:** numpy, pandas, scikit-learn, matplotlib, tqdm

**Directory:** `4_zmiany_semantyczne/`

### 5. Segmentacja multispektralna (Multispectral Segmentation)
Segment satellite terrain imagery using multispectral bands (visible + infrared). Includes data augmentation (rotation, scaling, noise).

**Directory:** `5_segmentacja_multispektralna/`

## Setup

Each task has its own `environment.yml`. Create the environment for the task you want to work on:

```bash
cd <task_directory>
conda env create -f environment.yml
conda activate <env_name>
jupyter notebook
```

## Requirements

- Python 3.11
- Conda (for environment management)
- CUDA-capable GPU recommended for tasks 1, 2, 3, and 5
- Task 4 is CPU-only

## Notebook Structure

Each notebook follows a consistent pattern:

| Function | Description |
|---|---|
| `define_model()` | Defines and returns the model |
| `train_solution(model)` | Trains the model on provided data |
| `predict_solution(model, input)` | Returns predictions for given input |
| `evaluate_algorithm(model, predict_fn, loader)` | Computes task-specific metrics and points |
