# Brain Tumour Classification Using Vision Transformer (ViT)

A deep learning project that uses a pretrained Vision Transformer (ViT) to classify brain MRI images into four categories: Glioma, Meningioma, No Tumour, and Pituitary. The project includes a training notebook, a pretrained model hosted on Hugging Face, evaluation results, and a Google Colab demo for image prediction.

This project is intended for educational and research purposes. It is not a clinically validated diagnostic system.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model and Dataset](#2-model-and-dataset)
3. [Dataset Classes and Distribution](#3-dataset-classes-and-distribution)
4. [Model Architecture](#4-model-architecture)
5. [Dataset Preparation and Preprocessing](#5-dataset-preparation-and-preprocessing)
6. [Training Configuration](#6-training-configuration)
7. [Model Performance](#7-model-performance)
8. [Repository Structure](#8-repository-structure)
9. [Installation and Environment Setup](#9-installation-and-environment-setup)
10. [Running the Demo](#10-running-the-demo)
11. [Retraining the Model](#11-retraining-the-model)
12. [Understanding the Evaluation Assets](#12-understanding-the-evaluation-assets)
13. [How the Project Works](#13-how-the-project-works)
14. [Limitations and Responsible Use](#14-limitations-and-responsible-use)
15. [Future Improvements](#15-future-improvements)
16. [Credits and References](#16-credits-and-references)

## 1. Project Overview

Brain tumour classification from MRI images is an important research application of medical image analysis. This project explores the use of transfer learning with a Vision Transformer to classify MRI images into four predefined categories.

The model uses a pretrained ViT architecture and fine-tunes it on the curated brain MRI dataset.

### Key Features

- Four-class brain MRI image classification.
- Transfer learning using a pretrained Vision Transformer.
- Training and evaluation through Google Colab notebooks.
- Performance analysis using accuracy, precision, recall, F1-score, and a confusion matrix.
- A pretrained model available on Hugging Face.
- An interactive notebook-based demo that accepts an MRI image uploaded by the user.
- A visual attribution overlay to help inspect image regions that influence the model's prediction.

## 2. Model and Dataset

The model and dataset are hosted separately on Hugging Face.

| Resource | Link |
|---|---|
| Trained model | [Brain Tumour ViT on Hugging Face](https://huggingface.co/Sohaaam/Brain_Tumour_ViT) |
| Curated dataset | [Brain Tumour Curated Dataset](https://huggingface.co/datasets/Sohaaam/Brain_Tumour_Curated) |
| Training notebook | [`notebooks/VIT_FINAL.ipynb`](notebooks/VIT_FINAL.ipynb) |
| Demo notebook | [`notebooks/VIT_DEMO.ipynb`](notebooks/VIT_DEMO.ipynb) |

The model repository contains the pretrained model files and configuration required for inference. The dataset repository provides access to the images and dataset information.

Before using or redistributing the data, review the dataset card and the licenses and conditions applicable to its original sources.

## 3. Dataset Classes and Distribution

The dataset contains **40,420 images** across four classes.

| Class | Number of images |
|---|---:|
| Glioma | 21,582 |
| Meningioma | 12,041 |
| No Tumour | 4,089 |
| Pituitary | 2,708 |
| **Total** | **40,420** |

### Class descriptions

- **Glioma:** A category of tumours that arise from glial cells or related cells in the central nervous system.
- **Meningioma:** A tumour that develops from the meninges, the membranes surrounding the brain and spinal cord.
- **No Tumour:** Images labelled as not showing a tumour according to the dataset labels.
- **Pituitary:** A category of tumours arising in or around the pituitary gland.

These categories represent the labels used by the dataset and model. They do not represent a complete list of brain tumour types.

The dataset counts above describe the overall dataset. The training notebook creates separate training, validation, and test sets for model development and evaluation.

## 4. Model Architecture

The project uses the following pretrained architecture:

**`google/vit-base-patch16-224-in21k`**

Vision Transformer (ViT) applies the transformer architecture to image classification by representing an image as a sequence of patches.

### Architecture details

| Component | Configuration |
|---|---|
| Model family | Vision Transformer |
| Pretrained checkpoint | `google/vit-base-patch16-224-in21k` |
| Input size | 224 × 224 pixels |
| Input channels | 3 (RGB) |
| Patch size | 16 × 16 pixels |
| Classification classes | 4 |
| Training approach | Transfer learning and fine-tuning |

### How ViT works

1. **Image preprocessing:** Each MRI image is converted to RGB and resized to 224 × 224 pixels.
2. **Patch extraction:** The image is divided into non-overlapping 16 × 16 patches.
3. **Patch embeddings:** Each patch is converted into a numerical representation, called an embedding.
4. **Positional information:** Positional embeddings help the transformer retain information about patch order and location.
5. **Transformer encoder:** Self-attention and feed-forward layers process the sequence of patch representations.
6. **Classification head:** The resulting representation is used to predict scores for the four classes.
7. **Prediction:** The class with the highest output score is returned as the model's predicted category.

The model starts with pretrained weights learned from a large image dataset and is fine-tuned for the brain MRI classification task.

## 5. Dataset Preparation and Preprocessing

The training pipeline prepares images before passing them to the model.

### Image preprocessing

- Convert images to RGB format.
- Resize images to 224 × 224 pixels.
- Convert image data into tensors.
- Normalize each RGB channel using a mean of `0.5` and a standard deviation of `0.5`.

For an input channel value scaled to the range `[0, 1]`, the normalization is:

`normalized_value = (pixel_value - 0.5) / 0.5`

This transforms the input range to approximately `[-1, 1]`.

### Class mapping

The model uses the following numerical labels:

| Class | Label |
|---|---:|
| Glioma | 0 |
| Meningioma | 1 |
| No Tumour | 2 |
| Pituitary | 3 |

These mappings must remain consistent during training, evaluation, and inference.

### Dataset splitting

The reported split contains:

| Subset | Number of images |
|---|---:|
| Training | 32,336 |
| Validation | 4,042 |
| Test | 4,042 |
| **Total** | **40,420** |

- **Training set:** Used to update the model's parameters.
- **Validation set:** Used to monitor performance and select the best checkpoint.
- **Test set:** Used to evaluate the selected model on data not used for training or checkpoint selection.

The test set should not be repeatedly used to tune model settings, as this can make the final evaluation less reliable.

### Class imbalance

The dataset contains substantially more Glioma images than Pituitary images. The training pipeline uses class-weighted cross-entropy loss to give additional importance to underrepresented classes.

This can help address class imbalance, but it does not guarantee equal performance across all classes.

## 6. Training Configuration

The model was fine-tuned using the following configuration.

| Parameter | Value |
|---|---|
| Architecture | ViT-Base, Patch 16, 224 |
| Pretrained weights | `google/vit-base-patch16-224-in21k` |
| Number of epochs | 3 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Loss function | Class-weighted cross-entropy |
| Mixed precision | FP16 where supported |
| Checkpoint selection | Best validation macro-F1 |
| Training environment | Google Colab with GPU acceleration |

### Training process

1. Load the curated dataset.
2. Preprocess the images and create the data splits.
3. Load the pretrained ViT checkpoint.
4. Configure the classification head for four output classes.
5. Calculate class weights for the imbalanced training data.
6. Train the model for three epochs using AdamW and the configured learning rate.
7. Evaluate the model on the validation set after training epochs.
8. Save the checkpoint with the best validation macro-F1.
9. Evaluate the selected checkpoint on the test set.
10. Save the classification report, predictions, confusion matrix, and other evaluation artifacts.

The number of epochs and other settings describe the recorded training run. Retraining may produce different results because of changes in the dataset, runtime, random seeds, package versions, or training configuration.

## 7. Model Performance

The following metrics were recorded for the trained model.

### Overall results

| Metric | Validation set | Test set |
|---|---:|---:|
| Accuracy | 88.25% | 87.58% |
| Macro-F1 | 91.13% | 90.38% |
| Loss | — | 0.2442 |

Accuracy measures the proportion of correctly classified images. Macro-F1 calculates the F1-score for each class and then averages the scores equally, making it useful when class sizes differ.

### Test-set classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Glioma | 86.51% | 91.57% | 88.97% | 2,158 |
| Meningioma | 84.18% | 75.58% | 79.65% | 1,204 |
| No Tumour | 98.71% | 93.64% | 96.11% | 409 |
| Pituitary | 93.77% | 100.00% | 96.79% | 271 |
| **Macro average** | **90.79%** | **90.20%** | **90.38%** | **4,042** |
| **Weighted average** | **87.54%** | **87.58%** | **87.44%** | **4,042** |

### Interpreting these results

- The model achieved 87.58% test accuracy on the recorded test split.
- The macro-F1 score of 90.38% summarizes performance across all four classes with equal class weighting.
- Meningioma has lower recall than the other classes, meaning a larger proportion of its test examples were missed by the model.
- Pituitary achieved 100% recall on this test split, but this does not guarantee perfect performance on new MRI images.

These results apply only to the evaluated dataset and split. They should not be interpreted as evidence of clinical accuracy or generalization to all MRI scanners, hospitals, populations, or imaging protocols.

## 8. Repository Structure

```text
BRAIN-TUMOUR-CLASSIFICATION-USING-VIT/
├── assets/
│   ├── class_metrics.png
│   ├── confusion_matrix.png
│   ├── demo_prediction1.png
│   └── demo_prediction2.png
├── docs/
│   ├── evaluation.txt
│   └── requirements.txt
├── notebooks/
│   ├── VIT_FINAL.ipynb
│   └── VIT_DEMO.ipynb
├── .gitignore
└── README.md
```

### Main files

- `notebooks/VIT_FINAL.ipynb`: Main notebook for dataset preparation, model training, validation, testing, and saving evaluation artifacts.
- `notebooks/VIT_DEMO.ipynb`: Notebook for loading the trained model, uploading an MRI image, and viewing a prediction and attribution overlay.
- `docs/evaluation.txt`: Text report containing the recorded model configuration and evaluation results.
- `docs/requirements.txt`: Python packages used by the project.
- `assets/`: Saved figures and example outputs for documenting the model's performance and demo.

Model checkpoints and full datasets are not intended to be committed to GitHub. The trained model is hosted separately on Hugging Face.

## 9. Installation and Environment Setup

The demo and full training workflow are designed for Google Colab. A local environment can also be used for package installation and supported inference workflows, but loading a large model may require substantial memory.

### 9.1 Clone the repository

```bash
git clone https://github.com/sohamkharatmal/BRAIN-TUMOUR-CLASSIFICATION-USING-VIT.git
cd BRAIN-TUMOUR-CLASSIFICATION-USING-VIT
```

### 9.2 Create a virtual environment (recommended)

On Linux or Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `venv` is unavailable, install it using your system package manager.

### 9.3 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r docs/requirements.txt
```

The requirements file includes the principal libraries used by the project, such as PyTorch, Transformers, Hugging Face Hub, Pillow, NumPy, Matplotlib, scikit-learn, and pandas.

For local PyTorch installation, choose a build appropriate for your system and hardware using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). GPU training is recommended for practical training times.

**Note:** Installing the packages locally does not automatically execute the notebooks. The training and demo workflows are contained in the Jupyter notebooks.

## 10. Running the Demo

The demo allows a user to upload a brain MRI image and obtain a prediction from the pretrained model. The workflow runs in Google Colab; no separate Python demo script is required.

### 10.1 Open the demo notebook

Open [`notebooks/VIT_DEMO.ipynb`](notebooks/VIT_DEMO.ipynb) in the repository and select **Open in Colab**, or use this direct link:

[Open the demo notebook in Google Colab](https://colab.research.google.com/github/sohamkharatmal/BRAIN-TUMOUR-CLASSIFICATION-USING-VIT/blob/main/notebooks/VIT_DEMO.ipynb)

If the direct link is unavailable, open [Google Colab](https://colab.research.google.com/), select the GitHub tab, and enter the repository URL.

### 10.2 Select the runtime

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select a GPU runtime if one is available.
4. Save the runtime settings.

GPU acceleration is helpful but may not be necessary for a single-image prediction.

### 10.3 Run the notebook

Execute the notebook cells in order, from top to bottom. The notebook should install or import the necessary libraries, load the model from Hugging Face, and prepare the prediction pipeline.

If prompted for access or authentication, follow the instructions displayed by Hugging Face or Colab. Do not place personal access tokens directly in notebook code or commit them to GitHub.

### 10.4 Upload an MRI image

When the notebook reaches its image-upload step:

1. Click the upload control provided by the notebook.
2. Select a brain MRI image from your computer.
3. Wait for the notebook to preprocess the image and run inference.
4. Inspect the predicted class, confidence scores if displayed, and the attribution visualization if generated.

The model returns one of four labels: **Glioma, Meningioma, No Tumour, or Pituitary**.

For a meaningful demonstration, use an MRI image that is suitable for the model's expected input format. The output is a prediction based on learned patterns, not confirmation of the actual medical condition.

### 10.5 Understanding the attribution overlay

The demo may display a visual attribution overlay intended to highlight image regions that influenced the model's output.

This overlay is an interpretability aid. It is **not a verified tumour boundary, segmentation mask, or clinical localization result**. Highlighted regions should not be treated as proof of where a tumour is present.

## 11. Retraining the Model

Use `notebooks/VIT_FINAL.ipynb` to reproduce or modify the training workflow.

[Open the final training notebook in Google Colab](https://colab.research.google.com/github/sohamkharatmal/BRAIN-TUMOUR-CLASSIFICATION-USING-VIT/blob/main/notebooks/VIT_FINAL.ipynb)

### 11.1 Prepare the runtime

1. Open the final notebook in Google Colab.
2. Select a GPU runtime if available.
3. Run the initial setup and import cells.
4. Follow the notebook's instructions for accessing and preparing the dataset.

### 11.2 Prepare the dataset

Run the cells that load the dataset, inspect its classes, preprocess images, and create the training, validation, and test splits.

Verify that:

- All four expected classes are present.
- The class-to-index mapping matches the mapping documented in this README.
- The split sizes are correct for the dataset version being used.
- Duplicate or near-duplicate images do not unintentionally appear across different splits.

The dataset may change over time. If its contents or class counts differ, document the actual counts used in the new training run rather than assuming they are unchanged.

### 11.3 Configure and train the model

The notebook uses the pretrained `google/vit-base-patch16-224-in21k` architecture and fine-tunes it for four-class classification.

Review the training configuration before executing the training cells:

- Epochs: `3`
- Batch size: `16`
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Optimizer: `AdamW`
- Loss: class-weighted cross-entropy

Run the training cells and monitor the validation metrics. The notebook selects the best checkpoint according to validation macro-F1 rather than assuming that the final epoch is always the best.

To experiment, you can adjust the learning rate, batch size, number of epochs, or other training settings. Record any changes so that the resulting metrics can be reproduced and compared fairly.

### 11.4 Evaluate the trained checkpoint

Run the evaluation cells to calculate test metrics and generate the classification report, confusion matrix, and prediction artifacts.

Use the validation set for model selection. Keep the test set separate until the final evaluation of the selected model.

### 11.5 Save and publish the model

The notebook saves the best checkpoint and evaluation outputs. To make a newly trained model available to the demo:

1. Save the selected model and its configuration.
2. Confirm that the class mapping and image preprocessing match the demo.
3. Upload the model files to a Hugging Face model repository if desired.
4. Update the model-loading reference in the demo notebook to point to the new repository or checkpoint revision.
5. Run the demo again and verify that its predictions and output labels work correctly.

The full ViT checkpoint is large. Keep model weights out of the GitHub repository unless there is a specific reason to store them there. Hugging Face is used to host the trained model separately.

## 12. Understanding the Evaluation Assets

The `assets/` folder contains figures and example outputs generated or collected for documenting the project.

### `assets/class_metrics.png`

A visual summary of class-wise evaluation metrics. It helps compare how the model performs across the four output classes. Use it alongside the numerical classification report, since a plot may not show every value precisely.

### `assets/confusion_matrix.png`

A confusion matrix comparing the actual labels with the model's predicted labels.

- Correct predictions appear on the main diagonal.
- Off-diagonal values indicate misclassifications.
- Rows and columns should be interpreted using the class order shown in the evaluation notebook.

The confusion matrix helps identify which classes the model tends to confuse.

### `assets/demo_prediction1.png` and `assets/demo_prediction2.png`

Example outputs illustrating the demo's prediction workflow. These are illustrative examples, not independent proof of model reliability.

### `docs/evaluation.txt`

A text record of the training configuration and evaluation results. It provides a convenient reference without requiring the reader to execute the notebook.

### `docs/requirements.txt`

The list of Python dependencies required by the project.

## 13. How the Project Works

The overall workflow is:

1. **Dataset:** Brain MRI images and their labels are loaded from the curated dataset.
2. **Preprocessing:** Images are converted, resized, normalized, and assigned numerical labels.
3. **Training:** The pretrained ViT is fine-tuned on the training split.
4. **Validation:** Validation metrics are used to select the best checkpoint.
5. **Testing:** The selected checkpoint is evaluated on the held-out test set.
6. **Model hosting:** The trained model is made available through Hugging Face.
7. **Demo inference:** A user uploads an MRI image in Google Colab; the notebook preprocesses it and obtains a class prediction.
8. **Interpretability:** An attribution overlay may help inspect image regions influencing the prediction.
9. **Evaluation artifacts:** Metrics, reports, and plots document the recorded results.

## 14. Limitations and Responsible Use

This project is an educational and research prototype. It has not been established as a clinically validated medical device.

- **Not for diagnosis:** Predictions must not be used to diagnose, rule out, or treat a medical condition.
- **Dataset limitations:** Performance depends on the composition, quality, labels, and source distribution of the dataset.
- **Generalization:** Results on this test split do not guarantee performance on images from other hospitals, scanners, patient populations, or imaging protocols.
- **Class imbalance:** Different class frequencies can influence model behavior and error rates.
- **Attribution limitations:** The overlay does not provide a verified tumour location or segmentation.
- **Image quality:** Unusual formats, poor-quality images, or images outside the training distribution may produce unreliable predictions.
- **Data privacy:** Do not upload identifiable or confidential medical images to third-party services unless you have appropriate authorization and understand the applicable privacy requirements.

A qualified medical professional must interpret MRI scans and make clinical decisions.

## 15. Future Improvements

Potential directions for further development include:

- Evaluating the model on an independent external dataset.
- Performing stronger duplicate and near-duplicate checks before splitting data.
- Comparing ViT against CNN-based architectures.
- Evaluating probability calibration and uncertainty estimates.
- Improving reproducibility by recording random seeds and package versions.
- Investigating explainability methods and their limitations.
- Expanding the evaluation to include additional relevant metrics and robustness tests.

## 16. Credits and References

- **Project repository:** [BRAIN-TUMOUR-CLASSIFICATION-USING-VIT](https://github.com/sohamkharatmal/BRAIN-TUMOUR-CLASSIFICATION-USING-VIT)
- **Trained model:** [Sohaaam/Brain_Tumour_ViT](https://huggingface.co/Sohaaam/Brain_Tumour_ViT)
- **Dataset:** [Sohaaam/Brain_Tumour_Curated](https://huggingface.co/datasets/Sohaaam/Brain_Tumour_Curated)
- **Pretrained architecture:** [Google ViT Base Patch16 224, IN21K](https://huggingface.co/google/vit-base-patch16-224-in21k)
- **Transformers documentation:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **PyTorch documentation:** [PyTorch](https://pytorch.org/docs/stable/index.html)

The dataset and pretrained architecture are credited to their respective sources. Please consult their documentation and licensing terms before reuse or redistribution.

---

**Disclaimer:** This project demonstrates machine-learning-based classification of brain MRI images for educational and research purposes only. Its outputs are not medical advice and must not be used as a substitute for professional medical evaluation.
