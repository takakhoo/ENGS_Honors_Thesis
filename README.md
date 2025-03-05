# Automatic Music Mastering Using Deep Learning

**Honors Thesis Project for ENGS 88 at Dartmouth College**  
**Primary Advisor:** Peter Chin  
**Secondary Consultant:** Michael Casey

## Overview

This project aims to develop an automatic music mastering system using deep learning techniques. Music mastering is a complex, creative process that involves subtle adjustments to dynamics, frequency balance, and spatial effects such as reverb and echo. The goal of this research is to build a neural network model that, when given an input audio track, can generate a "mastered" version with improved sonic qualities comparable to professional masters.

The project involves two main stages:
1. **Data Generation:**  
   Creating a large paired dataset by artificially "demastering" high-quality audio tracks. This is achieved by applying a series of realistic degradations (EQ, gain adjustment, echo, reverb, and compression) to the original mastered tracks.
2. **Neural Network Training:**  
   Training a deep neural network on these paired examples so that it learns to "remaster" degraded audio. The model is expected to take any random input and generate a version with enhanced dynamics, spectral balance, and spatial effects.

## Dataset

We use the **FMA (Free Music Archive) Small** dataset as the basis for our work. The FMA dataset is an open, Creative Commons–licensed collection of music tracks organized by genre. For this project, we work with the FMA small subset, which contains 8,000 30-second clips. The original FMA data is stored in the folder structure:  
```
project_root/
  data/
    raw/
      fma_small/
        000/
        001/
        002/
        ...
```
We process songs from selected subfolders (e.g., "001", "002", "003") to generate paired data.

We will eventually use much larger datasets when it comes time to train the network

## Data Generation – Demastering Pipeline

The demastering pipeline is implemented in a separate Jupyter Notebook (e.g., `demastering.ipynb`) located in the `notebooks/` folder. This notebook performs the following steps:

1. **Loading Audio:**  
   The system loads each MP3 file from the FMA dataset.

2. **Applying Degradations:**  
   A set of realistic degradations is applied sequentially to the entire track:
   - **EQ:** A peaking filter is applied with randomized parameters (center frequency, Q-factor, and gain).
   - **Gain Adjustment:** Overall level changes are applied.
   - **Echo:** A delayed, attenuated copy of the signal is added to simulate echo.
   - **Reverb:** The audio is convolved with a synthetic impulse response to simulate reverb.  
     *Note: The impulse response length is dynamically determined based on the sampling rate and is saved for each track.*
   - **Compression:** A simple dynamic range compressor is applied with randomized threshold, ratio, and makeup gain.

3. **Output Generation:**  
   For each processed track, the following are saved:
   - **Audio Files:** Original (mastered) and degraded (demastered) audio are saved as WAV files.
   - **Spectrogram Images:** Both the original and modified audio spectrograms are generated with annotated parameters.
   - **Parameter Files:** A text file containing the random parameters used for each track is generated for documentation.

   The outputs are organized in the following directory structure (located in the project root under the `experiments/` folder):

   ```
   experiments/
     output_full/
       output_audio/           # Original and degraded audio files (WAV)
       output_spectrograms/    # Spectrogram images (PNG)
       output_txt/             # Text files with processing parameters
   ```

## Neural Network Training Pipeline

Once the paired dataset is generated, the next phase is to train a neural network to learn how to "remaster" audio. The planned architecture and training pipeline are as follows:

### Features and Input Representation

- **Input Features:**  
  The model will primarily work on time–frequency representations (spectrograms or mel-spectrograms) computed from the degraded audio tracks. These representations capture the important spectral and temporal information of the audio.
  
- **Output Representation:**  
  Depending on the approach, the network can be designed in one of two ways:
  - **Direct Audio Restoration:**  
    A network (e.g., U-Net) maps the degraded spectrogram directly to a "clean" or mastered spectrogram.
  - **Parameter Prediction:**  
    The network predicts a set of processing parameters (EQ, reverb, echo, compression) that are then used in a differentiable processing chain to reconstruct the mastered audio.

### Neural Network Architecture

- **Model Type:**  
  We plan to explore a convolutional neural network (CNN) architecture, with a U-Net–style encoder–decoder design. The U-Net architecture is well-suited for tasks that require both global context and detailed local reconstruction.
  
- **Encoder:**  
  The encoder will extract hierarchical features from the input spectrogram using a series of convolutional and pooling layers.
  
- **Decoder:**  
  The decoder will reconstruct the mastered spectrogram using upsampling layers, and skip connections from the encoder will help preserve fine details.

- **Loss Functions:**  
  - **Spectral Loss:** The L1 or L2 difference between the predicted and target (original) spectrograms.
  - **Time-Domain Loss:** An optional loss computed on the waveforms, though perceptual quality is often better captured in the spectral domain.
  - **Perceptual or Adversarial Losses:** To encourage natural-sounding output, additional losses (possibly using a pretrained network) may be incorporated.

- **Optimization:**  
  The network will be trained using an optimizer such as Adam with a carefully tuned learning rate. Training will be validated on a separate validation set to avoid overfitting.

### Accessing the Paired Data

A custom PyTorch `Dataset` class (e.g., in `src/dataset.py`) will be implemented to load the paired data from the `experiments/paired/` folder. This class will:
- Read the original and degraded WAV files.
- Compute (or load precomputed) spectrograms for training.
- Optionally, load the parameter text files if needed.

### Training and Experimentation

- **Training Script:**  
  The training code (in `src/train.py`) will load the dataset, initialize the model (from `src/models.py`), and run the training loop. Model checkpoints and training logs will be stored in dedicated folders (e.g., `models/` and `experiments/logs/`).

- **Evaluation:**  
  Separate scripts (e.g., `src/evaluate.py`) will be used to test the model on unseen data and conduct subjective and objective evaluations.

## Project Organization

The recommended repository structure is as follows:

```
project_root/
│
├── data/
│   ├── raw/                 # Raw data (FMA small, etc.)
│   └── processed/           # (Optional) Additional processed datasets
│
├── experiments/             # Experiment outputs, logs, and configuration files
│   ├── output_full/         # Generated paired data from the demastering notebook
│   │   ├── output_audio/
│   │   ├── output_spectrograms/
│   │   └── output_txt/
│   └── logs/                # Training logs, experiment configurations, etc.
│
├── notebooks/               # Jupyter notebooks for data processing, exploration, and experiments
│   ├── demastering.ipynb    # Notebook that generates paired data from the FMA dataset
│   └── exploration.ipynb    # For analysis and visualization
│
├── src/                     # Source code for neural network training and evaluation
│   ├── dataset.py           # Custom PyTorch Dataset for loading paired data
│   ├── models.py            # Definition of neural network architectures (e.g., U-Net)
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation and inference code
│   └── utils.py             # Utility functions
│
├── models/                  # Saved trained model weights
│
├── README.md                # This file.
└── requirements.txt         # List of dependencies and versions.
```

### How Files Interact

- **Data Generation:**  
  The `notebooks/demastering.ipynb` notebook generates paired data (original and degraded) and stores the outputs in `experiments/output_full/`.
  
- **Training:**  
  The training code in `src/train.py` will load data from the paired dataset (or process data on-the-fly using the custom dataset class in `src/dataset.py`).
  
- **Evaluation and Analysis:**  
  Additional notebooks (like `notebooks/exploration.ipynb`) and scripts in `src/evaluate.py` will be used to analyze results and evaluate model performance.

- **Modularity:**  
  Keeping the data generation and neural network training in separate modules (and notebooks) makes the project easier to manage and understand. The demastering notebook is a standalone tool to generate paired data, while the neural network training code is modularized under `src/`.

---

## Conclusion

This project leverages a two-stage approach:
1. **Paired Data Generation:**  
   Create a large dataset of original and artificially degraded (demastered) audio pairs using the FMA dataset and a series of audio processing functions.
2. **Neural Network Training:**  
   Train a deep learning model (e.g., using a U-Net architecture) to learn the mapping from degraded input to a professionally mastered output.

The repository is organized into clearly defined subfolders for data, experiments, notebooks, and source code. This structure not only supports reproducibility and scalability (to the full FMA dataset) but also ensures that the work is cleanly documented for your honors thesis at Dartmouth College.

Feel free to further modify this README as your project evolves. Let me know if you have any questions or need additional details!