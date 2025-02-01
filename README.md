Taka Khoo ENGS Honors Thesis 2025

Overview
--------
AudioMaster is an innovative project that leverages deep learning and advanced
signal processing techniques to automate the mastering process for raw, unengineered
multitrack audio recordings. The goal is to transform poorly produced input
audio into a professionally mastered output, thereby reducing the time and expertise
required for traditional manual mixing and mastering.

This repository contains the code for a convolutional autoencoder that processes
audio spectrograms and reconstructs audio using the Griffin–Lim algorithm. It is part
of a BA Thesis project in Engineering Sciences at Dartmouth College.

Motivation & Objectives
-------------------------
- **Bridging Disciplines:** Combine machine learning, signal processing, and music production to create an AI-driven audio mastering pipeline.
- **Automation:** Develop a tool that can automatically enhance audio quality, making professional-level mastering accessible to amateur producers.
- **Research & Innovation:** Contribute to the field of audio processing by exploring novel neural network architectures and reconstruction methods.

Features
--------
- **Audio Preprocessing:** Convert raw audio files into normalized magnitude spectrograms using Librosa.
- **Neural Network Autoencoder:** Compress and reconstruct spectrograms using a convolutional architecture.
- **Audio Reconstruction:** Utilize the Griffin–Lim algorithm for phase reconstruction and conversion back to the audio domain.
- **Visualization:** Side-by-side comparisons of original and reconstructed spectrograms to evaluate model performance.
- **Modularity:** Clean, extendable codebase designed for further experimentation (e.g., improved architectures, additional loss functions).

Repository Structure
--------------------
AudioMaster/
├── README.txt               - This file.
├── audio_autoencoder.py     - Main source code for the audio mastering pipeline.
├── data/                    - Directory for input audio files (e.g., raw multitracks).
├── models/                  - (Optional) Directory for saving trained models and checkpoints.
├── outputs/                 - Directory where reconstructed audio and visualizations are saved.
└── requirements.txt         - List of required Python packages and dependencies.

Installation
------------
1. **Clone the Repository:**
   Open a terminal and run:
     git clone https://github.com/yourusername/AudioMaster.git
     cd AudioMaster

2. **Install Conda (Anaconda/Miniconda):**
   If you do not have Conda installed, download and install from:
     https://www.anaconda.com/products/distribution
   or
     https://docs.conda.io/en/latest/miniconda.html

3. **Create and Activate a Conda Environment:**
     conda create -n mastering_env python=3.9 -y
     conda activate mastering_env

4. **Install Dependencies:**
   Install all required packages by running:
     pip install -r requirements.txt

   The requirements.txt file includes:
     numpy
     matplotlib
     librosa
     torch
     torchaudio
     soundfile

Usage
-----
1. **Prepare Your Audio Data:**
   Place your input audio file (e.g., file.wav) in the repository root or update the file path in audio_autoencoder.py.

2. **Run the Main Script:**
     python audio_autoencoder.py
   The script will:
     - Load and preprocess the audio file.
     - Train a simple autoencoder (currently on a dummy dataset created by replicating the input).
     - Reconstruct the audio via the Griffin–Lim algorithm.
     - Display original and reconstructed spectrograms.
     - Save the final output audio as 'reconstructed_output.wav' in the project directory.

Future Work
-----------
- **Dataset Expansion:** Incorporate larger and more diverse datasets to train a robust model.
- **Architecture Enhancements:** Explore deeper network architectures and advanced loss functions (e.g., perceptual losses) for improved audio fidelity.
- **DAW Integration:** Develop plugins or interfaces (e.g., with Logic Pro) for seamless integration into professional music production workflows.
- **Publication & Dissemination:** The project is aimed for eventual publication in IEEE and presentation at relevant academic and industry conferences.

Contributing
------------
Contributions to AudioMaster are welcome! Please feel free to fork the repository, submit bug reports, propose enhancements, or contribute improvements via pull requests.

License
-------
This project is licensed under the [Your License Name] License. (Replace this with your actual license details.)

Acknowledgments
---------------
- **Professor Peter Chin:** Head Advisor - For guidance on machine learning and signal processing techniques.
- **Professor Michael Casey:** Secondary Advisor - For invaluable insights into music production and the application of AI in audio mastering.
- Special thanks to all contributors and the open-source community for their support and shared resources.

Contact
-------
For questions, feedback, or further information, please contact:
Taka Khoo
takakhoo@gmail.com or matthew.t.khoo.25@dartmouth.edu
Dartmouth College

================================================================================
