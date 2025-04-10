# Automatic Music Mastering Using Deep Learning by Taka Khoo

**Honors Thesis Project for ENGS 88 at Dartmouth College**  
**Primary Advisor:** Peter Chin  
**Secondary Consultant:** Michael Casey  
**Student:** Taka Khoo

---

## Overview

This project tackles the challenging problem of **automatic music mastering**—the final stage in audio post-production where subtle adjustments to dynamics, frequency balance, spatial imaging, and spectral content are applied to create a polished, professional sound. Our goal is to develop a deep learning system that can transform a degraded, "demastered" audio track into a professionally mastered version, mimicking the effect of human audio engineers.

The project is structured into two primary stages:  
1. **Paired Data Generation (Demastering Pipeline):**  
   High-quality (mastered) audio tracks are artificially degraded using a sequence of signal processing operations. This creates paired datasets where the input is the degraded audio (demastered) and the target is the original mastered version.
2. **Neural Network Training and Evaluation:**  
   A deep neural network is trained to learn the mapping from degraded to mastered audio. Our network takes a time–frequency representation (specifically, mel-spectrograms) as input and outputs either a restored spectrogram or a set of processing parameters that drive a differentiable audio-processing pipeline.

*Note:* At present, our evaluation script (`src/evaluate.py`) uses a Griffin–Lim inversion method to convert mel spectrograms back to audio. More advanced vocoders (such as MelGAN) are under consideration for future work.

---

## Current System Components

### 1. Demastering Pipeline & Data Generation

#### Data Source

- **Dataset:** We use the **FMA Small** dataset, which contains thousands of 30-second music clips under Creative Commons licenses. This serves as a basis for creating our paired dataset.
- **Directory Structure:**  



project_root/ data/ raw/ # Raw FMA tracks (organized by folders, e.g., 001, 002, etc.) experiments/ output_full/ output_audio/ # Paired original and degraded WAV files output_spectrograms/ # Spectrogram images (PNG) of both versions output_txt/ # Text files describing the degradations (e.g., EQ, compression parameters)

#### Processing Steps

1. **Audio Loading and Clipping:**  
 Each track is loaded (mono) and clipped to the range [-1.0, 1.0] to ensure safe processing.

2. **Mel Spectrogram Computation:**  
 The mel spectrogram is computed using:
 
 \[
 \text{mel\_spec} = \text{LibrosaFeature.melspectrogram}(y, \text{sr}=44100, \text{n\_fft}=2048, \text{hop\_length}=512, \text{n\_mels}=128)
 \]
 
 Then, conversion to a decibel (dB) scale is performed:
 
 \[
 \text{mel\_spec\_db} = 10 \cdot \log_{10}(\text{mel\_spec} + \epsilon)
 \]
 
 where \(\epsilon = 1e-6\) prevents \(\log(0)\). Finally, the spectrogram is normalized to the \([0, 1]\) range using:
 
 \[
 \text{normalized} = \frac{\text{mel\_spec\_db} - \min(\text{mel\_spec\_db})}{\max(\text{mel\_spec\_db}) - \min(\text{mel\_spec\_db}) + \epsilon}
 \]

3. **Artificial Degradations:**  
 The demastering process applies several effects in sequence:
 - **EQ:** Modeled as a Gaussian filter:
   
   \[
   R(f, t) = 1 + G(t) \cdot e^{-\frac{(f - f_c(t))^2}{2\left(\frac{f_c(t)}{Q(t)}\right)^2}}
   \]
   
   Here, \(f_c(t)\), \(Q(t)\), and \(G(t)\) (center, quality factor, and gain) are randomly sampled for each time frame.
   
 - **Dynamic Range Compression:**  
   A differentiable compressor simulates soft-knee behavior:
   
   \[
   \text{compressed\_level} = \text{threshold} + \frac{L - \text{threshold}}{\text{ratio}}
   \]
   
   with additional makeup gain applied.
   
 - **Reverb and Echo:**  
   Convolution with an exponentially decaying impulse response simulates reverb; echo is produced by delaying and attenuating a copy of the signal.

4. **Output Files:**  
 For each track, the pipeline saves:
 - The **original (mastered)** and **degraded (demastered)** audio files (WAV).
 - **Spectrogram images** showing both versions.
 - **Parameter files** detailing the exact processing parameters applied.

---

### 2. Neural Network Training Pipeline

#### Input/Output Representations

- **Input:**  
The model works on mel spectrograms computed as above (128 channels, normalized to \([0, 1]\)). These spectrograms capture both frequency and temporal patterns using the Short-Time Fourier Transform (STFT) followed by a mel filter bank conversion.

- **Output:**  
There are two modes:
1. **Direct Restoration:** The network (a U-Net architecture) maps the degraded spectrogram directly to a restored version.
2. **Parameter Prediction:** A forecasting module (LSTM) predicts a set of effect parameters for each time frame, which are then used in a differentiable audio processing chain to reconstruct the mastered sound.

#### Network Architecture

- **U-Net Module:**  
The U-Net is an encoder–decoder network with skip connections. Its encoder uses convolutional layers with batch normalization and ReLU activations to extract hierarchical features, while the decoder upsamples and fuses these features to reconstruct the spectrogram.

- **LSTM Forecasting:**  
A single-layer LSTM predicts per-timeframe effect parameters (such as gain, EQ parameters, compressor settings, reverb decay, echo delay, and attenuation). The forecasted parameters are then clamped to plausible ranges.

- **Loss Functions:**  
The loss is a weighted combination of:
- **Spectrogram Loss:** \(L_{spectro} = \|S_{\text{pred}} - S_{\text{target}}\|_1\)
- **Parameter Loss:** \(L_{param} = \|P_{\text{pred}} - P_{\text{target}}\|_2^2\) (Note: In practice, the predicted parameters are used as a pseudo-ground-truth since we lack exact targets.)
- **Perceptual Loss:** Additional L1 loss computed on perceptually scaled spectrograms.

The total loss is:

\[
L_{\text{total}} = \lambda_{1} L_{\text{spectro}} + \lambda_{2} L_{\text{param}} + \lambda_{3} L_{\text{perceptual}}
\]

where \(\lambda_i\) are empirically tuned weights.

#### Training Process

- **Dataset:**  
A custom PyTorch Dataset (in `src/dataset.py`) loads paired spectrograms from the demastering pipeline.

- **Training Script:**  
The training loop in `src/train.py` handles batching (with appropriate padding for variable-length time dimensions), loss computation, and learning rate scheduling (with ReduceLROnPlateau to monitor validation loss). Checkpoints and loss plots are saved for analysis.

---

### 3. Evaluation Pipeline

The evaluation script (`src/evaluate.py`) performs the following steps:

1. **Data Preparation:**  
 The input audio file is processed to produce a normalized 128-channel mel spectrogram exactly as in training.

2. **Model Inference:**  
 The cascaded model (consisting of the U-Net for restoration and the LSTM for parameter forecasting) produces a predicted spectrogram. During training, we observed certain channel and time-dimension adjustments (handled internally by the model).

3. **Spectrogram Inversion:**  
 Currently, we use **Griffin–Lim** to invert the denormalized spectrogram (converted back from the [0, 1] normalization to its original dB scale) into audio. Griffin–Lim reconstructs phase via an iterative algorithm, but it may introduce artifacts that sound “static,” robotic, or lack clarity. Future work is planned to integrate a neural vocoder for higher-fidelity reconstruction.

4. **Debugging Outputs:**  
 The script saves:
 - Input and output spectrogram images.
 - Side-by-side comparison images.
 - Predicted effect parameter statistics.
 - Input and reconstructed audio files.
 
 The directory structure for these outputs is maintained as follows:
 
runs/ audio/ # Contains output WAV files (per checkpoint epoch) spectrograms/ # Contains comparison images 
parameters/ # Contains text files with parameter statistics 
debug/ # Contains additional intermediate debugging images


*Note:* Although we attempted to integrate a MelGAN vocoder at one point, due to mismatches in channel dimensions the system currently defaults to Griffin–Lim for inversion.

---

## Major Challenges and Issues Encountered

1. **Time and Channel Dimension Mismatches:**  
- Our dataset and model pipelines originally worked with 128-channel mel spectrograms. However, many pretrained vocoders (e.g., MelGAN) expect 80 channels.  
- We had to introduce downsampling (via bilinear interpolation) for the vocoder branch, which might cause some loss of frequency resolution.
- The model’s architecture (especially the U-Net) sometimes caused slight reductions in the time dimension. We addressed this by padding inputs so that time dimensions are divisible by 16 and later interpolating back to match the original length.

2. **Phase Reconstruction Artifacts:**  
- Griffin–Lim is a classic algorithm for phase reconstruction, but it is inherently iterative and can produce artifacts such as a “static” or “robotic” sound.  
- Although the spectral content (as seen in the debug images) appears similar, the perceptual quality of the audio remains suboptimal.

3. **Loss Function and Parameter Prediction:**  
- The network’s predicted parameters (for EQ, compression, reverb, and echo) tend to remain very close to zero, suggesting that the training loss might not be sufficiently forcing robust transformations.  
- This could indicate that the loss function or training data may require additional refinement or that a perceptual loss might better drive the network toward musically meaningful modifications.

4. **Vocoder Integration:**  
- Efforts to integrate a MelGAN vocoder ran into issues not only with channel mismatch but also with method calls (e.g., using `.infer()`).  
- As a compromise, our current evaluate.py uses Griffin–Lim, although future improvements will aim at integrating a neural vocoder (or even training our own) that can handle 128-channel mel spectrograms directly.

---

## Future Steps

1. **Enhance Inversion Quality:**  
- Investigate alternative phase reconstruction techniques or neural vocoders (such as WaveGlow, MelGAN, or even a custom-trained vocoder on 128-channel mel spectrograms) to overcome the artifacts introduced by Griffin–Lim.

2. **Refine Loss Functions:**  
- Experiment with more perceptually motivated loss functions and adversarial losses to drive the network to produce outputs that are not only spectrally accurate but also sonically pleasing.
- Consider a multi-scale loss formulation that compares spectrograms at different resolutions.

3. **Model Architecture Improvements:**  
- Explore deeper or more advanced U-Net variants and incorporate attention mechanisms to better capture global context while preserving local details.
- Experiment with separate or multi-task losses for the parameter prediction branch to explicitly force non-trivial modifications.

4. **Dataset Expansion and Augmentation:**  
- Expand the dataset beyond FMA Small to include more diverse genres and higher-quality masters.
- Introduce additional augmentation strategies (e.g., random reverberation, noise injection) to simulate real-world mastering challenges.

5. **End-to-End Pipeline Optimization:**  
- Integrate the entire pipeline from raw audio to final audio output with a fully differentiable inversion stage, which would allow the model to learn the inversion process jointly with spectral restoration.

6. **Subjective Evaluation:**  
- Conduct listening tests and gather feedback from audio engineers to refine the model’s performance and validate the perceptual quality improvements.

---

## Mathematical Details

### Mel Spectrogram Conversion

Given an audio waveform \( y(t) \), the Short-Time Fourier Transform (STFT) is computed as:

\[
X(\omega, \tau) = \sum_{t} y(t) \cdot w(t - \tau) \cdot e^{-j\omega t}
\]

The mel spectrogram is then computed by applying a bank of mel filters \( M(f) \):

\[
S_{\text{mel}}(\tau, m) = \sum_{f} M(m, f) \cdot |X(f, \tau)|^2
\]

This spectrogram is converted to decibels via:

\[
S_{\text{dB}}(\tau, m) = 10 \log_{10}\left(S_{\text{mel}}(\tau, m) + \epsilon\right)
\]

where \(\epsilon\) is a small constant (e.g., \(1e{-}6\)).

### Differentiable Audio Effects

- **EQ Response:**  
A Gaussian-like filter for a parametric EQ is modeled as:

\[
R(f, t) = 1 + G(t) \cdot e^{-\frac{(f - f_c(t))^2}{2\left(\frac{f_c(t)}{Q(t)}\right)^2}}
\]

- **Dynamic Compression:**  
A soft knee compressor is modeled via:

\[
L_{\text{comp}}(t) = L(t) - \sigma\left(L(t) - \text{threshold}(t)\right) \cdot \left(1 - \frac{1}{\text{ratio}(t)}\right)
\]

where \(\sigma(\cdot)\) is a sigmoid function to enforce smooth transitions.

### Loss Formulation

The overall loss function is a weighted sum:

\[
L_{\text{total}} = \lambda_{\text{spec}} \|S_{\text{pred}} - S_{\text{target}}\|_1 + \lambda_{\text{param}} \|P_{\text{pred}} - P_{\text{target}}\|_2^2 + \lambda_{\text{percep}} \|S_{\text{pred}} - S_{\text{target}}\|_{1, \text{percep}}
\]

where the perceptual loss term \(\| \cdot \|_{1, \text{percep}}\) might be computed on multi-scale or compressed representations of the spectrogram.

---

## Conclusion

**Current Status:**  
- We have implemented an end-to-end pipeline that converts raw audio into mel spectrograms, applies artificial degradations, and trains a cascaded neural network (U-Net and LSTM) to “remaster” the degraded audio.  
- The system uses Griffin–Lim for spectrogram inversion, producing results that preserve much of the content but suffer from phase reconstruction artifacts.  
- Challenges include channel and time-dimension mismatches, limited dynamic range in predicted parameters, and inherent limitations of Griffin–Lim.

**Future Vision:**  
- Further work will target integrating neural vocoders for improved inversion, refining loss formulations, and expanding the dataset to drive perceptually significant changes.
- This roadmap will help guide discussions with advisors and peers, providing a clear framework for future research and collaborative problem-solving.

This project represents significant progress toward automated music mastering and offers a robust platform for further experimentation and improvement.

---
