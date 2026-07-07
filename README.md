# Multimodal Foundation Model for Sleep Stage Classification and Sleep Apnea Detection

## Overview

This repository contains research conducted under Prof. Nipun Batra at Sustainability Labs, IIT Gandhinagar. The project explores self-supervised contrastive learning for learning robust representations from physiological signals and developing foundation models for sleep stage classification and sleep apnea detection.

The primary objective is to reduce dependence on costly expert annotations by leveraging large amounts of unlabeled polysomnography (PSG) data and fine-tuning on limited labeled samples.

---

## Motivation

Sleep stage scoring and apnea detection require experts to manually annotate hundreds to thousands of 30-second epochs per patient. This process is expensive, time-consuming, and prone to inter-rater variability.

Self-supervised learning provides a scalable alternative by learning meaningful representations from unlabeled physiological signals before adapting them to downstream clinical tasks.

---

## Contrastive Learning Framework

The core idea of this project is to learn meaningful representations from large amounts of unlabeled data using self-supervised contrastive learning.

For each input sample, two augmented views are generated and passed through a shared encoder network. The encoder learns to bring representations of positive pairs closer in the embedding space while pushing representations of different samples apart using the InfoNCE (NT-Xent) loss. After pretraining, the encoder is fine-tuned using labeled data for downstream classification tasks.

This framework enables efficient utilization of unlabeled data and significantly reduces dependence on expensive expert annotations.

---

## Datasets

| Dataset   | Modality          | Task                       |
| --------- | ----------------- | -------------------------- |
| MNIST     | Grayscale Images  | Digit Classification       |
| CIFAR-10  | RGB Images        | Object Classification      |
| UCI-HAR   | Time-Series Data  | Activity Recognition       |
| Sleep-EDF | EEG & EMG Signals | Sleep Stage Classification |

### Sleep-EDF Signals

<p align="center">
  <img src="images/sleed_edf_signals.png" alt="Sleep EDF Signals" width="900">
</p>

<p align="center">
  <em>Example physiological signals from the Sleep-EDF dataset.</em>
</p>

---

## Results

| Dataset                               | Accuracy |
| ------------------------------------- | -------- |
| MNIST                                 | 97%      |
| CIFAR-10                              | 80–85%   |
| UCI-HAR                               | 95–97%   |
| Sleep-EDF (100% Supervised)           | 91%      |
| Sleep-EDF (Contrastive + Fine-Tuning) | 88%      |
| Sleep-EDF (10% Supervised)            | 86%      |

These results demonstrate that contrastive pretraining achieves performance close to fully supervised learning while requiring substantially fewer labeled samples.

---

## Key Findings

* Contrastive learning generalizes effectively across image and time-series modalities.
* Pretrained encoders significantly improve performance when labeled data is limited.
* Learned embeddings exhibit meaningful clustering and improved separability.
* Self-supervised learning is a promising direction for large-scale medical signal analysis.

---

## Repository Structure

```text
.
## Repository Structure

```text
.
├── Research Papers/
│   ├── SleepFM.pdf
│   ├── Stanford Sleep Bench.pdf
│   └── ...
│
├── Self_supervised_learning/
│   ├── MNIST.ipynb
│   ├── CIFAR.ipynb
│   ├── Contrastive.ipynb
│   └── uci_har.ipynb
│
├── Sleep_edf/
│   ├── sleep_edf.ipynb
│   ├── edf_contrastive.ipynb
│   ├── edf_supervised.ipynb
│   └── edf_supervised_10.ipynb
│
├── Learning-with-FrameProjections/
│   ├── main.py
│   ├── trainer_SSL_LE.py
│   ├── models/
│   └── data_preprocess/
│
├── images/
│   ├── Architecture.png
│   └── sleed_edf_signals.png
│
└── README.md
```



---

## Future Work

* Integrate additional PSG modalities including EOG and respiratory signals.
* Scale the encoder into a multimodal sleep foundation model.
* Pretrain on larger and more diverse sleep datasets.
* Fine-tune on clinical sleep data from AIIMS Delhi.
* Develop automated systems for sleep stage classification and obstructive sleep apnea detection.

---

## Acknowledgements

This work was conducted under the guidance of Prof. Nipun Batra at Sustainability Labs, IIT Gandhinagar.
