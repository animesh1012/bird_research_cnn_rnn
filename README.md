# Bioacoustics Bird Species Recognition in Realistic Environment Using CNN–RNN Classification Method

This repository is dedicated to the work done using a CNN-RNN hybrid model in bird species identification. :contentReference[oaicite:0]{index=0}

## 📌 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
  - [Conda Environment Setup](#conda-environment-setup)
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Author](#author)

## 🧠 Overview

A hybrid deep learning approach combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for identifying bird species from data.

## 🚀 Features

- Hybrid CNN + RNN architecture for bird species classification  
- Jupyter notebooks demonstrating model training and evaluation  
- Environment YAML for Conda setup  
- Python scripts for implementation and utility functions  
- Visualizations and evaluation results  

## 🛠️ Installation

### 🐍 Conda Environment Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/animesh1012/bird_research_cnn_rnn.git
    cd bird_research_cnn_rnn
    ```

2. **Set up Conda Environment**
Make sure you have Conda installed. Then create the environment from the provided YAML file:
    ```bash
    conda env create -f cnn_rnn_env.yml
    ```

3. **Activate the environment**
      ```bash
      conda activate bird_research_cnn_rnn
      ```

## ▶️ Usage

1. **Data Processing**
Run the `process_data.ipynb` notebook to preprocess the dataset and prepare it for training.

2. **Training & Evaluation**
- Use `CNN_RNN.ipynb` for training the hybrid model.
- Evaluate each model fold performance using the notebooks like `evaluate_ms_model_fold_1.ipynb`.
- Visualize embeddings using `ploting_tsne_best_model.ipynb`.

3. **Python Scripts**
- `Final_my_implementation.py`: Main script for running the model end-to-end.
- `models.py` / `models_original.py`: Contains model definitions.
- `utils.py`: Utility functions.

4. **📊 Results**
- Visualizations and accuracy plots are available, like `per_class_accuracy_bar.png`  and TSNE visualizations in the `tsne_visualization`/ folder.

5. **📘 Notes**
- Designed for Python 3.6 (using Conda environment)
- Ensure you have all necessary libraries installed per `cnn_rnn_env.yml`


## 📁 Project Structure
```bash
bird_research_cnn_rnn/
├── CNN_RNN.ipynb
├── Final_my_implementation.py
├── MS_CNN_RNN_Paper.ipynb
├── MS_CNN_RNN_Paper_256_hidden_size.ipynb
├── MS_CNN_RNN_original_paper_implementation.ipynb
├── evaluate_ms_model_fold.ipynb
├── models.py
├── models_original.py
├── process_data.ipynb
├── utils.py
├── cnn_rnn_env.yml
├── requirements.txt
├── ploting_tsne_best_model.ipynb
├── per_class_accuracy_bar.png
└── tsne_visualization/
```


## 👤 Author

animesh1012

