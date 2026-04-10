# Federated Learning Project

## Project Description

This project aims to implement and analyze a Federated Learning (FL) system for image classification. The objective is to understand how multiple clients can collaboratively train a model without sharing their local data.

The project focuses on both implementation and experimental analysis, including the comparison between centralized and federated learning, as well as the impact of data distribution across clients.

---

## Sprint 1: Implementation

During the first sprint, a complete Federated Learning pipeline was implemented from scratch using PyTorch.

The following components were developed:

* A baseline model (MLP) for image classification on MNIST
* Centralized training to establish a performance reference
* Simulation of multiple clients with IID data distribution
* Local training on each client
* Implementation of the Federated Averaging algorithm
* A full Federated Learning loop over multiple rounds
* Evaluation and comparison between centralized and federated performance

Results showed that the federated model converges over rounds and reaches performance close to the centralized model, with a small expected gap.

---

## Sprint 2: Experimental Analysis

The second sprint focuses on a deeper analysis of Federated Learning under more realistic conditions.

### Data Distribution Analysis

Several data distribution scenarios were implemented and compared:

* IID distribution (baseline)
* Non-IID distribution by classes (each client has different classes)
* Non-IID imbalanced distribution (unequal data across clients)

This allowed a detailed study of how data heterogeneity affects model performance.

---

### Experimental Studies

Additional experiments were conducted to better understand FL dynamics:

* Impact of the number of clients on convergence
* Impact of the number of communication rounds
* Visualization of accuracy evolution across rounds

---

### Model Evolution

Three different models were implemented and compared:

#### MLP (Sprint 1 baseline)
* Simple architecture
* Good baseline performance
* Limited capacity for image data

#### CNN V1
* Introduced convolutional layers
* Significant improvement over MLP
* Good performance across all scenarios

#### CNN V2 (BatchNorm + Dropout)
* Added Batch Normalization and Dropout
* Improved stability in IID settings
* Failed in non-IID scenarios due to inconsistent BatchNorm statistics across clients

#### CNN V3 (Final Model)
* Removed BatchNorm
* Kept Dropout for regularization
* Restored strong performance in non-IID settings
* Best trade-off between performance and robustness

---

### Key Findings

* Federated Learning performs well under IID conditions
* Non-IID data (especially class-based) significantly degrades performance
* Model architecture alone cannot solve non-IID challenges
* BatchNorm is not well-suited for Federated Learning with heterogeneous data
* A simpler and well-adapted model (CNN V3) provides better robustness

---

## Current Results

* Centralized model accuracy: ~97.0%
* Federated Learning (IID, CNN): ~98.5%
* Federated Learning (Non-IID classes): ~83–85%
* Federated Learning (Non-IID imbalanced): ~98–99%

These results highlight both the effectiveness and limitations of Federated Learning.

---

## Future Work

The next steps aim to extend the project toward more advanced and realistic applications.

### Model Improvements

* Explore autoencoders to learn better latent representations
* Combine feature extraction with Federated Learning

### Federated Learning Extensions

* Integrate Substra to simulate real-world FL pipelines
* Experiment with more advanced aggregation strategies

### Real-World Application

* Apply the pipeline to a subset of the MIMIC-IV dataset
* Explore privacy-preserving learning in healthcare

### Visualization

* Develop a Streamlit interface to:
  * visualize training dynamics
  * compare scenarios interactively
  * experiment with parameters

---

## Technologies

* Python
* PyTorch
* Substra (planned)
* Streamlit (planned)

---

## Conclusion

This project provides a complete implementation and analysis of Federated Learning, from basic concepts to advanced challenges such as non-IID data.

It highlights a key insight:

> Performance in Federated Learning is not only determined by the model, but also by the distribution of data across clients.

The project establishes a strong foundation for further research and real-world applications in distributed and privacy-preserving machine learning.