# Federated Learning Project

## Project Description

This project aims to implement and analyze a Federated Learning (FL) system for image classification. The objective is to understand how multiple clients can collaboratively train a model without sharing their local data.

The project focuses on both implementation and analysis, including the comparison between centralized and federated learning, as well as the impact of data distribution across clients.

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

Results show that the federated model converges over rounds and reaches performance close to the centralized model, with a small expected gap.

---

## Current Results

* Centralized model accuracy: 97.0%
* Federated Learning (IID) accuracy after 3 rounds: 94.88%

The results are consistent with the expected behavior of Federated Learning.

---

## Future Work

The next steps of the project aim to extend the current implementation and explore more realistic and advanced scenarios.

### Analysis

* Implement a non-IID data distribution across clients
* Compare IID and non-IID performance
* Study the impact of the number of clients on convergence
* Add visualizations (accuracy vs rounds)

### Model Improvements

* Introduce a Convolutional Neural Network (CNN) for better performance on image data
* Explore autoencoders to learn latent representations and improve feature extraction

### Extensions

* Implement a simplified Federated Learning pipeline using Substra
* Apply the approach to a subset of the MIMIC-IV dataset for a real-world healthcare use case
* Develop a Streamlit interface to visualize training dynamics and experiment with parameters

---

## Technologies

* Python
* PyTorch
* Substra (planned)
* Streamlit (planned)

---

## Conclusion

This project provides a complete implementation of Federated Learning and sets the foundation for further experimentation and real-world applications. It highlights the trade-off between performance and data privacy in distributed machine learning systems.
