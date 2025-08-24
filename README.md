# Quantum-Hamiltonian-Generator

A Python tool for generating large datasets of **random quantum Hamiltonians** with various physical models, suitable for **quantum physics research** and **machine learning applications**.

## Features

- Generate Hamiltonians from multiple families:
  - QECC (Quantum Error Correcting Codes)  
  - Kitaev Chain  
  - XXZ Model  
  - Heisenberg Chain  
  - Spin Glass  
  - Real XYZ Model  
  - Random Local Hamiltonians  
  - Transverse-Field Ising Model (TFIM)  
  - Long-Range ZZ Couplings  
  - Hybrid Models (mix of families)
- **Sparse representation**: stores Pauli strings and coefficients to reduce memory usage  
- **Batch saving**: generate datasets in chunks  
- **Parallel generation**: utilize multiple CPU cores  
- **Configurable**: number of qubits, total Hamiltonians, batch size, random seed, output folder  

## Requirements

- Python ≥ 3.8  
- Qiskit (`qiskit-terra`)  
- Numpy  
- tqdm  
- joblib  

Install dependencies via:

```bash
pip install qiskit numpy tqdm joblib
