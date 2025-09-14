# ⚛️ Quantum Hamiltonian Generator

A Python tool for generating large datasets of random quantum Hamiltonians across a variety of physical models.  
Designed for use in **quantum physics research**, **quantum machine learning**, and **algorithm benchmarking**.

---

## ✨ Features

### Multiple Hamiltonian families:
- Quantum Error Correcting Codes (**QECC**)
- Kitaev Chain
- XXZ Model
- Heisenberg Chain
- Spin Glass
- Real XYZ Model
- Random Local Hamiltonians
- Transverse-Field Ising Model (**TFIM**)
- Long-Range ZZ Couplings
- Hybrid Models (mixtures of families)

### Efficient dataset handling:
- **Sparse representation**: stores Pauli strings + coefficients (reduces memory footprint).
- **Batch saving**: split datasets into chunks for scalability.
- **Parallel generation**: use multiple CPU cores with `joblib`.

### Fully configurable:
- Number of qubits  
- Dataset size  
