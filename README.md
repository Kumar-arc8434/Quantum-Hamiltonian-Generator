Quantum Hamiltonian Generator

A Python tool for generating large datasets of random quantum Hamiltonians across a variety of physical models.
Designed for use in quantum physics research, quantum machine learning, and algorithm benchmarking.

✨ Features

Multiple Hamiltonian families:

Quantum Error Correcting Codes (QECC)

Kitaev Chain

XXZ Model

Heisenberg Chain

Spin Glass

Real XYZ Model

Random Local Hamiltonians

Transverse-Field Ising Model (TFIM)

Long-Range ZZ Couplings

Hybrid Models (mixtures of families)

Efficient dataset handling:

Sparse representation: stores Pauli strings + coefficients (reduces memory footprint).

Batch saving: split datasets into chunks for scalability.

Parallel generation: use multiple CPU cores with joblib.

Fully configurable:

Number of qubits

Dataset size

Batch size

Random seed (reproducibility)

Output directory

🛠️ Requirements

Python ≥ 3.8

Qiskit Terra

NumPy

tqdm

joblib

Install all dependencies with:

pip install -r requirements.txt

🚀 Usage

Run from the command line:

python HamGen_V10_refined.py --n_qubits 5 --num 1000 --batch 100 --jobs 4 --out dataset_folder


Arguments:

--n_qubits : Number of qubits (default: 5)

--num : Total number of Hamiltonians (default: 50,000)

--batch : Batch size for saving (default: 100)

--seed : Random seed (default: 11235813)

--jobs : Number of parallel jobs (default: 1)

--out : Output directory (default: ham_dataset_v10)

📊 Example
python HamGen_V10_refined.py --n_qubits 6 --num 500 --batch 50 --jobs 2


Generates 500 Hamiltonians on 6 qubits, saved in batches of 50.

📂 Output

The generated datasets are saved in compressed .npz format.
Each file contains:

pauli_strings : Encoded operators

coefficients : Corresponding weights

This structure makes it lightweight and easy to load for ML pipelines or further analysis.
