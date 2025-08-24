# HamGen_V9.py
# Improved: More models, sparse storage, parallel generation, flexible config

import os
import numpy as np
import random
import argparse
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------------- Configurable defaults ----------------
DEFAULT_N_QUBITS = 5
DEFAULT_NUM_HAMILTONIANS = 50_000
DEFAULT_BATCH_SIZE = 100
DEFAULT_SEED = 11235813

# ---------------- Weighted selection ----------------
WEIGHTS_DICT = {
    "qecc": 0.08,
    "kitaev_chain": 0.1,
    "xxz": 0.1,
    "heisenberg_chain": 0.1,
    "spin_glass": 0.1,
    "real_xyz": 0.05,
    "random_local": 0.2,
    "tfim": 0.12,
    "long_range": 0.1,
    "hybrid": 0.05
}
assert abs(sum(WEIGHTS_DICT.values()) - 1.0) < 1e-6, "Weights must sum to 1"

def weighted_choice(d: dict):
    return random.choices(list(d.keys()), weights=list(d.values()), k=1)[0]

# ---------------- Hamiltonian generators ----------------

def generate_qecc_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    for i in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'X'
        terms.append((''.join(pauli_str), 1.0))
    return SparsePauliOp.from_list(terms)

def generate_kitaev_chain_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    mu, t, delta = 1.0, 1.0, 1.0
    for i in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'Z'
        terms.append((''.join(pauli_str), -mu / 2))
    for i in range(n_qubits - 1):
        xx, yy, xy, yx = ['I']*n_qubits, ['I']*n_qubits, ['I']*n_qubits, ['I']*n_qubits
        xx[i], xx[i+1] = 'X','X'
        yy[i], yy[i+1] = 'Y','Y'
        xy[i], xy[i+1] = 'X','Y'
        yx[i], yx[i+1] = 'Y','X'
        terms += [
            (''.join(xx), t), (''.join(yy), t),
            (''.join(xy), delta), (''.join(yx), -delta)
        ]
    return SparsePauliOp.from_list(terms)

def generate_xxz_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    for i in range(n_qubits - 1):
        for pauli, coeff in zip(['X','Y','Z'], [1.0, 1.0, random.uniform(0.5, 1.5)]):
            ps = ['I']*n_qubits
            ps[i] = ps[i+1] = pauli
            terms.append((''.join(ps), coeff))
    return SparsePauliOp.from_list(terms)

def generate_heisenberg_chain_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    for i in range(n_qubits - 1):
        ps = ['I']*n_qubits
        ps[i], ps[i+1] = 'X','X'
        terms.append((''.join(ps), 1.0))
    for i in range(n_qubits):
        ps = ['I']*n_qubits
        ps[i] = 'Z'
        terms.append((''.join(ps), 1.0))
    return SparsePauliOp.from_list(terms)

def generate_spin_glass_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    for i, j in combinations(range(n_qubits), 2):
        ps = ['I']*n_qubits
        ps[i] = ps[j] = 'Z'
        terms.append((''.join(ps), random.choice([-1.0, 1.0])))
    return SparsePauliOp.from_list(terms)

def generate_real_xyz_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    for i, j in combinations(range(n_qubits), 2):
        for pauli in ['X','Y','Z']:
            ps = ['I']*n_qubits
            ps[i] = ps[j] = pauli
            terms.append((''.join(ps), random.uniform(0.1, 1.0)))
    return SparsePauliOp.from_list(terms)

def generate_random_local_hamiltonian(n_qubits) -> SparsePauliOp:
    terms = []
    for i in range(n_qubits):
        pauli = random.choice(['X','Y','Z'])
        ps = ['I']*n_qubits
        ps[i] = pauli
        terms.append((''.join(ps), random.uniform(-1,1)))
    for i, j in combinations(range(n_qubits), 2):
        pauli1, pauli2 = random.choice(['X','Y','Z']), random.choice(['X','Y','Z'])
        ps = ['I']*n_qubits
        ps[i], ps[j] = pauli1, pauli2
        terms.append((''.join(ps), random.uniform(-1,1)))
    return SparsePauliOp.from_list(terms)

# NEW: Transverse-Field Ising Model
def generate_tfim_hamiltonian(n_qubits) -> SparsePauliOp:
    h = random.uniform(0.5, 2.0)
    terms = []
    for i in range(n_qubits-1):
        ps = ['I']*n_qubits
        ps[i], ps[i+1] = 'Z','Z'
        terms.append((''.join(ps), -1.0))
    for i in range(n_qubits):
        ps = ['I']*n_qubits
        ps[i] = 'X'
        terms.append((''.join(ps), -h))
    return SparsePauliOp.from_list(terms)

# NEW: Long-range ZZ couplings with power-law decay
def generate_long_range_hamiltonian(n_qubits) -> SparsePauliOp:
    alpha = random.uniform(1.0, 3.0)
    terms = []
    for i, j in combinations(range(n_qubits), 2):
        ps = ['I']*n_qubits
        ps[i], ps[j] = 'Z','Z'
        dist = abs(i-j)
        coeff = 1.0/(dist**alpha)
        terms.append((''.join(ps), coeff))
    return SparsePauliOp.from_list(terms)

# NEW: Hybrid model (mixes random terms from families)
def generate_hybrid_hamiltonian(n_qubits) -> SparsePauliOp:
    all_terms = []
    sub_models = random.sample(list(HAMILTONIAN_MAP.values()), k=3)
    for gen in sub_models:
        ham = gen(n_qubits)
        term_list = ham.to_list()
        sampled = random.sample(term_list, min(3, len(term_list)))
        all_terms.extend(sampled)
    return SparsePauliOp.from_list(all_terms)

# ---------------- Mapping ----------------
HAMILTONIAN_MAP = {
    "qecc": generate_qecc_hamiltonian,
    "kitaev_chain": generate_kitaev_chain_hamiltonian,
    "xxz": generate_xxz_hamiltonian,
    "heisenberg_chain": generate_heisenberg_chain_hamiltonian,
    "spin_glass": generate_spin_glass_hamiltonian,
    "real_xyz": generate_real_xyz_hamiltonian,
    "random_local": generate_random_local_hamiltonian,
    "tfim": generate_tfim_hamiltonian,
    "long_range": generate_long_range_hamiltonian,
    "hybrid": generate_hybrid_hamiltonian
}

# ---------------- Generation ----------------
def generate_one_hamiltonian(n_qubits):
    choice = weighted_choice(WEIGHTS_DICT)
    return HAMILTONIAN_MAP[choice](n_qubits)

def generate_hamiltonians(num_hams, n_qubits, n_jobs=1):
    if n_jobs > 1:
        return Parallel(n_jobs=n_jobs)(delayed(generate_one_hamiltonian)(n_qubits) for _ in tqdm(range(num_hams)))
    else:
        return [generate_one_hamiltonian(n_qubits) for _ in tqdm(range(num_hams))]

# ---------------- Saving ----------------
def save_hamiltonians_sparse(hams, batch_size, output_dir, filename_prefix="hamiltonians"):
    os.makedirs(output_dir, exist_ok=True)
    num_batches = (len(hams)+batch_size-1)//batch_size
    for i in range(num_batches):
        start, end = i*batch_size, min((i+1)*batch_size, len(hams))
        batch = hams[start:end]
        # Save as list of (pauli, coeffs)
        batch_terms = [ham.to_list() for ham in batch]
        filename = os.path.join(output_dir, f"{filename_prefix}_part_{i+1:03d}.npz")
        np.savez_compressed(filename, batch=batch_terms)
        print(f"Saved batch {i+1}/{num_batches} ({len(batch)}) → {filename}")

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--num", type=int, default=DEFAULT_NUM_HAMILTONIANS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--out", type=str, default="ham_dataset_v9")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Generating {args.num} Hamiltonians ({args.n_qubits} qubits) in batches of {args.batch}...")
    hams = generate_hamiltonians(args.num, args.n_qubits, args.jobs)

    print("Saving to disk (sparse representation)...")
    save_hamiltonians_sparse(hams, args.batch, args.out)

    config = {
        "N_QUBITS": args.n_qubits,
        "NUM_HAMILTONIANS": args.num,
        "BATCH_SIZE": args.batch,
        "SEED": args.seed,
        "WEIGHTS": WEIGHTS_DICT
    }
    np.savez(os.path.join(args.out, "config.npz"), config=config, allow_pickle=True)
    print("✅ Dataset generation complete! Config saved.")
