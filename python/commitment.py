"""Torch-free host-side commitment + proof building (numpy + tensorbench_runtime).

Mirrors pearl_miner.commitment_hashes / build_proof / verify_proof_local but takes
numpy int8 matrices instead of torch tensors. Validated bit-exact against the GPU
tensor_hash commitment.
"""
from __future__ import annotations

import blake3
import numpy as np

import tensorbench_runtime as pm


def _tree(matrix_i8: np.ndarray):
    """MerkleTree over the chunk-padded matrix bytes (caller binds the key)."""
    return matrix_i8.view(np.uint8).tobytes()


def _merkle_tree(matrix_i8: np.ndarray, key: bytes):
    padded = pm.pad_to_chunk_boundary(np.ascontiguousarray(matrix_i8).view(np.uint8).tobytes())
    return pm.MerkleTree(padded, key)


def commitment_hashes(A: np.ndarray, B: np.ndarray, key: bytes) -> tuple[bytes, bytes]:
    """(noise_seed_A, noise_seed_B). A:[m,k] int8, B:[k,n] int8. Commits A and B^T."""
    root_A = bytes(_merkle_tree(A, key).root)
    root_B = bytes(_merkle_tree(np.ascontiguousarray(B.T), key).root)
    commitment_B = blake3.blake3(key + root_B).digest()
    commitment_A = blake3.blake3(commitment_B + root_A).digest()
    return commitment_A, commitment_B


def build_proof(A: np.ndarray, B: np.ndarray, winning_row: int, winning_col: int,
                key: bytes, noise_rank: int):
    """Consensus-valid PlainProof for a winning 16x16 tile (host numpy version)."""
    m, k = A.shape
    n = B.shape[1]
    TILE = 16
    a_rows = list(range(winning_row, min(winning_row + TILE, m)))
    b_cols = list(range(winning_col, min(winning_col + TILE, n)))
    tree_A = _merkle_tree(A, key)
    tree_B = _merkle_tree(np.ascontiguousarray(B.T), key)
    li_A = pm.MerkleTree.compute_leaf_indices_from_rows(a_rows, (m, k))
    li_B = pm.MerkleTree.compute_leaf_indices_from_rows(b_cols, (n, k))
    mp_A = pm.MatrixMerkleProof(tree_A.get_multileaf_proof(li_A), a_rows)
    mp_B = pm.MatrixMerkleProof(tree_B.get_multileaf_proof(li_B), b_cols)
    return pm.PlainProof(m, n, k, noise_rank, mp_A, mp_B, None)


def build_proof_bt(A: np.ndarray, Bt: np.ndarray, winning_row: int, winning_col: int,
                   key: bytes, noise_rank: int):
    m, k = A.shape
    n = Bt.shape[0]
    TILE = 16
    a_rows = list(range(winning_row, min(winning_row + TILE, m)))
    b_cols = list(range(winning_col, min(winning_col + TILE, n)))
    tree_A = _merkle_tree(A, key)
    tree_B = _merkle_tree(Bt, key)
    li_A = pm.MerkleTree.compute_leaf_indices_from_rows(a_rows, (m, k))
    li_B = pm.MerkleTree.compute_leaf_indices_from_rows(b_cols, (n, k))
    mp_A = pm.MatrixMerkleProof(tree_A.get_multileaf_proof(li_A), a_rows)
    mp_B = pm.MatrixMerkleProof(tree_B.get_multileaf_proof(li_B), b_cols)
    return pm.PlainProof(m, n, k, noise_rank, mp_A, mp_B, None)


def derive_key(header_bytes: bytes, mining_config) -> bytes:
    return blake3.blake3(header_bytes + mining_config.to_bytes()).digest()


def int_to_nbits(target: int) -> int:
    """Bitcoin-style compact target encoding (matches zk-pow's nbits decode)."""
    if target <= 0:
        return 0
    size = (target.bit_length() + 7) // 8
    if size <= 3:
        compact = target << (8 * (3 - size))
    else:
        compact = target >> (8 * (size - 3))
    if compact & 0x800000:
        compact >>= 8
        size += 1
    return (size << 24) | compact


def verify_proof_local(header_bytes: bytes, proof, nbits: int | None = None, cert_version: int = 2):
    hdr = pm.IncompleteBlockHeader.from_bytes(header_bytes)
    # verify_plain_proof_for_cert_version(cert_version, block_header, plain_proof,
    # nbits_override=None) — the only plain-verify surface the runtime exposes.
    return pm.verify_plain_proof_for_cert_version(cert_version, hdr, proof, nbits)
