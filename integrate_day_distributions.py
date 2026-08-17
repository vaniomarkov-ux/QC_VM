# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:29:18 2026

@author: vanio
"""

import numpy as np
from collections import defaultdict
from itertools import product


def integrate_distributions(L, max_len, alphabet):
    """
    Integrate daily sequence distributions into overall distributions.
    
    Parameters:
    -----------
    L : list
        List of daily observations, where each element contains max_len sublists
        (one for each sequence length from 1 to max_len)
    max_len : int
        Maximum sequence length
    alphabet : list
        List of possible values in sequences (e.g., [0,1,...,15])
    
    Returns:
    --------
    integrated_distributions : list
        List of max_len integrated distributions, one for each sequence length
    """
    
    # Initialize storage for counts across all days
    # integrated_counts[length][sequence] = total_count
    integrated_counts = [defaultdict(int) for _ in range(max_len + 1)]
    
    # Iterate through each day's observations
    for day_data in L:
        # Iterate through each sequence length (1 to max_len)
        for length_idx, length_distribution in enumerate(day_data):
            seq_length = length_idx + 1  # Length 1 is at index 0
            
            # Iterate through each sequence observation in this length
            for entry in length_distribution:
                sequence = entry[0]  # The sequence tuple
                count = entry[1]     # Number of occurrences
                
                # Accumulate counts
                integrated_counts[seq_length][sequence] += count
    
    # Convert to the same format as input: [sequence, count, total, probability]
    integrated_distributions = []
    
    for seq_length in range(1, max_len + 1):
        length_dist = []
        
        # Calculate total count for this length
        total_count = sum(integrated_counts[seq_length].values())
        
        # Create entries for each sequence
        for sequence, count in sorted(integrated_counts[seq_length].items()):
            probability = count / total_count if total_count > 0 else 0
            length_dist.append([sequence, count, total_count, probability])
        
        integrated_distributions.append(length_dist)
    
    return integrated_distributions

'''
def print_distribution_summary(distributions, max_display=10):
    """
    Print a summary of the integrated distributions.
    """
    for length_idx, dist in enumerate(distributions):
        seq_length = length_idx + 1
        print(f"\n=== Sequence Length {seq_length} ===")
        print(f"Total unique sequences: {len(dist)}")
        print(f"Total observations: {dist[0][2] if dist else 0}")
        print(f"\nTop {max_display} sequences by probability:")
        
        # Sort by probability (descending)
        sorted_dist = sorted(dist, key=lambda x: x[3], reverse=True)
        
        for i, entry in enumerate(sorted_dist[:max_display]):
            sequence, count, total, prob = entry
            print(f"  {i+1}. {sequence}: count={count}, prob={prob:.6f}")


# Example usage:
if __name__ == "__main__":
    # Example with your data structure
    # L would be your list of daily observations
    # Each element of L contains max_len sublists
    
    max_len = 7
    alphabet = list(range(16))  # [0, 1, ..., 15]
    
    # Example: L = [day1_data, day2_data, ...]
    # where each day_data = [length1_dist, length2_dist, ..., length7_dist]
    
    # Assuming you have L populated with your data:
    # integrated = integrate_distributions(L, max_len, alphabet)
    # print_distribution_summary(integrated)
    
    # For demonstration with dummy data:
    # Create a simple example with 2 days
    day1 = [
        # Length 1
        [[(0,), 100, 500, 0.2], [(1,), 150, 500,
'''


def integrate_conditional_class_distributions(
    L,
    max_len: int,
    n_classes: int = 3,
    alphabet=None,
    include_zero_sequences: bool = False,
):
    """
    Integrate daily class distributions conditioned on symbolic sequences.

    Expected input structure:
    -------------------------
    L : list of days

        Each day_data = [
            length_1_distribution,
            length_2_distribution,
            ...
            length_max_len_distribution
        ]

        Each length_distribution contains entries of the form:

            [
                sequence,        # tuple, e.g. (0, 5, 10)
                class_counts,    # list, e.g. [count_class0, count_class1, count_class2]
                class_probs,     # list, daily conditional probabilities
                total_count      # int, sum(class_counts)
            ]

    Returns:
    --------
    integrated_distributions : list

        Same format as input, one list per sequence length:

            [
                sequence,
                aggregated_class_counts,
                aggregated_class_probs,
                aggregated_total_count
            ]
    """

    # integrated_counts[length][sequence] = vector of class counts
    integrated_counts = [
        defaultdict(lambda: np.zeros(n_classes, dtype=np.int64))
        for _ in range(max_len + 1)
    ]

    # ------------------------------------------------------------
    # accumulate class counts across days
    # ------------------------------------------------------------
    for day_data in L:
        for length_idx, length_distribution in enumerate(day_data):
            seq_length = length_idx + 1

            if seq_length > max_len:
                break

            for entry in length_distribution:
                sequence = tuple(entry[0])
                class_counts = np.asarray(entry[1], dtype=np.int64)

                if len(class_counts) != n_classes:
                    raise ValueError(
                        f"Expected {n_classes} class counts, got {len(class_counts)} "
                        f"for sequence {sequence}."
                    )

                integrated_counts[seq_length][sequence] += class_counts

    # ------------------------------------------------------------
    # optionally include zero-count sequences from the full alphabet
    # ------------------------------------------------------------
    if include_zero_sequences:
        if alphabet is None:
            raise ValueError("alphabet must be provided if include_zero_sequences=True")

        for seq_length in range(1, max_len + 1):
            for sequence in product(alphabet, repeat=seq_length):
                _ = integrated_counts[seq_length][sequence]

    # ------------------------------------------------------------
    # convert back to list format
    # ------------------------------------------------------------
    integrated_distributions = []

    for seq_length in range(1, max_len + 1):
        length_dist = []

        for sequence, counts in sorted(integrated_counts[seq_length].items()):
            total = int(counts.sum())

            if total > 0:
                probs = (counts / total).astype(float).tolist()
            else:
                probs = [0.0] * n_classes

            length_dist.append([
                sequence,
                counts.astype(int).tolist(),
                probs,
                total,
            ])

        integrated_distributions.append(length_dist)

    return integrated_distributions