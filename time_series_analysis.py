"""
Time Series Analysis: Correlation, Mutual Information, and Transfer Entropy
For Symbolic/Categorical Time Series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def calculate_correlation(X, lagmax):
    """
    Calculate autocorrelation at different lags for symbolic data.
    Uses Pearson correlation which can still measure linear patterns
    in the numeric encoding, though interpretation differs from continuous data.
    
    Parameters:
    -----------
    X : pandas Series
        Time series data with symbolic values (integers 0..m-1)
    lagmax : int
        Maximum lag to consider
    
    Returns:
    --------
    dict : Dictionary with lags as keys and correlation coefficients as values
    
    Note:
    -----
    For truly symbolic data, this measures whether the numeric encoding
    has linear patterns. Consider mutual information as the primary measure
    for symbolic dependencies.
    """
    correlations = {}
    n = len(X)
    
    for lag in range(1, lagmax + 1):
        if lag < n:
            # Create lagged series
            X_lagged = X.iloc[:-lag].values
            X_current = X.iloc[lag:].values
            
            # Calculate Pearson correlation
            if len(X_lagged) > 1:
                corr, _ = pearsonr(X_lagged, X_current)
                correlations[lag] = corr
            else:
                correlations[lag] = np.nan
    
    return correlations


def calculate_mutual_information(X, lagmax):
    """
    Calculate mutual information between X(t) and X(t-lag) at different lags.
    Uses discrete probability distributions for symbolic data.
    
    Parameters:
    -----------
    X : pandas Series
        Time series data with symbolic/categorical values (integers 0..m-1)
    lagmax : int
        Maximum lag to consider
    
    Returns:
    --------
    dict : Dictionary with lags as keys and MI values as values
    """
    mi_values = {}
    n = len(X)
    
    for lag in range(1, lagmax + 1):
        if lag < n:
            # Create lagged series
            X_lagged = X.iloc[:-lag].values
            X_current = X.iloc[lag:].values
            
            # Calculate mutual information using discrete probabilities
            mi = _compute_mi_discrete(X_current, X_lagged)
            mi_values[lag] = mi
    
    return mi_values


def _compute_mi_discrete(X, Y):
    """
    Compute mutual information for discrete/symbolic variables.
    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    where H is the Shannon entropy.
    """
    n = len(X)
    
    # Calculate joint and marginal distributions
    joint_counts = Counter(zip(X, Y))
    x_counts = Counter(X)
    y_counts = Counter(Y)
    
    # Calculate entropies
    mi = 0.0
    for (x, y), count_xy in joint_counts.items():
        p_xy = count_xy / n
        p_x = x_counts[x] / n
        p_y = y_counts[y] / n
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    
    return max(0, mi)  # MI should be non-negative


def calculate_transfer_entropy(X, lagmax):
    """
    Calculate transfer entropy from X(t-lag) to X(t) given X(t-1).
    Transfer entropy measures: TE = I(X(t); X(t-lag) | X(t-1))
    
    This quantifies the information that X(t-lag) provides about X(t)
    that is not already contained in X(t-1).
    
    For symbolic data, this directly uses the discrete symbol values.
    
    Parameters:
    -----------
    X : pandas Series
        Time series data with symbolic values (integers 0..m-1)
    lagmax : int
        Maximum lag to consider
    
    Returns:
    --------
    dict : Dictionary with lags as keys and TE values as values
    """
    te_values = {}
    n = len(X)
    
    for lag in range(2, lagmax + 1):  # Start at 2 since we need t-1 and t-lag
        if lag < n - 1:
            # Get the relevant time points
            X_t = X.iloc[lag:].values  # Current
            X_t_minus_1 = X.iloc[lag-1:-1].values  # Previous
            X_t_minus_lag = X.iloc[:-(lag)].values  # Lagged
            
            # Make sure all arrays have the same length
            min_len = min(len(X_t), len(X_t_minus_1), len(X_t_minus_lag))
            X_t = X_t[:min_len]
            X_t_minus_1 = X_t_minus_1[:min_len]
            X_t_minus_lag = X_t_minus_lag[:min_len]
            
            # Calculate transfer entropy using discrete probabilities
            te = _compute_te_discrete(X_t, X_t_minus_1, X_t_minus_lag)
            te_values[lag] = te
    
    return te_values


def _compute_te_discrete(X_t, X_t_minus_1, X_t_minus_lag):
    """
    Compute transfer entropy using discrete probability distributions.
    TE = I(X(t); X(t-lag) | X(t-1))
       = H(X(t) | X(t-1)) - H(X(t) | X(t-1), X(t-lag))
    """
    n = len(X_t)
    
    # Count joint occurrences
    joint_all = Counter(zip(X_t, X_t_minus_1, X_t_minus_lag))
    joint_t_t1 = Counter(zip(X_t, X_t_minus_1))
    joint_t1_tlag = Counter(zip(X_t_minus_1, X_t_minus_lag))
    count_t1 = Counter(X_t_minus_1)
    
    # Calculate transfer entropy
    te = 0.0
    for (xt, xt1, xtlag), count in joint_all.items():
        p_all = count / n
        p_t_t1 = joint_t_t1[(xt, xt1)] / n
        p_t1_tlag = joint_t1_tlag[(xt1, xtlag)] / n
        p_t1 = count_t1[xt1] / n
        
        if p_all > 0 and p_t_t1 > 0 and p_t1_tlag > 0 and p_t1 > 0:
            te += p_all * np.log2((p_all * p_t1) / (p_t_t1 * p_t1_tlag))
    
    return max(0, te)  # TE should be non-negative


def visualize_results(correlations, mi_values, te_values, title="Time Series Analysis"):
    """
    Visualize correlation, mutual information, and transfer entropy results.
    
    Parameters:
    -----------
    correlations : dict
        Correlation values at different lags
    mi_values : dict
        Mutual information values at different lags
    te_values : dict
        Transfer entropy values at different lags
    title : str
        Title for the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot correlation
    lags_corr = list(correlations.keys())
    vals_corr = list(correlations.values())
    axes[0].plot(lags_corr, vals_corr, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Lag', fontsize=11)
    axes[0].set_ylabel('Correlation', fontsize=11)
    axes[0].set_title('Autocorrelation', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot mutual information
    lags_mi = list(mi_values.keys())
    vals_mi = list(mi_values.values())
    axes[1].plot(lags_mi, vals_mi, 'o-', linewidth=2, markersize=6, color='#A23B72')
    axes[1].set_xlabel('Lag', fontsize=11)
    axes[1].set_ylabel('Mutual Information (bits)', fontsize=11)
    axes[1].set_title('Mutual Information', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot transfer entropy
    lags_te = list(te_values.keys())
    vals_te = list(te_values.values())
    axes[2].plot(lags_te, vals_te, 'o-', linewidth=2, markersize=6, color='#F18F01')
    axes[2].set_xlabel('Lag', fontsize=11)
    axes[2].set_ylabel('Transfer Entropy (bits)', fontsize=11)
    axes[2].set_title('Transfer Entropy', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def analyze_time_series(X, lagmax, title="Time Series Analysis"):
    """
    Complete analysis: calculate and visualize correlation, MI, and TE.
    
    Parameters:
    -----------
    X : pandas Series
        Time series data with symbolic values (integers 0..m-1)
    lagmax : int
        Maximum lag to consider
    title : str
        Title for the plot
    
    Returns:
    --------
    results : dict
        Dictionary containing all calculated metrics
    fig : matplotlib.figure.Figure
        The generated figure
    """
    print(f"Analyzing symbolic time series of length {len(X)}")
    print(f"Value range: {X.min()} to {X.max()}")
    print(f"Number of symbols (m): {X.nunique()}")
    print(f"Symbol distribution:")
    print(X.value_counts().sort_index())
    print(f"\nMaximum lag: {lagmax}\n")
    
    # Calculate metrics
    print("Calculating autocorrelation...")
    correlations = calculate_correlation(X, lagmax)
    
    print("Calculating mutual information...")
    mi_values = calculate_mutual_information(X, lagmax)
    
    print("Calculating transfer entropy...")
    te_values = calculate_transfer_entropy(X, lagmax)
    
    # Visualize
    print("Generating visualization...")
    fig = visualize_results(correlations, mi_values, te_values, title)
    
    results = {
        'correlation': correlations,
        'mutual_information': mi_values,
        'transfer_entropy': te_values
    }
    
    return results, fig


# Example usage
if __name__ == "__main__":
    # Generate example symbolic time series (Markov chain)
    np.random.seed(42)
    n = 500
    m = 5  # 5 symbols: 0, 1, 2, 3, 4
    
    # Define transition matrix for a Markov chain
    # Higher probability to stay at same state or move to adjacent states
    transition_matrix = np.array([
        [0.4, 0.3, 0.2, 0.1, 0.0],  # from state 0
        [0.2, 0.4, 0.3, 0.1, 0.0],  # from state 1
        [0.1, 0.2, 0.4, 0.2, 0.1],  # from state 2
        [0.0, 0.1, 0.3, 0.4, 0.2],  # from state 3
        [0.0, 0.1, 0.2, 0.3, 0.4],  # from state 4
    ])
    
    # Generate symbolic sequence
    X_data = np.zeros(n, dtype=int)
    X_data[0] = np.random.randint(0, m)
    
    for i in range(1, n):
        current_state = X_data[i-1]
        X_data[i] = np.random.choice(m, p=transition_matrix[current_state])
    
    X = pd.Series(X_data)
    
    # Analyze
    results, fig = analyze_time_series(X, lagmax=20, 
                                       title="Example: Symbolic Markov Chain (5 symbols)")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nCorrelation at lag 1: {results['correlation'][1]:.4f}")
    print(f"Mutual Information at lag 1: {results['mutual_information'][1]:.4f} bits")
    if 2 in results['transfer_entropy']:
        print(f"Transfer Entropy at lag 2: {results['transfer_entropy'][2]:.4f} bits")
    
    # For a Markov chain, TE should be close to 0 at lag > 1
    # because X(t-1) captures all the information needed to predict X(t)
    print("\nMarkov property check (TE should be low at higher lags):")
    for lag in [2, 3, 4, 5]:
        if lag in results['transfer_entropy']:
            print(f"  TE at lag {lag}: {results['transfer_entropy'][lag]:.4f} bits")
    
    plt.savefig('/mnt/user-data/outputs/symbolic_time_series_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: /mnt/user-data/outputs/symbolic_time_series_analysis.png")
    plt.show()


def align_lag(x, y, lag, missing=-1):
    x = np.asarray(x)
    y = np.asarray(y)
    if lag > 0:
        x2, y2 = x[lag:], y[:-lag]
    elif lag < 0:
        lag = -lag
        x2, y2 = x[:-lag], y[lag:]
    else:
        x2, y2 = x, y
    m = (x2 != missing) & (y2 != missing)
    return x2[m], y2[m]

def _row_ids(*cols):
    M = np.column_stack([np.asarray(c, dtype=np.int64) for c in cols])
    M = np.ascontiguousarray(M)
    v = M.view(np.dtype((np.void, M.dtype.itemsize * M.shape[1])))
    _, inv = np.unique(v, return_inverse=True)
    return inv.astype(np.int64)

def entropy_discrete(x):
    x = np.asarray(x, dtype=np.int64).ravel()
    if x.size == 0:
        return np.nan
    _, inv = np.unique(x, return_inverse=True)
    inv = inv.ravel()
    c = np.bincount(inv)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def mi_discrete(x, y):
    hX  = entropy_discrete(x)
    hY  = entropy_discrete(y)
    hXY = entropy_discrete(_row_ids(x, y))
    return hX + hY - hXY

def compute_mi_discrete(X, Y, maxlag, missing=-1, min_n=50):
    lags = np.arange(-maxlag, maxlag + 1)
    mi = np.full_like(lags, np.nan, dtype=float)
    for i, lag in enumerate(lags):
        x, y = align_lag(X, Y, lag, missing=missing)
        if len(x) >= min_n:
            mi[i] = mi_discrete(x, y)
    return lags, mi