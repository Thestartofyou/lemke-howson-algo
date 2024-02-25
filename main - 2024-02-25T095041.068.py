import numpy as np

def lemke_howson(A, B):
    """
    Lemke-Howson algorithm for finding Nash equilibria in two-player normal-form games.

    Parameters:
    A (numpy.ndarray): Payoff matrix for Player 1.
    B (numpy.ndarray): Payoff matrix for Player 2.

    Returns:
    numpy.ndarray or None: Nash equilibrium mixed strategies for both players,
                           or None if no equilibrium is found.
    """
    n = A.shape[0]
    m = B.shape[1]
    A_hat = np.hstack((np.eye(n), -B.T))
    B_hat = np.hstack((-A, np.eye(m)))
    tableaux = np.vstack((A_hat, B_hat))
    
    basis = [-1] * (n + m)  # Initialize basis with -1
    for j in range(n):
        entering = j
        leaving = n + j
        basis[leaving] = entering
        pivot_row = tableaux[:, entering]
        pivot_col = tableaux[leaving, :]
        pivot_value = tableaux[leaving, entering]
        tableaux = tableaux - np.outer(pivot_row, pivot_col) / pivot_value
        tableaux[leaving, entering] = 1 / pivot_value

        while True:
            entering_candidates = np.where(tableaux[-1, :-1] < 0)[0]
            if len(entering_candidates) == 0:
                break
            entering = entering_candidates[0]
            leaving = np.argmin(tableaux[:-1, entering] / tableaux[:-1, -1])
            basis[leaving] = entering
            pivot_row = tableaux[:, entering]
            pivot_col = tableaux[leaving, :]
            pivot_value = tableaux[leaving, entering]
            tableaux = tableaux - np.outer(pivot_row, pivot_col) / pivot_value
            tableaux[leaving, entering] = 1 / pivot_value

    for j in range(n):
        if basis[j] >= n:
            continue
        strategy = tableaux[j, -1]
        if strategy != 0:
            return tableaux[j, n:] / strategy, tableaux[-1, :n] / strategy
    return None

# Example usage
A = np.array([[3, 0], [5, 1]])  # Payoff matrix for Player 1
B = np.array([[2, 1], [4, 3]])  # Payoff matrix for Player 2

equilibrium = lemke_howson(A, B)
if equilibrium is not None:
    player1_strategy, player2_strategy = equilibrium
    print("Nash Equilibrium Found:")
    print("Player 1 Strategy:", player1_strategy)
    print("Player 2 Strategy:", player2_strategy)
else:
    print("No Nash Equilibrium Found.")
