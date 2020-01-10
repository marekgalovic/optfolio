import numpy as np

from optfolio import YEAR_BARS


def annualized_return(solutions : np.ndarray, returns_mean : np.ndarray) -> np.ndarray:
    returns = np.matmul(solutions, returns_mean)

    return (returns + 1) ** YEAR_BARS - 1


def annualized_volatility(solutions : np.ndarray, returns_cov : np.ndarray) -> np.ndarray:
    volatilities = np.sum(solutions * np.matmul(solutions, returns_cov), -1)

    return np.sqrt(volatilities * YEAR_BARS)


def unit_sum_constraint(solutions : np.ndarray, eps : float = 1e-4) -> np.ndarray:
    return np.clip(np.abs(np.sum(solutions, -1) - 1) - eps, 0, None)
