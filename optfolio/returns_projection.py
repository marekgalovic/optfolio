from multiprocessing import Pool, cpu_count

import numpy as np


def cumulative_n_period_returns(traces, period_bars):
    assert traces.shape[1] % period_bars == 0
    n_periods = int(traces.shape[1] / period_bars)

    result = traces.reshape((-1, n_periods, period_bars))
    result = np.prod(result + 1, axis=-1)
    return np.cumprod(result, axis=-1) - 1


def _weighted_returns_pmf(returns : np.ndarray, bins : int = 1000) -> (np.ndarray, np.ndarray):
    hist, bins = np.histogram(returns, bins=bins, density=True)

    return (
        hist / np.sum(hist),
        bins[:-1] + (bins[1:] - bins[:-1]) / 2.0
    )


def sample_returns(returns : np.ndarray, n_bars : int, n_traces : int = 10000, pdf_bins : int = 1000) -> np.ndarray:
    period_returns_p, period_returns = _weighted_returns_pmf(returns, bins=pdf_bins)

    return np.random.choice(period_returns, size=(n_traces, n_bars), p=period_returns_p, replace=True)


def _mcmc_trace(P : np.ndarray, returns : np.ndarray, ret_min : float, ret_max : float, n_traces : int, n_bars : int) -> np.ndarray:
    ret_range = ret_max - ret_min
    bin_size = ret_range / P.shape[0]
    
    result = np.empty((n_traces, n_bars), dtype=np.float32)
    for i in range(n_traces):
        initial_return = np.random.choice(returns)
        state_idx = int(np.floor((initial_return - ret_min) / bin_size))

        for j in range(n_bars):
            lb = (state_idx / P.shape[0]) * ret_range + ret_min
            ub = lb + bin_size
            result[i,j] = np.random.uniform(lb, ub)

            state_idx = np.random.choice(P.shape[1], p=P[state_idx,:])

    return result


def mcmc_sample_returns(returns : np.ndarray, n_bars : int, n_traces : int = 1000, mc_states : int = 10, n_jobs : int = -1) -> np.ndarray:
    ret_min, ret_max = np.min(returns), np.max(returns) + 1e-4
    ret_range = ret_max - ret_min
    bin_size = ret_range / mc_states

    # Build transition matrix
    P = np.zeros((mc_states, mc_states), dtype=np.float32)
    for i in range(len(returns) - 1):
        curr_state_idx = int(np.floor((returns[i] - ret_min) / bin_size))
        next_state_idx = int(np.floor((returns[i+1] - ret_min) / bin_size))
        P[curr_state_idx, next_state_idx] += 1

    P /= np.sum(P, -1, keepdims=True)
    if np.any(np.isnan(P)):
        raise ValueError('Transition matrix contains NaN. Try lower mc_states value or provide more data.')

    initial_return = np.random.choice(returns)
    state_idx = int(np.floor((initial_return - ret_min) / bin_size))

    n_partitions = n_jobs if n_jobs > 0 else cpu_count()
    partition_size = int(n_traces / n_partitions)
    pool = Pool(n_partitions)
    try:
        result = pool.starmap(_mcmc_trace, ((P, returns, ret_min, ret_max, partition_size, n_bars) for _ in range(n_partitions)))
        pool.close()

        return np.concatenate(result, 0)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()
