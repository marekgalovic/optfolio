import time

import numpy as np

from optfolio.objectives import annualized_return, annualized_volatility, unit_sum_constraint
from optfolio.nsga2 import non_dominated_fronts, tournament_selection, flat_crossover, gaussian_mutation, select_top_individuals


class Optimizer:

    def __init__(self, population_size : int = 5000, max_iter : int = 100, mutation_p : float = 0.3, mutation_p_decay : float = 0.98, mutation_sigma : float = 0.1, verbose : bool = False):
        self._population_size = population_size
        self._max_iter = max_iter
        self._mutation_p = mutation_p
        self._mutation_p_decay = mutation_p_decay
        self._mutation_sigma = mutation_sigma
        self._verbose = verbose

    def run(self, returns : np.ndarray) -> (np.ndarray, dict):
        stats = {
            'return': {'min': [], 'max': [], 'avg': []},
            'volatility': {'min': [], 'max': [], 'avg': []},
            'constraints_violation': {'min': [], 'max': [], 'avg': []},
            'time_per_generation': []
        }
        returns_mean = np.mean(returns, 0)
        returns_cov = np.cov(returns.T)

        population = self._init_population(len(returns_mean))
        return_obj = annualized_return(population, returns_mean)
        volatility_obj = annualized_volatility(population, returns_cov)
        constraints_val = unit_sum_constraint(population)
        fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)

        for gen_idx in range(self._max_iter):
            gen_start_time = time.time()
            mutation_p = self._mutation_p * (self._mutation_p_decay ** gen_idx)

            # Generate offspring
            offspring = np.empty_like(population)
            for i in range(self._population_size):
                (p1_idx, p2_idx) = tournament_selection(fronts, crowding_distances)
                offspring[i,:] = flat_crossover(population[p1_idx], population[p2_idx])
                if np.random.uniform() < mutation_p:
                    offspring[i,:] = gaussian_mutation(offspring[i,:], sigma=self._mutation_sigma)

            offspring = np.clip(offspring, 0, 1)

            # t+1 population
            population = np.concatenate((population, offspring), axis=0)
            return_obj = annualized_return(population, returns_mean)
            volatility_obj = annualized_volatility(population, returns_cov)
            constraints_val = unit_sum_constraint(population)
            fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)
            population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val = select_top_individuals(population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val)

            self._append_stats(stats, return_obj[fronts == 0], volatility_obj[fronts == 0], constraints_val[fronts == 0], float(time.time() - gen_start_time))
            if self._verbose:
                print('Generation: %d, Time: %.2f, Return: %.2f, Volatility: %.2f, Constraints violation: %.4f, N pareto solutions: %d' % (
                    gen_idx, stats['time_per_generation'][-1], stats['return']['avg'][-1] * 100, stats['volatility']['avg'][-1] * 100, stats['constraints_violation']['max'][-1], np.sum(fronts == 0)
                ))

        pareto_front_ids = np.argwhere(fronts == 0).reshape((-1,))
        return population[pareto_front_ids], stats

    def _init_population(self, n_assets : int) -> np.ndarray:
        population = np.random.uniform(0, 1, size=(self._population_size, n_assets))
        
        return population / np.sum(population, 1).reshape((-1, 1))

    def _append_stats(self, stats : dict, return_obj : np.ndarray, volatility_obj : np.ndarray, constraints_val : np.ndarray, tpg : float):
        stats['time_per_generation'].append(tpg)
        for (k, v) in [('return', return_obj), ('volatility', volatility_obj), ('constraints_violation', constraints_val)]:
            stats[k]['min'].append(np.min(v))
            stats[k]['max'].append(np.max(v))
            stats[k]['avg'].append(np.mean(v))
