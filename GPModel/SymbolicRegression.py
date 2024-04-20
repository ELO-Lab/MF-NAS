import numpy as np
import itertools
from time import time
from gplearn.functions import _function_map, _Function, sig1 as sigmoid
from gplearn.fitness import _fitness_map, _Fitness
from gplearn.utils import _partition_estimators
from gplearn.utils import check_random_state
from gplearn.genetic import SymbolicRegressor
from GPModel import CloneProgram

from sklearn.base import RegressorMixin
from joblib import Parallel, delayed

MAX_INT = np.iinfo(np.int32).max


def tournament_selection(pool, output_size, size=2):
    # Tournament Selection
    np.random.shuffle(pool)
    pool_survive = []
    pool = np.array(pool)

    n_repeat = output_size // (len(pool) // size)
    for _ in range(n_repeat):
        I = np.random.permutation(len(pool)).reshape((len(pool) // size, size))
        for i in I:
            list_fitness = [candidate.raw_fitness_ for candidate in pool[i]]
            winner = pool[i][np.argmax(list_fitness)]
            pool_survive.append(winner)
    return pool_survive


def calculate_rank(data):
    data = np.array(data)
    rank = np.array([np.arange(1, len(data) + 1)] * data.shape[1])
    for i in range(data.shape[1]):
        rank[i] = [sorted(list(-data[:, i])).index(x) + 1 for x in -data[:, i]]
    mean_rank = np.mean(rank, axis=0)
    return mean_rank


def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    # tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']

    max_samples = int(max_samples * n_samples)

    max_depth = 6
    # Build programs
    programs = []

    if parents is not None:
        list_winner = tournament_selection(parents, output_size=len(parents) * 2, size=2)

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        while True:
            if parents is None:
                program = None
                genome = None
            else:
                # print(len(list_winner))
                method = random_state.uniform()
                # parent, parent_index = _tournament()
                idx = np.random.choice(len(list_winner))
                parent = list_winner[idx]
                list_winner.remove(list_winner[idx])
                if len(list_winner) == 0:
                    list_winner = tournament_selection(parents, output_size=len(parents) * 2, size=2)

                if method < method_probs[0]:
                    # crossover
                    # donor, donor_index = _tournament()
                    idx_ = np.random.choice(len(list_winner))
                    donor = list_winner[idx_]
                    list_winner.remove(list_winner[idx_])
                    if len(list_winner) == 0:
                        list_winner = tournament_selection(parents, output_size=len(parents) * 2, size=2)

                    program, removed, remains = parent.crossover(donor.program,
                                                                 random_state)
                    genome = {'method': 'Crossover',
                              'parent_idx': idx,
                              'parent_nodes': removed,
                              'donor_idx': idx_,
                              'donor_nodes': remains}
                elif method < method_probs[1]:
                    # subtree_mutation
                    program, removed, _ = parent.subtree_mutation(random_state)
                    genome = {'method': 'Subtree Mutation',
                              'parent_idx': idx,
                              'parent_nodes': removed}
                elif method < method_probs[2]:
                    # hoist_mutation
                    program, removed = parent.hoist_mutation(random_state)
                    genome = {'method': 'Hoist Mutation',
                              'parent_idx': idx,
                              'parent_nodes': removed}
                elif method < method_probs[3]:
                    # point_mutation
                    program, mutated = parent.point_mutation(random_state)
                    genome = {'method': 'Point Mutation',
                              'parent_idx': idx,
                              'parent_nodes': mutated}
                else:
                    # reproduction
                    program = parent.reproduce()
                    genome = {'method': 'Reproduction',
                              'parent_idx': idx,
                              'parent_nodes': []}

            program = CloneProgram(function_set=function_set,
                                   arities=arities,
                                   init_depth=init_depth,
                                   init_method=init_method,
                                   n_features=n_features,
                                   metric=metric,
                                   transformer=transformer,
                                   const_range=const_range,
                                   p_point_replace=p_point_replace,
                                   parsimony_coefficient=parsimony_coefficient,
                                   feature_names=feature_names,
                                   random_state=random_state,
                                   program=program)

            program.parents = genome

            if program.depth_ > max_depth:
                continue

            # Draw samples, using sample weights, and then fit
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()
            oob_sample_weight = curr_sample_weight.copy()

            indices, not_indices = program.get_all_indices(n_samples,
                                                           max_samples,
                                                           random_state)

            curr_sample_weight[not_indices] = 0
            oob_sample_weight[indices] = 0

            program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
            if max_samples < n_samples:
                # Calculate OOB fitness
                program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

            programs.append(program)
            break

    return programs


class CloneSymbolicRegressor(SymbolicRegressor):
    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None, multiple=False):
        super().__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)
        self.our_program = {'program': None, 'score': -np.inf}
        self.archive = {'programs': [], 'scores': []}
        self.multiple = multiple

    def fit(self, X, y, sample_weight=None):
        random_state = check_random_state(self.random_state)

        X, y = self._validate_data(X, y, y_numeric=True)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))
        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not ((isinstance(self.const_range, tuple) and
                 len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            print('Warm-start fitting without increasing n_estimators does not '
                  'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            offsprings = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            offsprings = list(itertools.chain.from_iterable(offsprings))

            if self.multiple:
                for program in offsprings:
                    self.archive['programs'].append(program)
                    self.archive['scores'].append(program.raw_fitness_)
                    program.raw_fitness_ = round(np.mean(program.raw_fitness_), 6)

            if gen == 0:
                population = offsprings
            else:
                cur_population = self._programs[gen - 1]
                pool = cur_population + offsprings
                population = tournament_selection(pool, len(pool) // 2, self.tournament_size)
            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            if self.multiple:
                mean_rank = calculate_rank(self.archive['scores'])
                idx_best = np.argmin(mean_rank)
                # mean_scores = np.array(np.mean(self.archive['scores'], axis=1))
                # idx_best = np.argmax(mean_scores)

                best_program = self.archive['programs'][idx_best]
                # print(self.archive['scores'][idx_best], mean_rank[idx_best])
                self.our_program['program'] = best_program
                self.our_program['score'] = best_program.raw_fitness_

                idx_sort = np.argsort(mean_rank)
                self.archive['programs'] = np.array(self.archive['programs'])[idx_sort].tolist()[:1]
                self.archive['scores'] = np.array(self.archive['scores'])[idx_sort].tolist()[:1]
            else:
                if self._metric.greater_is_better:
                    best_program = population[np.argmax(fitness)]
                else:
                    best_program = population[np.argmin(fitness)]

                if (best_program.raw_fitness_ > self.our_program['score']) or (
                        best_program.raw_fitness_ == self.our_program['score'] and best_program.length_ < self.our_program['program'].length_):
                    self.our_program['program'] = best_program
                    self.our_program['score'] = best_program.raw_fitness_

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(self.our_program['program'].length_)
            self.run_details_['best_fitness'].append(self.our_program['score'])
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

            self._program = self.our_program['program']
            if np.round(np.mean(fitness), 6) == np.round(self.our_program['program'].raw_fitness_, 6):
                break

        print('Best model:', self.our_program['program'])
        print('Depth:', self.our_program['program'].depth_)
        print()
        return self