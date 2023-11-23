from models import Network

def sampling_solution(problem):
    while True:
        network = Network()
        network.genotype = problem.search_space.sample(genotype=True)
        if problem.search_space.is_valid(network.genotype):
            return network

def update_log(best_network, cur_network, **kwargs):
    algorithm = kwargs['algorithm']
    algorithm.trend_best_network.append(best_network)
    algorithm.trend_time.append(algorithm.total_time)

    algorithm.network_history.append(cur_network)
    algorithm.score_history.append(cur_network.score)