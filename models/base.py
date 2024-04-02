class Network:
    def __init__(self):
        self.score = -99999999999999
        self.phenotype = None
        self.genotype = None
        self.model = None
        self.info = {
            'cur_iepoch': [0],
            'train_time': [0.0],
        }
