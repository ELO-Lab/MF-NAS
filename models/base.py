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
    
    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.info[key] = value

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.info:
            return self.info[key]
        return None

    def __call__(self):
        print('Genotype:', self.genotype, 'Score:', self.score)