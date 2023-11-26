class Network:
    def __init__(self):
        self.score = None
        self.phenotype = None
        self.genotype = None
        self.model = None
        self.info = {
            'cur_iepoch': [0],
            'train_time': [0.0],
        }

    def set(self, key, value):
        if isinstance(key, list):
            for i, _key in enumerate(key):
                if _key in self.__dict__:
                    self.__dict__[_key] = value[i]
                else:
                    self.info[_key] = value[i]
        else:
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                self.info[key] = value

    def get(self, key):
        if isinstance(key, list):
            data = {}
            for _key in key:
                if _key in self.__dict__:
                    data[_key] = self.__dict__[_key]
                elif _key in self.info:
                    data[_key] = self.info[_key]
                raise ValueError
            return data
        else:
            if key in self.__dict__:
                return self.__dict__[key]
            elif key in self.info:
                return self.info[key]
            raise ValueError