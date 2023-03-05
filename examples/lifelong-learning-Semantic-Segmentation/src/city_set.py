class CitySet:
    """
    This class provides the city folders for various train_sets
    """
    t_set1 = ['hanover', 'aachen', 'strasbourg', 'dusseldorf']
    t_set2 = ['jena', 'monchengladbach', 'ulm', 'tubingen', 'bochum', 'hamburg', 'weimar']
    t_set3 = ['zurich', 'darmstadt', 'bremen', 'cologne', 'stuttgart', 'krefeld']
    t_set0 = t_set1 + t_set2 + t_set3
    bench_val = ['erfurt']

    @classmethod
    def get_city_set(cls, train_set):
        if train_set == 1:
            return cls.t_set1
        elif train_set == 2:
            return cls.t_set2
        elif train_set == 3:
            return cls.t_set3
        elif train_set == 0:
            return cls.t_set0
        elif train_set == -1:
            return cls.bench_val
        else:
            return None
