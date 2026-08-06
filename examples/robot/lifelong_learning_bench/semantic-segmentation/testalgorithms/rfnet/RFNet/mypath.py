class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return './dataset/cityscapes/'      # folder that contains leftImg8bit/
        elif dataset == 'citylostfound':
            return './dataset/cityscapesandlostandfound/'  # folder that mixes Cityscapes and Lost and Found
        elif dataset == 'cityrand':
            return './dataset/cityrand/'
        elif dataset == 'target':
            return './dataset/target/'
        elif dataset == 'xrlab':
            return './dataset/xrlab/'
        elif dataset == 'e1':
            return './dataset/e1/'
        elif dataset == 'mapillary':
            return './dataset/mapillary/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
