class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return './ianvs/project/RFNet-master/Data/cityscapes/'      # folder that contains leftImg8bit/
        elif dataset == 'citylostfound':
            return './ianvs/project/RFNet-master/Data/cityscapesandlostandfound/'  # folder that mixes Cityscapes and Lost and Found
        elif dataset == 'cityrand':
            return './ianvs/project/RFNet-master/Data/cityrand/'
        elif dataset == 'target':
            return './ianvs/project/RFNet-master/Data/target/'
        elif dataset == 'xrlab':
            return './ianvs/project/RFNet-master/Data/xrlab/'
        elif dataset == 'e1':
            return './ianvs/project/RFNet-master/Data/e1/'
        elif dataset == 'mapillary':
            return './ianvs/project/RFNet-master/Data/mapillary/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
