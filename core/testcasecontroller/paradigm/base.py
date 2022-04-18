import os
from core.common import utils
from core.testcasecontroller.metrics import get_metric_func


class ParadigmBase:
    def __init__(self, test_env, algorithm, workspace):
        self.test_env = test_env
        self.dataset = test_env.dataset
        self.algorithm = algorithm
        self.workspace = workspace
        os.environ["LOCAL_TEST"] = "TRUE"

    def eval_overall(self, result):
        """ eval overall results """
        metric_funcs = []
        for metric_dict in self.test_env.metrics:
            metric = get_metric_func(metric_dict=metric_dict)
            if callable(metric):
                metric_funcs.append(metric)

        eval_dataset_file = self.dataset.eval_dataset
        eval_dataset = self.load_data(eval_dataset_file, data_type="eval overall", label=self.dataset.label)
        metric_res = {}
        for metric in metric_funcs:
            metric_res[metric.__name__] = metric(eval_dataset.y, result)
        return metric_res

    def preprocess_dataset(self, splitting_times=1):
        output_dir = os.path.join(self.workspace, "dataset")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset_files = self.dataset.splitting_more_times(self.dataset.train_dataset,
                                                          self.dataset.format,
                                                          self.dataset.train_ratio,
                                                          output_dir,
                                                          times=splitting_times)
        return dataset_files

    def load_data(self, file: str, data_type: str, label=None, use_raw=False, feature_process=None):
        from sedna.datasources import CSVDataParse, TxtDataParse
        from core.common.constant import DatasetFormat
        format = utils.get_file_format(file)

        if format == DatasetFormat.CSV.value:
            data = CSVDataParse(data_type=data_type, func=feature_process)
            data.parse(file, label=label)
        elif format == DatasetFormat.TXT.value:
            data = TxtDataParse(data_type=data_type, func=feature_process)
            data.parse(file, use_raw=use_raw)

        return data
