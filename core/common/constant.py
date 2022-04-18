from enum import Enum


class DatasetFormat(Enum):
    """
    dataset format
    """
    CSV = "csv"
    TXT = "txt"


class ParadigmKind(Enum):
    """
    paradigm kind
    """
    SingleTaskLearning = "singletasklearning"
    IncrementalLearning = "incrementallearning"
