from dataclasses import dataclass

@dataclass
class JointInferenceResult:
    is_hard_example : bool
    result : str
    edge_result: str
    cloud_result: str

    @classmethod
    def from_list(cls, *args):
        return cls(*args)