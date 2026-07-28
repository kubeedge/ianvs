import json

from sedna.common.class_factory import ClassFactory, ClassType

@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeDecodingDatasetProcessor")
class SpeculativeDecodingDatasetProcessor:
    def __init__(self, **kwargs):
        sample_size = kwargs.get("sample_size")
        self.sample_size = int(sample_size) if sample_size is not None else None
        warmup_samples = kwargs.get("warmup_samples")
        self.warmup_samples = max(0, int(warmup_samples)) if warmup_samples is not None else 0

    def __call__(self, dataset):
        dataset_name = getattr(dataset, "dataset_name", "default")
        processed = []
        for index, (x, y) in enumerate(zip(dataset.x, dataset.y)):
            item = self._parse_item(x)
            processed.append(
                {
                    "request_id": item.get("request_id") or f"request-{index:03d}",
                    "query": item.get("query", str(x)),
                    "gold": y,
                    "task_name": item.get("task_name", dataset_name),
                    "conversation_id": item.get("conversation_id"),
                    "turn_index": item.get("turn_index"),
                    "turn_count": item.get("turn_count"),
                    "question_id": item.get("question_id"),
                    "category": item.get("category"),
                    "sample_index": index,
                }
            )

        if self.sample_size is not None and self.sample_size > 0:
            processed = processed[: self.sample_size]
            dataset.y = dataset.y[: self.sample_size]

        effective_warmup = min(self.warmup_samples, len(processed))
        for index, item in enumerate(processed):
            item["sample_index"] = index
            item["warmup_samples"] = effective_warmup
            item["is_warmup"] = index < effective_warmup

        dataset.x = processed
        return dataset

    @staticmethod
    def _parse_item(value):
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return {"query": value}
