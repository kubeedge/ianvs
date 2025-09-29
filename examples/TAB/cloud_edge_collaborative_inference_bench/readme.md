# Cloud-Edge Collaborative Inference Bench (TAB)

This example implements a privacy-first cloud-edge collaborative LLM inference workflow validated with the TAB (Text Anonymization Benchmark) dataset, including:

- Edge-first inference with hard sample mining
- Adaptive privacy desensitization (regex, NER masking, differential privacy)
- Privacy and performance metrics and visualization

## Layout

```
TAB/
  cloud_edge_collaborative_inference_bench/
    benchmarkingjob.yaml
    requirements.txt
    readme.md
    test_algorithms/
      test_algorithms.yaml
      cloud_model/cloud_model.py
      edge_model/edge_model.py
      hard_sample_mining/hard_sample_mining.py
      privacy_desensitization/
        __init__.py
        regex_pseudonymization.py
        ner_masking.py
        differential_privacy.py
        privacy_evaluator.py
    test_env/
      test_env.yaml
      privacy_metrics.py
      performance_metrics.py
      visualization_tools.py
```

See `test_algorithms/test_algorithms.yaml` and `test_env/test_env.yaml` to configure algorithms and environment.


