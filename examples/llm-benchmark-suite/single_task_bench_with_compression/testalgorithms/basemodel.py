from sedna.common.class_factory import ClassFactory, ClassType
from llama_cpp import Llama
import os


@ClassFactory.register(ClassType.GENERAL, alias="LlamaCppModel")
class LlamaCppModel:
    def __init__(self, **kwargs):
        """
        初始化 LlamaCppModel
        """
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("Model path is required.")

        # 初始化 Llama 模型
        self.model = Llama(
            model_path=model_path,
            n_ctx=kwargs.get("n_ctx", 512),
            n_gpu_layers=kwargs.get("n_gpu_layers", 0),
            seed=kwargs.get("seed", -1),
            f16_kv=kwargs.get("f16_kv", True),
            logits_all=kwargs.get("logits_all", False),
            vocab_only=kwargs.get("vocab_only", False),
            use_mlock=kwargs.get("use_mlock", False),
            embedding=kwargs.get("embedding", False),
        )

    def predict(self, data=None, input_shape=None, **kwargs):
        """
        使用模型进行预测

        Args:
            data (list or None): 输入数据，忽略此参数
            input_shape: 未使用
            **kwargs: 其他参数

        Returns:
            dict: 包含预测结果的字典
        """
        # 确保忽略 data 参数，直接在代码中写死 prompt
        prompt = (
            "Q: Name the planets in the solar system? A: "
        )

        # 捕获标准输出，包括 llama-cpp-python 的日志
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            # 调用模型进行生成
            output = self.model(
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 32),
                stop=kwargs.get("stop", ["Q:", "\n"]),
                echo=kwargs.get("echo", True),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 40),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
            )
        # 获取捕获的标准输出内容
        stdout_output = f.getvalue()

        # 解析 timing 信息
        timings = self._parse_timings(stdout_output)

        # 提取生成的文本
        generated_text = output['choices'][0]['text']

        predict_dict = {
            "results": [generated_text],
            "timings": [timings]
        }
        return predict_dict

    def _parse_timings(self, stdout_output):
        """
        解析 llama-cpp-python 输出的时间信息

        Args:
            stdout_output (str): 标准输出内容

        Returns:
            dict: 解析后的时间信息
        """
        import re
        timings = {}
        for line in stdout_output.split('\n'):
            match = re.match(r'llama_print_timings:\s+(.*)\s+=\s+([\d\.]+)\s+ms', line)
            if match:
                key = match.group(1).strip()
                value = float(match.group(2))
                timings[key] = value
        return timings

    def evaluate(self, data, model_path=None, **kwargs):
        """
        评估模型
        """
        if data is None or data.x is None or data.y is None:
            raise ValueError("Evaluation data is None.")

        if model_path:
            self.load(model_path)

        # 进行预测
        predict_dict = self.predict(data.x, **kwargs)

        # 使用指定的评估函数计算指标
        metric_name = kwargs.get("metric_name", "accuracy")
        metric_func = kwargs.get("metric_func")

        if callable(metric_func):
            metric_value = metric_func(data.y, predict_dict["results"])
            return {metric_name: metric_value}
        else:
            raise ValueError(f"Metric function is not callable or not provided.")

    def save(self, model_path):
        # llama-cpp-python 不需要保存模型，因为它使用预训练的模型
        pass

    def load(self, model_url):
        # 模型在初始化时已经加载，这里不需要额外的操作
        pass
    def train(self, train_data, valid_data=None, **kwargs):
        """
        LlamaCpp 不支持训练，此方法可以留空或抛出异常
        """
        raise NotImplementedError("Training is not supported for LlamaCppModel.")
    def train(self, train_data, valid_data=None, **kwargs):
        print("Training is not supported for this model. Skipping training step.")
        return