import os
import zipfile
import logging
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from peft import LoraConfig,get_peft_model,TaskType,PeftModel
from transformers import AutoModelForCausalLM,TrainingArguments,Trainer,pipeline,AutoTokenizer,DataCollatorForSeq2Seq
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.common.log import LOGGER
from functools import partial
import datasets
import json
import os

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'

@ClassFactory.register(ClassType.GENERAL, alias="LLM_agent")
class BaseModel:
    def __init__(self, **kwargs):
        config=kwargs.get("config")
        with open(config, 'r', encoding='utf-8') as file:
            self.config = json.load(file)
        train_config=kwargs.get("train_config")
        with open(train_config, 'r', encoding='utf-8') as file:
            self.train_config = json.load(file)

        self.tokenizer_dir = self.config["tokenizer_dir"]
        self.auth_token=self.config["auth_token"]
        self.token_factor=self.config["token_factor"]
        self.MAX_LENGTH = 128
        self.model = AutoModelForCausalLM.from_pretrained(self.tokenizer_dir, token=self.auth_token,device_map=self.config["device"],trust_remote_code=self.config["trust_remote"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir,token=self.auth_token)

    def train(self, train_data, **kwargs):
        preprocessed_data = []
        for x, y in zip(train_data.x, train_data.y):
            processed_sample = self.preprocess(str(x), str(y), self.MAX_LENGTH, self.tokenizer)
            preprocessed_data.append(processed_sample)

        train_dataset = datasets.Dataset.from_dict({
        "input_ids": [d["input_ids"] for d in preprocessed_data],
        "attention_mask": [d["attention_mask"] for d in preprocessed_data],
        "labels": [d["labels"] for d in preprocessed_data]
        })
        config_lora=LoraConfig(task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha = 32,
                    lora_dropout = 0.05
                    )
        model=get_peft_model(self.model,config_lora)
        half = self.train_config["half_lora"]
        if half==True:
            model=model.half()
        del self.train_config["half_lora"]
        args=TrainingArguments(adam_epsilon=(1e-4 if half else 1e-8)
                       ,**self.train_config)
        trainer=Trainer(model=model,args=args,data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer,padding=True),train_dataset=train_dataset, eval_dataset=None)
        trainer.train()
        self.model = trainer.model
        return self.model

    from transformers import pipeline
    def predict(self, data, **kwargs):
        results = []
        for text in data:
            prompt="\n".join(["user: ", str(text)])+"\n\nassistant: "
            inputs=self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.MAX_LENGTH)
            input_len=inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs=self.model.generate(**inputs, max_new_tokens=8, pad_token_id=self.tokenizer.eos_token_id)
            new_tokens=outputs[0][input_len:]
            decoded=self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            decoded=decoded.strip().split()[0] if decoded.strip() else decoded
            results.append(decoded)
        return results


    def evaluate(self, data, **kwargs):
        pass


    def load(self, model_url, **kwargs):
        if model_url:
            logging.info("load model url: ",model_url)

    def save(self, model_path = None):
        pass


    def preprocess(self, prompt=None, plan=None, MAX_LENGTH=None, tokenizer=None):
        if prompt is None:
            return None
        input_ids,attention_mask,labels=[],[],[]
        instruction=tokenizer("\n".join(["user: ",prompt])+"\n\nassistant: ",add_special_tokens=False)
        response=tokenizer(plan,add_special_tokens=False)
        input_ids=instruction["input_ids"]+response["input_ids"]+[tokenizer.eos_token_id]
        attention_mask=instruction["attention_mask"]+response["attention_mask"]+[1]
        labels=len(instruction["input_ids"])*[-100]+response["input_ids"]+[tokenizer.eos_token_id]
        if len(labels)>MAX_LENGTH:
            input_ids=input_ids[:MAX_LENGTH]
            attention_mask=attention_mask[:MAX_LENGTH]
            labels=labels[:MAX_LENGTH]
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels
        }
