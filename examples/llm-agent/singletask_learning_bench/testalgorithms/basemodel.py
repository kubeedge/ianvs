<<<<<<< HEAD
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
        self.model = AutoModelForCausalLM.from_pretrained(self.tokenizer_dir, use_auth_token=self.auth_token,device_map=self.config["device"],trust_remote_code=self.config["trust_remote"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir,token=self.auth_token)
    
    def train(self, train_data, **kwargs):
        preprocessed_data = []
        for x, y in zip(train_data.x, train_data.y):
            sample = type('Sample', (object,), {'x': x, 'y': y})()
            processed_sample = self.preprocess(sample, self.MAX_LENGTH, self.tokenizer)
            preprocessed_data.append(processed_sample)

        train_dataset = datasets.Dataset.from_dict({
        "input_ids": [d["input_ids"][1] for d in preprocessed_data],  
        "attention_mask": [d["attention_mask"][1] for d in preprocessed_data],
        "labels": [d["labels"][1] for d in preprocessed_data]
        })
        config_lora=LoraConfig(task_type=TaskType.CAUSAL_LM,
                    lora_alpha = 1,
                    lora_dropout = 0.0
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
        pipe=pipeline("text2text-generation",model=self.model,tokenizer=self.tokenizer)
        y_pred = pipe(data)
        return y_pred


    def evaluate(self, data, **kwargs):
        pass


    def load(self, model_url, **kwargs):
        if model_url:
            logging.info("load model url: ",model_url)

    def save(self, model_path = None):
        pass

    
    def preprocess(self, samples, MAX_LENGTH, tokenizer):
        input_ids,attention_mask,labels=[],[],[]
        prompt=samples.x 
        plan=samples.y
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
            "input_ids":[[None],input_ids],
            "attention_mask":[[None],attention_mask],
            "labels":[[None],labels]
        }
=======
version https://git-lfs.github.com/spec/v1
oid sha256:31177d8ae658da6772bc1a6de53642f6a621e68af5eaa0625fad5133e3c69e08
size 4258
>>>>>>> 9676c3e (ya toh aar ya toh par)
