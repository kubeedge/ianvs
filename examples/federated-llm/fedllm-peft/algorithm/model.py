<<<<<<< HEAD
# Copyright 2025 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import os, torch, numpy as np, re
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, AutoModel)
from peft import (get_peft_model, LoraConfig, TaskType,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from sedna.common.class_factory import ClassType, ClassFactory
from evaluate import load as load_metric

os.environ["BACKEND_TYPE"] = "TORCH"

# ---------------------------  Dataset  ----------------------------------
class AdDataset(Dataset):
    def __init__(self, attrs, tgts, tokenizer,
                 bos_id, eos_id, mask_id, ignore_index=-100):
        self.attrs = attrs
        self.tgts  = tgts
        self.tok   = tokenizer
        self.BOS   = bos_id
        self.EOS   = eos_id
        self.MASK  = mask_id
        self.IGNORE_INDEX = ignore_index

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, idx):
        attrs, tgt = self.attrs[idx], self.tgts[idx]

        prompt_ids = self.tok(attrs, add_special_tokens=True).input_ids
        tgt_ids = self.tok(tgt, add_special_tokens=True).input_ids

        ids = [self.BOS] + prompt_ids + [self.MASK] + tgt_ids + [self.EOS]
        labels = ([self.IGNORE_INDEX] * (len(prompt_ids) + 2)) + tgt_ids + [self.EOS]
        return ids[-1024:], labels[-1024:]


def build_collate_fn(tokenizer, ignore_index):
    """工厂函数：返回一个闭包 collate(batch)"""
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        ids, labels = zip(*batch)
        ids_pad = tokenizer.pad({"input_ids": ids}, padding=True,
                                return_tensors="pt").input_ids
        labels_pad = tokenizer.pad({"input_ids": labels}, padding=True,
                                   return_tensors="pt").input_ids
        labels_pad[labels_pad == pad_id] = ignore_index
        attn_mask = (ids_pad != pad_id)
        return ids_pad, attn_mask, labels_pad 
    return collate


# -------------------------  Model Wrapper  ------------------------------
@ClassFactory.register(ClassType.GENERAL, alias="fedllm-peft")
class LLMFederatedModel:
    def __init__(self, **kwargs):
        # ---------- hyperparams ----------
        self.batch_size = int(kwargs.get("batch_size", 16))
        self.epochs     = int(kwargs.get("local_epochs", 1))
        self.lr         = float(kwargs.get("learning_rate", 5e-5))
        self.save_dir   = kwargs.get("save_dir", "./project/save_model/fedllm/chatglm_lora")
        self.peft_method = kwargs.get("peft_method", "lora")
        self.device      = torch.device("cpu")

        # ---------- load base model ----------
        self.init_dir = kwargs.get("initial_model_url")
        model_name = kwargs.get("model_name", "THUDM/chatglm-6b")

        if self.init_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(self.init_dir, local_files_only=True, trust_remote_code=True)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(self.init_dir, local_files_only=True, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        # --------- set pad_token if missing ----------
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model.config.pad_token_id = self.tokenizer.eos_token_id

        # ---------- PEFT ----------
        self.model = self.apply_peft(base_model, self.peft_method)
        self.model = self.model.half().float()
        trainables = sum(p.numel() for p in self.model.parameters()
                         if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded with PEFT: {self.peft_method}, lr={self.lr} | "
              f"Trainable / Total params: {trainables:,} / {total:,}")

    # ------------------------------------------------------------
    def apply_peft(self, base_model, method):
        if method == "lora":
            cfg = LoraConfig(
                r=8, lora_alpha=32, lora_dropout=0.05, bias="none",
                target_modules=["query_key_value"],
                task_type=TaskType.CAUSAL_LM)
            return get_peft_model(base_model, cfg)
        elif method == "ptuning":
            pre_len = 20
            proj    = False
            base_model.config.pre_seq_len       = pre_len
            base_model.config.prefix_projection = proj
            model = AutoModel.from_pretrained(
                self.init_dir,
                config=base_model.config,
                trust_remote_code=True
            )
            model.load_state_dict(base_model.state_dict(), strict=False)
            for n, p in model.named_parameters():
                p.requires_grad = n.startswith("transformer.prefix_encoder")
            model = model.half()
            model.transformer.prefix_encoder.float()
            return model
        else:
            raise ValueError(f"Unsupported PEFT method: {method}")

    # ------------------------------------------------------------
    def _unwrap(self):
        return self.model

    def _move_model(self, device):
        self.model = self.model.to(device, non_blocking=True)
        for m in self.model.modules():
            if hasattr(m, "max_seq_len_cached"):
                m.max_seq_len_cached = None
                if hasattr(m, "cos_cached"):
                    m.cos_cached = None
                    m.sin_cached = None

    # -----------------------  train  -----------------------------
    def train(self, train_data, valid_data=None, device_id=0, client_id=0, **kwargs):
        # ---- device ----
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        print(f"Client {client_id} is using device: {self.device}")
        self._move_model(self.device)
        # ---- optimizer ----
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        x, y = train_data
        assert len(x) == len(y), "length of x and y must be equal"
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        mask = self.tokenizer.mask_token_id
        IGNORE_INDEX = -100

        # ---- DataLoader ----
        dataset = AdDataset(x, y, self.tokenizer,
                            bos, eos, mask, IGNORE_INDEX)
        collate_fn = build_collate_fn(self.tokenizer, IGNORE_INDEX)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            drop_last=False)

        for epoch in range(self.epochs):
            total_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
            for ids_tensor, attn_mask, labels_tensor in pbar:
                ids_tensor    = ids_tensor.to(self.device)
                labels_tensor = labels_tensor.to(self.device)
                attn_mask     = attn_mask.to(self.device)
                loss = self.model(input_ids=ids_tensor,
                                  labels=labels_tensor).loss 
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                torch.cuda.empty_cache() 
                total_loss += loss.item() * ids_tensor.size(0)
                pbar.set_postfix(step_loss=f"{loss.item():.4f}")

            mean_loss = total_loss / len(dataset)
            print(f"[Local Epoch {epoch+1} on Client {client_id}] Mean loss: {mean_loss:.4f}")
        self.save()
        self._move_model("cpu")
        # clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        # time.sleep(5)  # wait for GPU memory to be released
        return {"num_samples": len(dataset)}

    # -------------------  inference  ---------------------------
    def _generate_one(self, attrs, max_new_tokens: int = 1024):
        prompt = attrs
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        enc["attention_mask"] = enc["attention_mask"].bool()

        with torch.inference_mode():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True
            )
        full = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return full.replace(prompt, "").strip()


    # -------------------  批量推理（主力函数） -------------------
    def predict(
        self,
        data,                       # List[str]
        kwargs=None
    ):
        self.model.eval()
        self.model.to(self.device)
        outputs = []
        for start in range(0, len(data), self.batch_size):
            prompts = data[start:start + self.batch_size]
            enc = self.tokenizer(
                prompts,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            enc["attention_mask"] = enc["attention_mask"].bool()
            with torch.inference_mode():
                out_ids = self.model.generate(
                    **enc,
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True
                )
            decoded = self.tokenizer.batch_decode(
                out_ids, skip_special_tokens=True
            )
            cleaned = [
                d.replace(p, "").strip() for d, p in zip(decoded, prompts)
            ]
            outputs.extend(cleaned)
        return outputs

    # ------------------  FedAvg weights  ------------------------
    def get_weights(self):
        if self.peft_method == "lora":
            return get_peft_model_state_dict(self._unwrap())
        elif self.peft_method == "ptuning":
            return {
                k[len("transformer.prefix_encoder."):]: v.cpu()
                for k, v in self.model.state_dict().items()
                if k.startswith("transformer.prefix_encoder.")
            }
        else:
            raise ValueError(f"Unknown peft_method: {self.peft_method}")
    
    def set_weights(self, weights):
        if self.peft_method == "lora":
            set_peft_model_state_dict(self._unwrap(), weights)
        elif self.peft_method == "ptuning":
            prefix_state = {
                "transformer.prefix_encoder." + k: v for k, v in weights.items()
            }
            missing, unexpected = self.model.load_state_dict(
                prefix_state, strict=False
            )
            assert not unexpected, f"Unexpected keys: {unexpected}"
        else:
            raise ValueError(f"Unknown peft_method: {self.peft_method}")

    # ------------------------- save -----------------------------
    def save(self):
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)

    # ------------------------- eval -----------------------------
    def eval(self, data, cur_round: int, **kwargs):
        self.model.eval()
        self.model = self.model.to(self.device)
        x, refs = data
        # preds = [self._generate_one(attrs) for attrs in x]
        preds = self.predict(x)
        rouge = load_metric("rouge")
        scores = rouge.compute(predictions=preds,
                               references=refs,
                               use_stemmer=True)
        # clear GPU memory
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        return float(scores["rouge1"])
=======
version https://git-lfs.github.com/spec/v1
oid sha256:10f395d765897c3cbf148e2674f83136c47e1fbe17b075390819f33e154d68f4
size 12637
>>>>>>> 9676c3e (ya toh aar ya toh par)
