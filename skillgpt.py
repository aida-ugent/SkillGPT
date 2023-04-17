import json
import logging
import uuid


import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import STREAM_INTERVAL
from redisearch import RedisMemory

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
    
def load_model(model_path, num_gpus):
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "13GiB" for i in range(num_gpus)},
        }

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # To resolve the error 'tokenizer does not have a padding token'
    model = AutoModelForCausalLM.from_pretrained(
       model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, **kwargs)

    if num_gpus == 1:
        model.cuda()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


class SkillGPT:
    def __init__(self, model_path, model_name, num_gpus, memory_backend):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]
        self.memory_backend = memory_backend
        
        logger.info(f"Loading the model {self.model_name} ...")
        self.tokenizer, self.model, self.context_len = load_model(model_path, num_gpus)

    def get_status(self):
        return {
            "model_name": self.model_name
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids]).cuda(), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                out = model(input_ids=torch.as_tensor([[token]], device="cuda"),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % STREAM_INTERVAL == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break

        del past_key_values
        
    @torch.inference_mode()
    def get_embedding(self, prompt):
        tokenizer, model = self.tokenizer, self.model        
        inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        n_tokens = attention_mask.sum(1, keepdim=True)
        last_layer_hidden_state = model(input_ids.cuda(), output_hidden_states=True, use_cache=True)["hidden_states"][-1].cpu()
        
        res =  list(((attention_mask.unsqueeze(-1) * last_layer_hidden_state).sum(1) / n_tokens).numpy().ravel().astype(float))        
        # del input_ids, attention_mask, last_layer_hidden_state, inputs
        return res
            
    
    def embed_text(self, params):
        prompt = params["prompt"]
        yield json.dumps({"embedding": self.get_embedding(prompt)})                
        
        
    def label_embedding_parquet(self, params):
        text_emb, esco_index, num_relevant = params["embedding"], params["esco_index"], params.get("num_relevant", 5)
        
        text_emb = np.array(text_emb)
        text_emb = text_emb / norm(text_emb)
        
        esco_embs = np.vstack(self.esco_data[esco_index]["emb"].values)
        esco_embs = esco_embs / norm(esco_embs, axis=1)[:,None]
        
        scores = (text_emb @ esco_embs.T).astype(float)
        
        top_k_indices = np.argsort(scores)[-num_relevant:][::-1]
        df_res = self.esco_data[esco_index].iloc[top_k_indices].copy().drop("emb", axis=1)
        df_res["scores"] = list(scores[top_k_indices])

        return df_res   

    def label_embedding_redis(self, params):
        text_emb, esco_index, num_relevant = params["embedding"], params["esco_index"], params.get("num_relevant", 5)
        redis_host, redis_port = params.get("redis_host", "localhost"), params.get("redis_port", "6379")
        memory = RedisMemory(redis_host, redis_port)
        res = memory.get_relevant(text_emb, esco_index, num_relevant)
        return res
        
        
    def label_text_gate(self, params):
        params["embedding"] = self.get_embedding(params["prompt"])
        res = self.label_embedding_redis(params)
        yield json.dumps({"labels": res})
        del params["embedding"], res
                
    def label_embedding_gate(self, params):
        res = self.label_embedding_redis(params)
        yield json.dumps({"labels": res})
        del res
        

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
            
    def embed_text_gate(self, params):
        try:
            for x in self.embed_text(params):
                yield x
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

    def init_esco_embedding_db(self, params):
        # embed
        # save to parquet
        if self.memory_backend == "redis":
            redis_host, redis_port = params.get("redis_host", "localhost"), params.get("redis_port", "6379")
            memory = RedisMemory(redis_host, redis_port, wipe_redis_on_start=True)
            memory.init_esco_embeddings()
        yield "ESCO embedding database is initialized."