import itertools
import json

from typing import Any, List, Optional


import numpy as np
import pandas as pd
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from tqdm import tqdm

from constants import ESCO_EMBEDDINGS_DIR, REDIS_ESCO_INDICES, VECTOR_SIZE

SCHEMA = [
    TextField("data"),
    VectorField(
        "embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_SIZE,
            "DISTANCE_METRIC": "COSINE"
        }
    ),
]

def chunk(it, size):
    it = iter(it)
    while True:
        p = dict(itertools.islice(it, size))
        if not p:
            break
        yield p

class RedisMemory:
    def __init__(self, redis_host, redis_port, redis_password=None, wipe_redis_on_start=False):        
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=0  # Cannot be changed
        )           
        if wipe_redis_on_start:
            self.redis.flushall()
            self.init_esco_embeddings()
        self.vec_nums = {}
        for esco_index in REDIS_ESCO_INDICES:
            existing_vec_num = self.redis.get(f'{esco_index}-vec_num')
            self.vec_nums[esco_index] = int(existing_vec_num.decode('utf-8')) if existing_vec_num else 0
        print("Number of redis entries:\n", self.vec_nums)

    def init_esco_embeddings(self) -> None:
        def chunk(it, size):
            it = iter(it)
            while True:
                p = dict(itertools.islice(it, size))
                if not p:
                    break
                yield p

        for esco_index in REDIS_ESCO_INDICES:            
            try:                
                self.redis.ft(f"{esco_index}").create_index(
                    fields=SCHEMA,
                    definition=IndexDefinition(
                        prefix=[f"{esco_index}:"],
                        index_type=IndexType.HASH
                        )
                    )
            except Exception as e:
                print("Error creating Redis search index: ", e)
                continue
            print("Index:", esco_index)
            
            df_data = pd.concat((pd.read_parquet(f"{ESCO_EMBEDDINGS_DIR}/df_{esco_index}_{language}.parquet") for language in ["en", "nl", "fr"]))            
            data_records, esco_embeddings = df_data.drop(columns=["emb"]).to_dict("records") , np.vstack(df_data["emb"])            
            vec_num = len(esco_embeddings)            
            for batch in chunk(zip(range(vec_num), zip(data_records, esco_embeddings)), 10000):
                pipe = self.redis.pipeline(transaction=False)
                for key, (data, embedding) in batch.items():                    
                    pipe.hset(f"{esco_index}:{key}", mapping={b"data": json.dumps(data), "embedding": embedding.astype(np.float32).tobytes()})                    
                pipe.execute()            
            print(f"Inserting {vec_num} data entries into memory.")            
            self.redis.set(f'{esco_index}-vec_num', vec_num)
        print("Total number of keys:", len(self.redis.keys()))
            
    def get(self, data: str) -> Optional[List[Any]]:
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        self.redis.flushall()
        return "Obliviated"

    def get_relevant(
        self,
        query_embedding: List[float],
        esco_index: str,
        num_relevant: int = 5
    ) -> Optional[List[Any]]:     
        base_query = f"*=>[KNN {num_relevant} @embedding $vector AS vector_score]"
        query = Query(base_query).return_fields(
            "data",
            "vector_score"
        ).sort_by("vector_score").dialect(2)
        query_vector = np.array(query_embedding).astype(np.float32).tobytes()

        try:
            results = self.redis.ft(f"{esco_index}").search(
                query, query_params={"vector": query_vector}
            )
        except Exception as e:
            print("Error calling Redis search: ", e)
            return None        
        return [result.data for result in results.docs]

    def get_stats(self):
        for memory_index in REDIS_ESCO_INDICES:
            return print(self.redis.ft(f"{memory_index}").info())
