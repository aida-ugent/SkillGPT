"""
API gateway
"""
import argparse
import asyncio
import os

import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from skillgpt import SkillGPT

load_dotenv()
API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
MODEL_PATH = os.getenv("MODEL_PATH")

app = FastAPI()

global_counter = 0
model_semaphore = None



def release_model_semaphore():
    model_semaphore.release()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = skill_gpt.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/embed_text")
async def embed_text(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = skill_gpt.embed_text_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)

@app.post("/label_text")
async def label_text(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = skill_gpt.label_text_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)

@app.post("/label_embedding")
async def label_embedding(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = skill_gpt.label_embedding_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/init_esco_embedding")
async def init_esco_embedding(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = skill_gpt.init_esco_embedding_db(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/get_status")
async def get_status(request: Request):
    return skill_gpt.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--host", type=str, default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--memory-backend", type=str, default="redis")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--model-name", type=str, default="vicuna-13b")    
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    args = parser.parse_args()    

    skill_gpt = SkillGPT(args.model_path,
                         args.model_name,
                         args.num_gpus,                         
                         args.memory_backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
