"""
main.py
By Nick Bukovec
dspy server to answer queries w/ updates via server-sent events
"""
import os
import dspy
from dotenv import load_dotenv
import asyncio
import uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import functools
import boto3
import janus
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import json
from contextlib import suppress
from modules import BasicRAG, BasicQA, ChainOfThoughtQA, MultiHopSearch
from clients import EmittableAWSMistral, EmittableColBERTv2
import traceback

COLBERT_URL = os.getenv("COLBERT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
bedrock = dspy.Bedrock(region_name=AWS_REGION)
haiku = dspy.AWSAnthropic(bedrock, "anthropic.claude-3-haiku-20240307-v1:0", max_new_tokens=200)
# colbert = dspy.ColBERTv2(url="http://37.27.89.182:8893/api/search")
dspy.configure(lm=None, rm=None)
load_dotenv()
app = FastAPI()

def get_new_queue():
  return janus.Queue()

async def asyncify(fn, lm, rm, *args, **kwargs):
    def with_context(fn, lm=lm, rm=rm, *args, **kwargs):
        with dspy.context(lm=lm, rm=rm):
            return fn(*args, **kwargs)
    return await run_in_threadpool(functools.partial(with_context, fn, lm=lm, rm=rm, *args, **kwargs))

async def dspy_wrapper(model, query, queue: janus.SyncQueue):
    lm = EmittableAWSMistral(queue, aws_provider=bedrock, model="mistral.mistral-7b-instruct-v0:2", max_output_tokens=800)
    model.lm = lm
    rm = EmittableColBERTv2(queue, url=COLBERT_URL)
    model.rm = rm
    try:
        res = await asyncify(model, lm, rm, question=query)
        queue.put_nowait(dict(source="server", to="client", message=res.answer))
    except Exception as e:
        queue.put_nowait(dict(source="error", message=str(e)))
        print(traceback.format_exc())
    finally:
        queue.put_nowait("<SENTINEL>")

@app.get('/multihop-optimized')
async def multi_hop_optimized(query: str):
    queue = get_new_queue()
    m = MultiHopSearch()
    m.load('compiled_5.json')
    task = asyncio.create_task(dspy_wrapper(m, query, queue.sync_q))
    async def stream():
        data = await queue.async_q.get()
        queue.async_q.task_done()
        try:
            while data != "<SENTINEL>":
                yield {"data": json.dumps(data)}
                data = await queue.async_q.get()
                queue.async_q.task_done()
        except asyncio.CancelledError:
            print("Exception caught")
            raise
        finally:
            print(queue.async_q.qsize())
            with suppress(asyncio.CancelledError):
                await asyncio.gather(queue.async_q.join(), task)
                print("Streaming should be pretty complete by now")
    return EventSourceResponse(stream())

@app.get("/multihop")
async def multi_hop(query: str):
    queue = get_new_queue()
    task = asyncio.create_task(dspy_wrapper(MultiHopSearch(), query, queue.sync_q))
    async def stream():
        data = await queue.async_q.get()
        queue.async_q.task_done()
        try:
            while data != "<SENTINEL>":
                yield {"data": json.dumps(data)}
                data = await queue.async_q.get()
                queue.async_q.task_done()
        except asyncio.CancelledError:
            print("Exception caught")
            raise
        finally:
            print(queue.async_q.qsize())
            with suppress(asyncio.CancelledError):
                await asyncio.gather(queue.async_q.join(), task)
                print("Streaming should be pretty complete by now")
    return EventSourceResponse(stream())


@app.get("/basic-rag")
async def basic_rag(query: str):
    queue = get_new_queue()
    task = asyncio.create_task(dspy_wrapper(BasicRAG(), query, queue.sync_q))
    async def stream():
        data = await queue.async_q.get()
        queue.async_q.task_done()
        try:
            while data != "<SENTINEL>":
                yield {"data": json.dumps(data)}
                data = await queue.async_q.get()
                queue.async_q.task_done()
        except asyncio.CancelledError:
            print("Exception caught")
            raise
        finally:
            print(queue.async_q.qsize())
            with suppress(asyncio.CancelledError):
                await asyncio.gather(queue.async_q.join(), task)
                print("Streaming should be pretty complete by now")
    return EventSourceResponse(stream())


@app.get("/test")
async def test():
    async def mystream():
        yield {"data": "foobar"}
    return EventSourceResponse(mystream())

@app.get("/cot-qa")
async def cot_qa(query: str):
    queue = get_new_queue()
    task = asyncio.create_task(dspy_wrapper(ChainOfThoughtQA(), query, queue.sync_q))
    async def stream():
        data = await queue.async_q.get()
        queue.async_q.task_done()
        try:
            while data != "<SENTINEL>":
                yield {"data": json.dumps(data)}
                data = await queue.async_q.get()
                queue.async_q.task_done()
        except asyncio.CancelledError:
            print("Exception caught")
            raise
        finally:
            print(queue.async_q.qsize())
            with suppress(asyncio.CancelledError):
                await asyncio.gather(queue.async_q.join(), task)
                print("Streaming should be pretty complete by now")
    return EventSourceResponse(stream())

@app.get("/base-qa")
async def base_model_with_prompt(query: str):
    queue = get_new_queue()
    task = asyncio.create_task(dspy_wrapper(BasicQA(), query, queue.sync_q))

    async def stream():
        data = await queue.async_q.get()
        queue.async_q.task_done()
        try:
            while data != "<SENTINEL>":
                yield {"data": json.dumps(data)}
                data = await queue.async_q.get()
                queue.async_q.task_done()
        except asyncio.CancelledError:
            print("Exception caught")
            raise
        finally:
            print(queue.async_q.qsize())
            with suppress(asyncio.CancelledError):
                await asyncio.gather(queue.async_q.join(), task)
                print("Streaming should be pretty complete by now")

    return EventSourceResponse(stream())

# TODO: restrict origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/base-model", response_class=EventSourceResponse)
async def base_model(query: str):
    queue = get_new_queue()
    task = asyncio.create_task(dspy_wrapper(dspy.Predict("question -> answer"), query, queue.sync_q))
    async def stream():
        data = await queue.async_q.get()
        queue.async_q.task_done()
        try:
            while data != "<SENTINEL>":
                yield {"data": json.dumps(data)}
                data = await queue.async_q.get()
                queue.async_q.task_done()
        except asyncio.CancelledError:
            print("Exception caught")
            raise
        finally:
            print(queue.async_q.qsize())
            with suppress(asyncio.CancelledError):
                await asyncio.gather(queue.async_q.join(), task)
                print("Streaming should be pretty complete by now")
    return EventSourceResponse(stream())

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")