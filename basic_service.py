import os
import json
import time
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_WORKERS = 1  # Number of llama.cpp worker instances
WORKER_TIMEOUT = 300.0  # Max time to wait for a worker (seconds)
FLUSH_INTERVAL = 0.15  # Streaming buffer flush interval (seconds)
MAX_BUFFER_SIZE = 800  # Max buffer size before forced flush (~160 tokens)


# Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 16384
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20


def download_hf_model(repo_id, filename, models_dir="models"):
    """Download a model file from Hugging Face using the Python library"""
    os.makedirs(models_dir, exist_ok=True)

    local_path = os.path.join(models_dir, filename)
    if os.path.exists(local_path):
        logger.info(f"Model already exists at {local_path}")
        return local_path

    logger.info(f"Downloading {filename} from {repo_id}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded model to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(f"Failed to download model: {e}")


class LlamaWorker:
    """Individual worker that holds a Llama model instance"""

    def __init__(self, worker_id: int, model_path: str, **llama_kwargs):
        self.worker_id = worker_id
        self.model = Llama(
            model_path=model_path,
            **llama_kwargs
        )
        self.busy = False
        logger.info(f"Worker {worker_id} initialized")


class LlamaWorkerPool:
    """Manages a pool of Llama workers"""

    def __init__(self, num_workers: int, model_path: str, **llama_kwargs):
        self.num_workers = num_workers
        self.model_path = model_path
        self.llama_kwargs = llama_kwargs
        self.workers = []
        self.available_workers = asyncio.Queue(maxsize=num_workers)
        self.initialized = False

    async def initialize(self):
        """Initialize all workers - done async to not block startup"""
        if self.initialized:
            return

        logger.info(f"Initializing {self.num_workers} workers...")

        # Create workers one by one to manage VRAM properly
        for i in range(self.num_workers):
            worker = await self._create_worker(i)
            self.workers.append(worker)
            await self.available_workers.put(worker)
            logger.info(f"Worker {i} added to pool")

        self.initialized = True
        logger.info(f"All {self.num_workers} workers initialized")

    async def _create_worker(self, worker_id: int) -> LlamaWorker:
        """Create a worker in an async context"""
        loop = asyncio.get_event_loop()
        # Run the blocking Llama initialization in a thread pool
        worker = await loop.run_in_executor(
            None,
            lambda: LlamaWorker(worker_id, self.model_path, **self.llama_kwargs)
        )
        return worker

    @asynccontextmanager
    async def acquire_worker(self, timeout: float = WORKER_TIMEOUT):
        """Context manager to acquire and automatically release a worker"""
        worker = None
        try:
            # Wait for an available worker with timeout
            worker = await asyncio.wait_for(
                self.available_workers.get(),
                timeout=timeout
            )
            worker.busy = True
            logger.debug(f"Worker {worker.worker_id} acquired")
            yield worker
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail=f"No workers available after {timeout} seconds"
            )
        finally:
            if worker:
                worker.busy = False
                await self.available_workers.put(worker)
                logger.debug(f"Worker {worker.worker_id} released")

    def get_stats(self):
        """Get pool statistics"""
        available = self.available_workers.qsize()
        busy = self.num_workers - available
        return {
            "total_workers": self.num_workers,
            "available_workers": available,
            "busy_workers": busy
        }


class StreamingService:
    """Handles streaming generation with workers"""

    def __init__(self, worker_pool: LlamaWorkerPool):
        self.worker_pool = worker_pool

    async def generate_streaming_response(
            self,
            message_history: List[dict],
            temperature: float = 0.7,
            max_tokens: int = 16384,
            top_p: float = 0.8,
            top_k: int = 20
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using a worker from the pool"""

        async with self.worker_pool.acquire_worker() as worker:
            logger.info(f"Processing request with worker {worker.worker_id}")

            # Buffer for batching tokens
            buffer = ""
            last_flush_time = time.time()

            try:
                # Run the blocking create_chat_completion in executor
                loop = asyncio.get_event_loop()

                # Create a synchronous generator wrapper
                def create_completion():
                    return worker.model.create_chat_completion(
                        messages=message_history,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=0,
                        stream=True,
                        stop=["<|im_end|>"]
                    )

                # Process chunks from the model
                response_generator = await loop.run_in_executor(None, create_completion)

                # Convert to async iteration
                async for chunk in self._async_wrapper(response_generator, loop):
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]

                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']
                            buffer += content

                            # Check if we should flush
                            current_time = time.time()
                            should_flush = (
                                    len(buffer) >= MAX_BUFFER_SIZE or
                                    (current_time - last_flush_time) >= FLUSH_INTERVAL
                            )

                            if should_flush and buffer:
                                yield f"data: {json.dumps({'content': buffer, 'done': False})}\n\n"
                                buffer = ""
                                last_flush_time = current_time

                        elif choice.get('finish_reason') is not None:
                            # Flush any remaining buffer
                            if buffer:
                                yield f"data: {json.dumps({'content': buffer, 'done': False})}\n\n"
                                buffer = ""

                            # Send completion signal
                            yield f"data: {json.dumps({'done': True, 'finish_reason': choice['finish_reason']})}\n\n"
                            break

                # Final flush if needed
                if buffer:
                    yield f"data: {json.dumps({'content': buffer, 'done': False})}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    async def _async_wrapper(self, sync_generator, loop):
        """Wrap a synchronous generator to work with async for"""

        def get_next():
            try:
                return next(sync_generator)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, get_next)
            if chunk is None:
                break
            yield chunk

            # Small yield to prevent blocking the event loop
            await asyncio.sleep(0)


# Global worker pool instance
worker_pool = None
streaming_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global worker_pool, streaming_service

    # Startup
    logger.info("Starting up FastAPI application...")

    # Download model if needed
    model_path = download_hf_model(
        "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        "Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf"
    )

    # Initialize worker pool
    worker_pool = LlamaWorkerPool(
        num_workers=NUM_WORKERS,
        model_path=model_path,
        n_ctx=16384,
        n_gpu_layers=-1,  # Use all GPU layers
        n_threads=8,
        verbose=False,
        n_batch=2048,
        use_mmap=True,
        use_mlock=True,
        f16_kv=True,
        logits_all=False,
    )

    await worker_pool.initialize()
    streaming_service = StreamingService(worker_pool)

    logger.info(f"Application ready with {NUM_WORKERS} workers")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Async LLaMA Worker Pool API",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream chat completion with worker pool"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    message_history = [msg.dict() for msg in request.messages]

    return StreamingResponse(
        streaming_service.generate_streaming_response(
            message_history,
            request.temperature,
            request.max_tokens,
            request.top_p,
            request.top_k
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not worker_pool:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "healthy",
        "workers": worker_pool.get_stats()
    }


@app.get("/workers/stats")
async def get_worker_stats():
    """Get detailed worker pool statistics"""
    if not worker_pool:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = worker_pool.get_stats()
    stats["workers"] = [
        {
            "id": w.worker_id,
            "busy": w.busy
        }
        for w in worker_pool.workers
    ]
    return stats


if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn for async operation
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        access_log=True,
        log_level="info"
    )