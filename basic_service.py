import os
import json
import time
import threading
import queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel
from typing import List, Optional
from huggingface_hub import hf_hub_download


# Add these models above your BasicService class
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
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Check if file already exists
    local_path = os.path.join(models_dir, filename)
    if os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        return local_path

    print(f"Downloading {filename} from {repo_id}...")

    try:
        # Download to the specified directory
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )

        print(f"Successfully downloaded model to {downloaded_path}")
        return downloaded_path

    except Exception as e:
        print(f"Download failed: {e}")
        raise RuntimeError(f"Failed to download model: {e}")


class BasicService:
    def __init__(self):
        model_path = download_hf_model("unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
                                       "Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf")

        self.model = Llama(
            model_path=model_path,
            n_ctx=16384,
            n_gpu_layers=-1,
            n_threads=8,
            verbose=False,
            n_batch=2048,  # Increase batch size for faster processing
            use_mmap=True,  # Memory-map the model file
            use_mlock=True,  # Lock model in memory to prevent swapping
            f16_kv=True,  # Use half-precision for key-value cache
            logits_all=False,  # Only compute logits for last token
        )

        print(f"Model loaded successfully")

    def generate_streaming_response(self, message_history, temperature=0.7, max_tokens=16384, top_p=0.8, top_k=20):
        # Configuration for batching
        FLUSH_INTERVAL = 0.15  # 150ms
        MAX_BUFFER_SIZE = 800  # ~160 tokens

        # Shared state between threads
        buffer = ""
        buffer_lock = threading.Lock()
        should_stop = threading.Event()
        force_flush = threading.Event()
        output_queue = queue.Queue()
        error_occurred = threading.Event()
        completion_info = {}

        def flush_worker():
            """Background thread that flushes buffer every 150ms"""
            nonlocal buffer
            last_flush_time = time.time()

            while not should_stop.is_set():
                current_time = time.time()

                # Check if it's time to flush, force flush requested, or if we should stop
                if (current_time - last_flush_time >= FLUSH_INTERVAL) or force_flush.is_set():
                    with buffer_lock:
                        if buffer:  # Only flush if there's content
                            output_queue.put(f"data: {json.dumps({'content': buffer, 'done': False})}\n\n")
                            buffer = ""
                            last_flush_time = current_time

                    # Clear force flush flag after processing
                    if force_flush.is_set():
                        force_flush.clear()

                # Small sleep to prevent busy waiting
                time.sleep(0.01)

        # Start the flush worker thread
        flush_thread = threading.Thread(target=flush_worker, daemon=True)
        flush_thread.start()

        def token_collector():
            """Collects tokens from the model and adds them to buffer"""
            nonlocal buffer, completion_info

            try:
                response = self.model.create_chat_completion(
                    messages=message_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=0,
                    stream=True,
                    stop=["<|im_end|>"],
                )

                for chunk in response:
                    if should_stop.is_set():
                        break

                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]

                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']

                            with buffer_lock:
                                buffer += content

                                # Check if buffer is getting too large
                                if len(buffer) >= MAX_BUFFER_SIZE:
                                    output_queue.put(f"data: {json.dumps({'content': buffer, 'done': False})}\n\n")
                                    buffer = ""

                        elif choice.get('finish_reason') is not None:
                            # Store completion info
                            completion_info['finish_reason'] = choice['finish_reason']

                            # Force flush any remaining buffer content FIRST
                            force_flush.set()

                            # Wait a moment for flush to complete
                            time.sleep(0.05)

                            # Now send completion signal
                            output_queue.put(
                                f"data: {json.dumps({'done': True, 'finish_reason': choice['finish_reason']})}\n\n")

                            # Signal to stop everything
                            should_stop.set()
                            break

            except Exception as e:
                error_occurred.set()

                # Force flush any remaining buffer content before error
                force_flush.set()
                time.sleep(0.05)

                should_stop.set()
                output_queue.put(f"data: {json.dumps({'error': str(e), 'done': True})}\n\n")

        # Start token collection in a separate thread
        collector_thread = threading.Thread(target=token_collector, daemon=True)
        collector_thread.start()

        # Yield responses from the output queue
        try:
            while True:
                try:
                    # Get response with timeout to periodically check if we should stop
                    response = output_queue.get(timeout=0.1)
                    yield response

                    # Check if this was the final message
                    if '"done": true' in response:
                        break

                except queue.Empty:
                    # Check if both threads are done and queue is empty
                    if should_stop.is_set() and output_queue.empty():
                        break
                    continue

        finally:
            # Cleanup: ensure threads are stopped
            should_stop.set()
            force_flush.set()  # Make sure any final content gets flushed

            # Wait for threads to complete (with longer timeout for final flush)
            flush_thread.join(timeout=2.0)
            collector_thread.join(timeout=2.0)


# Create FastAPI app and service instance
app = FastAPI(title="BasicService Streaming API", version="1.0.0")
service = BasicService()


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    message_history = [msg.dict() for msg in request.messages]

    return StreamingResponse(
        service.generate_streaming_response(
            message_history,
            request.temperature,
            request.max_tokens,
            request.top_p,
            request.top_k
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)