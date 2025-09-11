import os
import json
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
        model_path = download_hf_model("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                                       "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")

        self.model = Llama(
            model_path=model_path,
            n_ctx=16384,
            n_gpu_layers=-1,
            n_threads=8,
            verbose=False,
        )

        print(f"Model loaded successfully")

    def generate_streaming_response(self, message_history, temperature=0.7, max_tokens=16384, top_p=0.8, top_k=20):
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
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]

                    if 'delta' in choice and 'content' in choice['delta']:
                        content = choice['delta']['content']
                        yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"

                    elif choice.get('finish_reason') is not None:
                        yield f"data: {json.dumps({'done': True, 'finish_reason': choice['finish_reason']})}\n\n"
                        break

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


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
