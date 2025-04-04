# SPDX-License-Identifier: Apache-2.0
"""
An example shows how to generate chat completions from reasoning models
like DeepSeekR1.

To run this example, you need to start the vLLM server with the reasoning 
parser:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
     --enable-reasoning --reasoning-parser deepseek_r1
```

Unlike openai_chat_completion_with_reasoning.py, this example demonstrates the
streaming chat completions feature.

The streaming chat completions feature allows you to receive chat completions
in real-time as they are generated by the model. This is useful for scenarios
where you want to display chat completions to the user as they are generated
by the model.

Remember to check content and reasoning_content exist in `ChatCompletionChunk`,
content may not exist leading to errors if you try to access it.
"""

from openai import AsyncOpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


async def main():
    models = await client.models.list()
    model = models.data[0].id

    debug(model)

    messages = [
        # {"role": "system", "content": "Answer like a pirate."},
        {"role": "user", "content": "What is the meaning of life"},
    ]
    config = {
        "model": model,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
    }
    debug(config)
    stream = await client.chat.completions.create(messages=messages, **config)

    print("client: Start streaming chat completions...")
    printed_reasoning_content = False
    printed_content = False

    async for chunk in stream:
        reasoning_content = None
        content = None
        # Check the content is reasoning_content or content
        if hasattr(chunk.choices[0].delta, "reasoning_content"):
            reasoning_content = chunk.choices[0].delta.reasoning_content
        elif hasattr(chunk.choices[0].delta, "content"):
            content = chunk.choices[0].delta.content

        if reasoning_content is not None:
            if not printed_reasoning_content:
                printed_reasoning_content = True
                print("reasoning_content:", end="", flush=True)
            print(reasoning_content, end="", flush=True)
        elif content is not None:
            if not printed_content:
                printed_content = True
                print("\ncontent:", end="", flush=True)
            print(content, end="", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
