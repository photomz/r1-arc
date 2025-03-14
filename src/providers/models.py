from dataclasses import dataclass, field
from enum import Enum
import os
import re
from typing import Optional, AsyncGenerator, Dict, List, Union
import numpy as np
import openai
import typer
import asyncio
import time
from dotenv import load_dotenv
from functools import cached_property
from tenacity import retry, stop_after_attempt, wait_exponential
from src.prompts import render_prompt, TASKS

load_dotenv()

# Since we're making OpenAI-compatible classes
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from src.utils.devtools import debug


class ModelName(str, Enum):
    # DeepSeek China models
    DEEPSEEK_REASONER = "deepseek-reasoner"
    DEEPSEEK_CHAT = "deepseek-chat"

    # Groq. SpecDec has big degradation.
    DEEPSEEK_R1 = "DeepSeek-R1-Distill-Llama-70b"
    DEEPSEEK_R1_SPEC = "DeepSeek-R1-Distill-Llama-70b-SpecDec"

    # Hyperbolic models
    HYPERBOLIC_R1 = "deepseek-ai/DeepSeek-R1"
    # HYPERBOLIC_R1_ZERO = "deepseek-ai/DeepSeek-R1-Zero"

    # Self-host any HF host on vLLM
    R1_QWEN_7B = "unsloth/DeepSeek-R1-Distill-Qwen-14B"


class ProviderName(str, Enum):
    DEEPSEEK_CHINA = "deepseek"
    GROQ = "groq"
    HYPERBOLIC = "hyperbolic"
    VLLM = "vllm"


@dataclass
class Price:
    """Pricing information for models in USD per million tokens"""

    input: float
    output: float
    input_cache: Optional[float] = None
    output_cache: Optional[float] = None

    def __post_init__(self):
        # If cache pricing not available, use regular pricing
        self.input_cache = self.input_cache or self.input
        self.output_cache = self.output_cache or self.output

    def calculate(self, usage: "Usage") -> float:
        """Compute cost in USD"""
        M = 1_000_000
        return (
            (usage.input - usage.input_cache) * self.input
            + (usage.output - usage.output_cache) * self.output
            + usage.input_cache * self.input_cache
            + usage.output_cache * self.output_cache
        ) / M


@dataclass
class Usage:
    input: int
    output: int
    think: int
    input_cache: int = 0
    output_cache: int = 0

    @staticmethod
    def from_openai(u: Optional[openai.types.CompletionUsage]) -> Optional["Usage"]:
        if not u:
            return None
        think = 0
        if d := getattr(u, "completion_tokens_details", 0):
            think = getattr(d, "reasoning_tokens", 0)

        input_cache = 0
        if d := getattr(u, "prompt_tokens_details", 0):
            # Stanford OpenAI format
            input_cache = getattr(d, "cached_tokens", 0)
        elif d := getattr(u, "prompt_cache_hit_tokens", 0):
            # DeepSeek-reasoner format (of 2025/03)
            input_cache = d

        return Usage(
            input=u.prompt_tokens,
            output=u.completion_tokens,
            think=think,
            input_cache=input_cache,
            output_cache=0,
        )


# 1 provider -> n models
# 1 model -> n responses
# 1 model -> 1 pricing
# 1 response -> 1 usage
# 1 response -> n thinking


@dataclass
class FirstResponse:
    fulltext: str
    reasoning: str
    output: str
    cost: float
    usage: Optional[Usage] = None


@dataclass
class Provider:
    """Base LLM Provider"""

    name: ProviderName
    # Each provider supports many models
    models: List[ModelName]
    env_key: str
    base_url: str
    # Each model has its own pricing
    model_prices: Dict[ModelName, Price]
    supports_cache: bool = False
    # Regex to extract <think>(1)</think>(2) from reasoning
    think_tagged = True

    @cached_property
    def client(self) -> AsyncOpenAI:
        """All supported providers are OpenAI-compatible, so far"""
        debug(os.environ[self.env_key])
        return AsyncOpenAI(api_key=os.environ[self.env_key], base_url=self.base_url)

    @classmethod
    def on_new_token(cls, delta) -> None:
        """Hook for new token"""
        print(delta.content or delta.reasoning_content, end="", flush=True)

    @classmethod
    def on_end_token(cls, accum: FirstResponse) -> FirstResponse:
        """Hook for provider-specific post-processing"""
        if cls.think_tagged:
            if m := re.search(r"<think>(.*?)</think>(.*)", accum.fulltext, re.DOTALL):
                # (1) is reasoning, (2) is output
                accum.reasoning = m.group(1)
                accum.output = m.group(2)
        return accum

    async def reduce_stream(
        self, promise: AsyncStream[ChatCompletionChunk], model: ModelName
    ) -> FirstResponse:
        content = ""
        reasoning_content = ""
        usage = None
        t_first_token = debug.timer("Time to first token").start()
        latencies = []
        t_prev = time.time()
        k = 0

        async for chunk in promise:
            if chunk.choices:
                d = chunk.choices[0].delta
                if c := (
                    getattr(d, "content", "") or getattr(d, "reasoning_content", "")
                ):
                    t_now = time.time()
                    if k == 0:
                        # debug(chunk)
                        t_first_token.capture()
                    latencies.append(t_now - t_prev)
                    t_prev = t_now
                    k += 1
                    # Prefix
                    if c == d.content:
                        content += c
                    else:
                        reasoning_content += c
                    # Postfix
                    self.on_new_token(d)

            if u := getattr(chunk, "usage", None):
                usage = Usage.from_openai(u)

        p95_latency = np.percentile(latencies, 95) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        debug(p95_latency, p99_latency)

        return self.on_end_token(
            FirstResponse(
                fulltext=reasoning_content + content,
                reasoning=reasoning_content.strip(),
                output=content.strip(),
                usage=usage,
                cost=self.model_prices[model].calculate(usage) if usage else 0,
            )
        )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=1e3),
        retry=lambda e: isinstance(e, openai.OpenAIError),
    )
    async def complete(
        self, messages, stream=False, model: ModelName = None, params={}
    ) -> FirstResponse:
        with debug.timer(f"Model complete"):

            if model not in self.models:
                print("Model not specified, default to", self.models[0])
                model = self.models[0]
            params["model"] = model.value

            if stream:
                params["stream"] = True
                if model != ModelName.R1_QWEN_7B:
                    params["stream_options"] = {
                        "include_usage": True
                    }  # TODO: Same Providers don't support this kwarg?
            debug(params)
            response = await self.client.chat.completions.create(
                messages=messages, **params
            )
            if stream:
                return await self.reduce_stream(response, model)
            else:
                # response: ChatCompletion = response
                m = response.choices[0].message
                u = Usage.from_openai(response.usage)
                return self.on_end_token(
                    FirstResponse(
                        fulltext=m.content,
                        reasoning=m.reasoning_content.strip(),
                        output=m.content.strip(),
                        usage=u,
                        cost=self.model_prices[model].calculate(u) if u else 0,
                    )
                )


# Price structure updated: 1 Model -> 1 Price


@dataclass
class DeepSeekChina(Provider):
    name: ProviderName = field(default=ProviderName.DEEPSEEK_CHINA)
    models: List[ModelName] = field(
        default_factory=lambda: [ModelName.DEEPSEEK_REASONER, ModelName.DEEPSEEK_CHAT]
    )
    env_key: str = field(default="DEEPSEEK_API_KEY")
    base_url: str = field(default="https://api.deepseek.com/v1")
    # https://api-docs.deepseek.com/quick_start/pricing
    # 75% discount on off-peak 16:30-00:30 UTC (1:30am-5:30pm PDT)
    model_prices: Dict[ModelName, Price] = field(
        default_factory=lambda: {
            ModelName.DEEPSEEK_REASONER: Price(
                input=0.55, output=2.19, input_cache=0.14
            ),
            ModelName.DEEPSEEK_CHAT: Price(input=0.27, output=1.10, input_cache=0.07),
        }
    )
    supports_cache: bool = field(default=True)
    think_tagged: bool = field(default=False)


@dataclass
class Groq(Provider):
    name: ProviderName = field(default=ProviderName.GROQ)
    # SpecDec has bad degradation: Aggressive prompt caching, doesn't respect temperature.
    models: list[ModelName] = field(
        default_factory=lambda: [ModelName.DEEPSEEK_R1]  # ModelName.DEEPSEEK_R1_SPEC]
    )
    env_key: str = field(default="GROQ_API_KEY")
    base_url: str = field(default="https://api.groq.com/openai/v1")
    model_prices: Dict[ModelName, Price] = field(
        default_factory=lambda: {
            ModelName.DEEPSEEK_R1: Price(input=0.75, output=0.99),
            ModelName.DEEPSEEK_R1_SPEC: Price(input=0.75, output=0.99),
        }
    )
    supports_cache: bool = field(default=True)


@dataclass
class Hyperbolic(Provider):
    name: ProviderName = field(default=ProviderName.HYPERBOLIC)
    models: list[ModelName] = field(
        default_factory=lambda: [
            ModelName.HYPERBOLIC_R1,
            # ModelName.HYPERBOLIC_R1_ZERO,
        ]
    )
    env_key: str = field(default="HYPERBOLIC_API_KEY")
    base_url: str = field(default="https://api.hyperbolic.xyz/v1")
    model_prices: Dict[ModelName, Price] = field(
        default_factory=lambda: {
            ModelName.HYPERBOLIC_R1: Price(input=2, output=2),
            # ModelName.HYPERBOLIC_R1_ZERO: Price(input=2, output=2),
        }
    )
    supports_cache: bool = field(default=False)


@dataclass
class vLLM(Provider):
    name: ProviderName = field(default=ProviderName.VLLM)
    models: list[ModelName] = field(
        default_factory=lambda: [
            ModelName.R1_QWEN_7B,
        ]
    )
    env_key: str = field(default="STUB")
    base_url: str = field(default="http://localhost:8000/v1")
    model_prices: Dict[ModelName, Price] = field(
        default_factory=lambda: {
            ModelName.R1_QWEN_7B: Price(input=0, output=0),
        }
    )
    supports_cache: bool = field(default=True)


providers: dict[ProviderName, Provider] = {
    ProviderName.DEEPSEEK_CHINA: DeepSeekChina(),
    ProviderName.GROQ: Groq(),
    ProviderName.HYPERBOLIC: Hyperbolic(),
    ProviderName.VLLM: vLLM(),
}
