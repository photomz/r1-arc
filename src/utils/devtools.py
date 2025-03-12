from enum import Enum
from typing_extensions import TypedDict
from devtools.debug import Debug, DebugArgument
from deepseek_tokenizer import ds_token
import tiktoken
from typing import Any, Optional, Callable
import typer


class TokenizerMethods(TypedDict):
    encode: Callable
    to_tokens: Callable


openai_coder = tiktoken.get_encoding("cl100k_base")

TOKENIZERS: dict[str, TokenizerMethods] = {
    "deepseek": {
        "encode": ds_token.encode,
        "to_tokens": ds_token.convert_ids_to_tokens,
    },
    "openai": {
        "encode": openai_coder.encode,
        "to_tokens": openai_coder.decode_single_token_bytes,
    },
}


class TokenArg(DebugArgument):
    def __init__(self, value: Any, *, name: Optional[str] = None, **extra: Any) -> None:
        super().__init__(value, name=name, **extra)
        if isinstance(value, str):
            n_tokens = len(TOKENIZERS["deepseek"]["encode"](value))
            self.extra.append(("tok", n_tokens))  # Deepseek tokens


class HotfixedDebug(Debug):
    output_class = Debug.output_class
    output_class.arg_class = TokenArg


debug = HotfixedDebug()


app = typer.Typer()


@app.command()
def tokenize(text: str, use: str = "deepseek"):
    """Count the number of tokens in a string"""
    coder = TOKENIZERS[use]
    ids = coder["encode"](text)
    tokens = list(map(coder["to_tokens"], ids))

    debug(tokens)


if __name__ == "__main__":
    app()
