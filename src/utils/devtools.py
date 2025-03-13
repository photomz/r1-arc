from enum import Enum
import sys
from types import FrameType
from typing_extensions import TypedDict
from devtools.debug import Debug, DebugArgument, DebugOutput
from deepseek_tokenizer import ds_token
import tiktoken
from typing import Any, Optional, Callable
import typer
from io import StringIO


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


class TokenDebug(Debug):
    output_class = Debug.output_class
    output_class.arg_class = TokenArg


class LocalsDebug(Debug):
    output_class = Debug.output_class
    output_class.arg_class = DebugArgument

    """Hotwire Debug to capture local vars in caller frame, and return formatted str."""

    def __call__(self, *args: "Any", frame_depth_: int = 2, **kwargs: "Any") -> str:
        output_buffer = StringIO()
        super().__call__(
            *args,
            file_=output_buffer,
            flush_=False,
            frame_depth_=frame_depth_ + 1,
            **kwargs,
        )
        return output_buffer.getvalue().strip()

    def _process(self, args: "Any", kwargs: "Any", frame_depth: int) -> DebugOutput:
        debug_output = super()._process(args, kwargs, frame_depth + 1)

        # Get the current frame to extract locals
        try:
            call_frame: "FrameType" = sys._getframe(frame_depth)
        except ValueError:
            return debug_output  # If frame retrieval fails, return standard output

        # Capture local variables
        locals_args = [
            self.output_class.arg_class(value, name=name)
            for name, value in call_frame.f_locals.items()
        ]
        debug_output.arguments += locals_args
        return debug_output


# Instantiate the new debug class
verbose_debug = LocalsDebug()


debug = TokenDebug()


app = typer.Typer()


@app.command()
def tokenize(text: str, use: str = "deepseek"):
    """Count the number of tokens in a string"""
    coder = TOKENIZERS[use]
    ids = coder["encode"](text)
    tokens = list(map(coder["to_tokens"], ids))

    debug(tokens)


@app.command()
def locals(text: str = "howdy"):
    hidden_var = "aye mate"  # Should show in locals print
    for i in range(1):
        out = verbose_debug(text)
        print(out)


if __name__ == "__main__":
    app()
