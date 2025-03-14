import typer
import asyncio
from dotenv import load_dotenv

from src.utils import FORMATTERS, FormatterNames, debug, TASKS
from src.prompts import render_prompt
from src.providers.models import ModelName, ProviderName, providers
from src.training.env import debug_in_terminal

load_dotenv()
app = typer.Typer()

# fmt = FORMATTERS[FormatterNames.CELL_DIFF]


@app.command()
def prompt(
    id: str = "5521c0d9",
    provider: ProviderName = ProviderName.VLLM,
    model: ModelName = ModelName.DEEPSEEK_REASONER,
    stream: bool = True,
):

    final_prompt = render_prompt(TASKS[id])

    debug(final_prompt)
    messages = [
        {"role": "user", "content": final_prompt},
    ]

    params = {
        "temperature": 0.9,
        "top_p": 0.95,
    }

    async def run():
        # debug("Calling", model)
        return await providers[provider].complete(
            messages, stream=stream, model=model, params=params
        )

    response = asyncio.run(run())
    debug_in_terminal(response.output, id)


if __name__ == "__main__":
    app()
