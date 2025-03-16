from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Input, Pretty
from textual.reactive import reactive
from typing import List

from src.utils.devtools import TOKENIZERS


class TokenCounterApp(App):
    """A Textual app that shows token counts and tokens in real-time."""

    # Reactive variable to store the current input text
    current_text = reactive("")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Input(placeholder="Type text to analyze tokens...", id="text_input"),
            Pretty([]),  # Will hold our token analysis
            id="main_container",
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update the token analysis whenever the input changes."""
        self.current_text = event.value

    def watch_current_text(self, text: str) -> None:
        """React to changes in the input text."""
        analysis = []

        for tokenizer_name, tokenizer in TOKENIZERS.items():
            tokens = tokenizer["encode"](text)
            token_strings = [tokenizer["to_tokens"](token) for token in tokens]

            result = {
                "tokenizer": tokenizer_name,
                "token_count": len(tokens),
                "tokens": token_strings,
            }
            analysis.append(result)

        # Update the Pretty widget with the new analysis
        pretty_widget = self.query_one(Pretty)
        pretty_widget.update(analysis)


if __name__ == "__main__":
    app = TokenCounterApp()
    app.run()
