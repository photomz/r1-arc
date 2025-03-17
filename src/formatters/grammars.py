from lark import Lark, UnexpectedToken, Tree
from lark.indenter import PythonIndenter
from src.utils import ROOT

# Load Python grammar from Lark
kwargs = dict(postlex=PythonIndenter(), parser="lalr")

larkf = ROOT / "src/formatters/python.lark"
lark_schema = larkf.open().readlines()

pydsl_parser = Lark.open(larkf, start="start", **kwargs)
parse = lambda s, **kw: pydsl_parser.parse(s, on_error=handle_error, **kw)
# python3_parser = Lark.open_from_package(
#     "lark", "python.lark", ["grammars"], start="file_input", **kwargs
# )


class StreamingParser:
    def __init__(self, parser):
        self.parser = parser

    def parse(self, text):
        try:
            return self.parser.parse(text)
        except UnexpectedToken as e:
            # if e.token.type == "$END":
            # Get the last valid tree from the stack
            for item in reversed(e.state.value_stack):
                if isinstance(item, (Tree)):
                    return item
            # If no tree found, create a minimal valid tree
            return Tree("start", [])
            # raise


parser = StreamingParser(pydsl_parser)

# md disable: blockquotes
completion = """
<think>
Hello I'm thinking about ... a hash # 

Wait, Case 2 ...

This is **bold**
With _italics_ and code
* Lists
# Headers
[Links](https://example.com)

Still case 2 ...

Alternative, ... take 3

I'm stuck here.... case 4

</think>
# Solution

My solution is blah blah ... 
```py
def solve(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O
```

Explanation of DSL: ....
""".strip()

tree = pydsl_parser.parse(completion)
print(tree.pretty())

dsl_code = """
def solve(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O
"""

# tree = python3_parser.parse(dsl_code)
# print(tree.pretty())
