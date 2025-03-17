from deepseek_tokenizer import ds_token
from src.formatters.grammars import parser

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


def test_streaming(snapshot):
    token_ids = ds_token.encode(completion)
    token_stream = [ds_token.decode(token_ids[:i]) for i in range(len(token_ids))]

    parses = []
    for pre in token_stream:
        tree = parser.parse(pre)
        parses.append(tree.pretty() if tree else "No parse")
    assert parses == snapshot
