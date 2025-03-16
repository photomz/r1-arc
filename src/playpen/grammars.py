from lark import Lark
from lark.indenter import PythonIndenter

# Load Python grammar from Lark
kwargs = dict(postlex=PythonIndenter(), start="start")

# Official Python grammar by Lark
python_parser3 = Lark.open("./src/playpen/python.lark", parser="lalr", **kwargs)

# dsl_grammar = """
# %import python.lark

# // Restrict function calls to only allow DSL function names
# %override atom_expr: DSL_FUNC "(" arguments? ")"

# // Define the set of valid DSL functions
# DSL_FUNC: "valmin" | "argmax" | "argmin" | "hmirror" | "subgrid" | "partition" | "recolor" | "move" | "ofcolor"

# // Restrict valid statements to function definitions, assignments, and return statements
# stmt: funcdef | assignment | return_stmt

# // Keep expressions but only allow simple function calls or variables
# expr: funccall | VAR
# """

# dsl_parser = Lark(dsl_grammar, parser="lalr", **kwargs)


# Example DSL code snippet
dsl_code = """
```py
def solve(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O
```
"""

tree = python_parser3.parse(dsl_code)
print(tree.pretty())
