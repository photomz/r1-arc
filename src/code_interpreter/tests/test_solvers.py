solve_f25fbde4 = """
def solve(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = upscale(x3, TWO)
    return O
""".strip()


solve_44d8ac46 = """
def solve(I):
    x1 = objects(I, T, F, T)
    x2 = apply(delta, x1)
    x3 = mfilter(x2, square)
    O = fill(I, TWO, x3)
    return O
""".strip()
