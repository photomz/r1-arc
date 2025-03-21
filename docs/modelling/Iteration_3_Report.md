# Iteration 3. Report

_2025-03-18_

## Intro

## Related Work

## Initial Attempts
- SFT baseline by rejection sampling
- Long thinking problem
- Hallucinated verification

## Method
- GRPO
- Environment design: DSL
- Execution Sandbox
- Lexers, Linters, and Parsers
- Reward Function

## Method
- Training Settings

## Results
- Graphs
- Reward Hacking
- No 'Aha' Moment
- Variance Reduction

## Conclusion
- Future Work
- Broader purpose


# (A) Problem: Redefines DSL type signatures, often wrongly
# Solution: Count func defs in Abstract Syntax Tree, and penalize for > 3
def rotate_ccw(angle: Optional[Integer]) -> Callable[[Object], Object]:
   pass
def rotate_cw(angle: Optional[Integer]) -> Callable[[Object], Object]:
   pass
def rotate_left(angle: Optional[Integer]) -> Callable[[Object], Object]:
   pass
...
# (B) Problem: Index chasing by cases makes it easy to introduce off-by-one errors.
# Solution: Small style reward per DSL call that abstracts away indexing
if rel_i == -1 and rel_j == -1:
   output[i][j] = 2
elif rel_i == -1 and rel_j == 0:
   output[i][j] = 3
elif rel_i == -1 and rel_j == 1:
   output[i][j] = 4
elif rel_i == 0 and rel_j == -1:
   output[i][j] = 6
elif rel_i == 0 and rel_j == 1:
...
# (C) Problem: LM loses coherence occasionally at long context sizes
# Solution: Length penalty for > 100 lines.
from example import ovoid, ovoid, to_star, to_apex, to_obstacle, to_ground, to_point, to_obstacle, to_point, to_star, to_apex, to_obstacle, to_point, to_obstacle, to_point, to_star, to_apex ....


# 20
if (
    and 8 in cключи(c, x0)
    and 8 not in cells(x1, F)
    and (max(x2) - min(x2) == x2)
):
    result = integer(x1, x2, "", "", "", "", "", "", "", "", "")
...
# 200
while queue:
    x, y = queue.popleft()
    size += 1
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < height and 0 <= ny < width:
            if grid[nx][ny] == current_value and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))
...
# 700
Following the rule:
- 4 at (1,1) moves down to (3,1).
- 7 at (2,3) moves right beyond the grid, so it stays at (2,3).
- 8 at (1,3) doesn't move.

Thus, the TEST Output is:
    |    |
0->4|    |
0->4|0|0|0->8

</think>
def solve(I):
    x1 = vfrontier(I)          # Identify cells that can move down
    x2 = bordering(I, grid=I)  # Identify cells that are at the bottom edge
    x3 = right(x1)             # Shift cells that can move down to the right
    x4 = differ(x3, x2)        # Replace cells that are at the bottom edge
    O = paint(I, x4)           # Apply the transformation
    return O
  ...