<examples>

## Puzzle 1
Input

	A|B|C|D|E|F|G|H|I|J
1	0|0|0|0|9|0|0|0|0|0
2	0|0|0|0|0|0|0|0|0|0
3	0|0|0|0|0|0|0|0|0|0
4	0|0|0|0|8|8|0|0|0|0
5	0|0|0|0|8|8|0|0|0|0
6	0|0|0|0|8|8|0|0|0|0
7	6|0|0|0|8|8|0|0|0|0
8	0|0|0|0|0|0|0|0|0|0
9	0|0|0|0|0|0|0|0|0|0
10	0|0|0|0|0|4|0|0|0|0

Diff

 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- |8->9| -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- |8->6|8->4| -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 

## Pattern
- Color each outer square on center shape to match square it is facing.
copy the colors onto blue square in same location
- Put colors in the central block that exist in the grid. Closest to where they lie outside.
- Place a square on edge of `8` rectangle. The color and position should correspond to the color of the square on the edges of the grid.

## Solver
```py
def solve(I):
    x1 = objects(I, T, F, T)          # Find all distinct grid objects
    x2 = sizefilter(x1, ONE)          # Get single-cell objects only
    x3 = difference(x1, x2)           # Isolate purple rectangle object
    x4 = first(x3)                    # Extract purple rectangle
    x5 = rbind(gravitate, x4)         # Bind rectangle to gravitate
    x6 = compose(crement, x5)         # Movement direction from gravity
    x7 = fork(shift, identity, x6)    # Combine shift with original properties
    x8 = mapply(x7, x2)              # Transform all single cells
    O = paint(I, x8)                  # Paint transformed cells onto grid
    return O
```

## Puzzle 2
Input 1

	A|B|C|D|E|F|G|H|I|J|K|L|M|N
1	8|8|8|8|8|8|8|8|8|8|8|8|8|8
2	8|8|8|8|8|8|8|8|8|8|8|8|8|8
3	8|8|8|8|8|8|8|8|8|8|8|8|8|8
4	8|8|8|8|0|8|8|8|8|8|8|8|8|8
5	8|8|8|0|8|0|8|8|8|8|8|8|8|8
6	8|8|8|8|0|8|2|8|8|8|8|8|8|8
7	8|8|8|8|8|8|8|8|8|8|8|8|8|8
8	8|8|8|8|8|8|8|8|8|8|8|8|8|8
9	8|8|8|8|8|8|8|8|8|8|8|8|8|8
10	8|8|8|8|8|8|8|8|8|8|8|8|8|8
11	8|8|8|8|8|8|8|8|8|8|8|8|8|8
12	8|8|8|8|8|8|8|8|8|8|8|8|8|8

Diff 1 (I->O)

 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- |8->2| -- |8->2| -- | -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- |8->2| -- |8->2| -- | -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- |8->2| -- |8->2| -- | -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- |8->2| -- |8->2| -- | -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- |8->2| -- |8->2| -- | -- 
 -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |8->2| -- |8->2| -- 

## Pattern
- Copy the shape in black using the assigned color in the square. This shape in the new color should be on the diagonal, and should repeat until there are no more grid squares.
- The exact shape of the black blocks needs to be replicated over and over with the color of the block connected to the block, and in that direction.

## Solver
```py
def solve(I):
    x1 = leastcolor(I)                    # Grid -> Integer: Gets least frequent color
    x2 = ofcolor(I, ZERO)                 # (Grid, Integer) -> Indices: Black cell positions
    x3 = ofcolor(I, x1)                   # (Grid, Integer) -> Indices: Target color positions
    x4 = position(x2, x3)                 # (Patch, Patch) -> IntegerTuple: Directional vector
    x5 = fork(connect, ulcorner, lrcorner)  # Creates (Patch -> Indices) function composition
    x6 = x5(x2)                           # Indices: Diagonal line through shape
    x7 = intersection(x2, x6)             # (FrozenSet, FrozenSet) -> FrozenSet: Common points
    x8 = equality(x6, x7)                 # Purpose: Check if shape is diagonal
    x9 = fork(subtract, identity, crement)  # Creates (Numerical -> Numerical) transform
    x10 = fork(add, identity, x9)         # Combines transforms with addition
    x11 = branch(x8, identity, x10)       # (Boolean, Callable, Callable) -> Callable
    x12 = shape(x2)                       # Piece -> IntegerTuple: Gets (height, width)
    x13 = multiply(x12, x4)               # (Numerical, Numerical) -> Numerical: Scale shape
    x14 = apply(x11, x13)                 # Purpose: Transform scaled dimensions
    x15 = interval(ONE, FIVE, ONE)        # Creates Tuple[int]: Range 1-5
    x16 = lbind(multiply, x14)            # Fixes left arg of multiply
    x17 = apply(x16, x15)                 # Purpose: Generate multiple offsets
    x18 = lbind(shift, x2)                # Creates (IntegerTuple -> Patch) shifter
    x19 = mapply(x18, x17)                # Purpose: Create multiple shifted copies
    O = fill(I, x1, x19)                  # (Grid, Integer, Patch) -> Grid: Final fill
    return O
```
</examples>