# Litmus sources of the Dist line

One file per row of [the litmus table](../../documents/dist/LitmusTable.md),
named by the row ID. Each file is the program of its row in the front-end
language the example comes from: Triton-distributed at `7908e4e` or TileScale
at `71e7145`. The header comment gives the row, the source of the example and
the launch the row assumes.

These files are sources for the tickets that run the rows on the event model
and on the GPU. Nothing here runs them and nothing imports them; the only
check is `python3 -m py_compile`.
