# Graph Utilities

Block Gibbs needs the free blocks to be **independent sets** — no two nodes in
one block may share a factor — so building a sampler for a new graph starts with
a graph coloring. `rlf_coloring` implements Recursive Largest First, which
produces the color classes that become the free blocks.

The high-level model front doors (`IsingEBM` via `ising_sample`, and the
continuous programs) call this for you; reach for it directly when you are
assembling a custom PGM and need the sampling order yourself.

::: hamon.graph_utils.rlf_coloring
