# Block Management

Blocks are disjoint subsets of nodes. `BlockSpec` handles the index
bookkeeping — mapping between block-local and global state representations,
padding variable-size blocks for JAX array stacking, and verifying consistency.

::: hamon.Block
    options:
        members: false

::: hamon.BlockSpec
    options:
        members:
            - __init__

::: hamon.block_state_to_global

::: hamon.scatter_block_to_global

::: hamon.get_node_locations

::: hamon.from_global_state

::: hamon.make_empty_block_state

::: hamon.verify_block_state
