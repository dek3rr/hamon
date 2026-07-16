def test_readme_quick_example():
    import jax
    import jax.numpy as jnp

    from hamon import Block, SamplingSchedule, SpinNode, sample_states
    from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init

    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    biases = jnp.zeros((5,))
    weights = jnp.ones((4,)) * 0.5
    beta = jnp.array(1.0)
    model = IsingEBM(nodes, edges, biases, weights, beta)

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.key(0)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

    assert samples[0].shape == (1000, 5)


def test_readme_continuous_example():
    import jax
    import jax.numpy as jnp

    from hamon import Block, GaussianNode, SamplingSchedule, sample_states
    from hamon.models import GaussianEBM, GaussianSamplingProgram, gaussian_init

    nodes = [GaussianNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    model = GaussianEBM(
        nodes,
        edges,
        diag=jnp.full(5, 2.0),
        lin=jnp.zeros(5),
        couplings=jnp.full(4, -0.5),
        beta=jnp.array(1.0),
    )

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = GaussianSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.key(0)
    k_init, k_samp = jax.random.split(key)
    init_state = gaussian_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

    assert samples[0].shape == (1000, 5)
    assert samples[0].dtype == jnp.float32


def test_readme_annealed_example():
    import jax.numpy as jnp

    from hamon import Block, GaussianNode
    from hamon.models import (
        AnnealedEBM,
        DoubleWellEBM,
        DoubleWellSamplingProgram,
        GaussianEBM,
    )

    nodes = [GaussianNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]

    reference = GaussianEBM(
        nodes, [], jnp.full(5, 2.0), jnp.zeros(5), jnp.zeros(0), jnp.array(1.0)
    )
    target = DoubleWellEBM(
        nodes,
        edges,
        barrier=jnp.ones(5),
        lin=jnp.zeros(5),
        couplings=jnp.full(4, -0.6),
        beta=jnp.array(1.0),
    )
    annealed = AnnealedEBM(reference, target, jnp.array(1.0))
    program = DoubleWellSamplingProgram(annealed, free_blocks, clamped_blocks=[])

    # beta_range=(0.0, 1.0) is valid: the beta=0 member is the reference.
    assert annealed.proper_at_beta_zero is True
    assert annealed.beta_affine is True
    assert program.samplers
