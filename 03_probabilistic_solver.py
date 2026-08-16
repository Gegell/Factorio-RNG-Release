"""A probabilistic solver for the Uranium-235 enrichment problem.

This was an idea which did not pan out.
The hope was that by tracking the probability of each bit being set we could
converge on a likely value for each bit of the RNG state.
However any information we do get quickly diffuses across all bits, hence removing
any likelihood of us converging onto a single value for each bit.
"""

import numpy as np
from math import log2, floor

from register import ProbabilisticRegister, SymbolicRegister
from rngs import ExtractedRNG


def count_set_bits(upper_value: int, bit_index: int) -> int:
    """Count the number of set bits in the given bit position in the range [0, upper_value]."""
    if upper_value == 0:
        return 0
    next_smaller_exp2 = floor(log2(upper_value))
    next_smaller = 2**next_smaller_exp2
    if bit_index == next_smaller_exp2:
        return upper_value - next_smaller + 1
    elif bit_index > next_smaller_exp2:
        return 0
    else:  # bit_index < next_smaller_exp2
        return next_smaller // 2 + count_set_bits(upper_value - next_smaller, bit_index)


def naive_count(upper_value: int, bit_index: int) -> int:
    """Count the number of set bits in the given bit position in the range [0, upper_value]."""
    return sum((x >> bit_index) & 1 for x in range(upper_value + 1))


def get_bit_probabilities(upper_value: int, lower_value: int = 0, register_width: int = 32) -> list[float]:
    """For each bit, returns the probability that it is set, given a range of possible register values."""
    assert 0 <= lower_value <= upper_value < 2**register_width
    probabilities = [0.0] * register_width

    for bit_index in range(register_width):
        set_bits = count_set_bits(upper_value, bit_index)
        if lower_value > 0:
            set_bits -= count_set_bits(lower_value - 1, bit_index)
        probabilities[-bit_index - 1] = set_bits / (upper_value - lower_value + 1)

    return probabilities


if __name__ == "__main__":
    simulation_rng = ExtractedRNG(
        registers=(
            SymbolicRegister(32, "A", value=0x01234567),
            SymbolicRegister(32, "B", value=0x89ABCDEF),
            SymbolicRegister(32, "C", value=0xFEDCBA98),
        )
    )

    prob_rng = ExtractedRNG(
        registers=(
            ProbabilisticRegister(32),
            ProbabilisticRegister(32),
            ProbabilisticRegister(32),
        )
    )

    # Skip first n iterations
    for _ in range(177 * 2):
        simulation_rng.step()

    # Start simulation
    u235_value = int(simulation_rng.get_int())
    simulation_rng.step()
    u238_value = int(simulation_rng.get_int())
    simulation_rng.step()

    u235_threshold = int(2**32 * 0.003)
    u238_threshold = int(2**32 * 0.997)

    gets_u235 = u235_value <= u235_threshold
    gets_u238 = u238_value <= u238_threshold
    print(f"{gets_u238:1d} {gets_u235:1d}  {u238_value:08x} {u235_value:08x}")

    prior = np.array(
        get_bit_probabilities(lower_value=0, upper_value=u235_threshold)
        if gets_u235
        else get_bit_probabilities(lower_value=u235_threshold + 1, upper_value=2**32 - 1)
    )
    for register in prob_rng.state:
        register._bits = (prior - 0.5) / 3.0 + 0.5

    print(f"Prior entropy: {prob_rng.state[0].entropy():.3f} bits")
    print(f"{prob_rng.state[0]}")
    for _ in range(10):
        prob_rng.step()
        print(f"Entropy: {prob_rng.state[0].entropy():.3f} bits")
        print(f"{prob_rng.state[0]}")
