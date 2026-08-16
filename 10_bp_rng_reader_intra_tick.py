r"""
This blueprint will perform a bunch of RNG samples in the same tick, to read
out the RNG state. This is useful as this can be used to circumvent some of the
issues we have when reading it out over multiple ticks.

Benefits:
- Faster readout of RNG state
- As far as I know, no RNG calls that could disrupt the computation can occur
  between these machines because they should all be updated in the same order.
  This is especially true if the machines are built together, with no other
  entities inserted between them; they then lie sequentially in the game's
  entity list.

Drawback:
- The construction order must be controlled so that the machines are updated
  in the correct order.

In terms of functionality, this is similar to the `rng_advancer` because we need to
solve a linear equation system to obtain the underlying RNG state.
The equation of interest is:
$$
        [T_1^k   0      0  ]
    P * [  0   T_2^k    0  ] * s^{(t)} = o^{(t+k)}$
        [  0     0    T_3^k]
        \..... = T^k ...../
$$
where we have:
- $P = [I, I, I] \in GF(2)^{32 \times 96}$ is the projection matrix / the matrix
  which combines the 3 LFSR states into a single RNG sample.
- $T_i \in GF(2)^{32 \times 32}$ is the matrix which advances the i-th LFSR one step.
  It is raised to the power of k to advance the LFSR k steps.
- $s^{(t)} \in GF(2)^{96}$ is the RNG state at time t.
- $o^{(t+k)} \in GF(2)^{32}$ is the observed RNG state at time t+k.

Since a single observation is not enough to solve the equation system, we perform
multiple observations, and recombine them into a single equation system.
Moreover, this can also be done to compensate for the fact that each observation
cannot observe the full RNG state, but only part of it (in the worst case, only
the most significant bit).

Lastly we also can reformulate the equation system to be in the form of:
    $P * T^{-k} * s^{(t')} = o^{(t' - k)}$
where we substitute $t' = t + k$. This is neat, as it allows us to immediately
solve the equation system for the next RNG state, given the previous observations,
instead of then having to fast forward the state afterwards.
Also note, that this requires that the transition Matrix is invertible, which is
the case for the LFSR transition matrices due to their construction (each galois
LFSR can be turned into a Fibonacci LFSR and vice versa, where one is the other
but run in reverse).
"""

from draftsman.blueprintable import Blueprint  # type: ignore
from draftsman.classes.group import Group  # type: ignore
from draftsman.prototypes.decider_combinator import DeciderInput, DeciderOutput  # type: ignore
from draftsman.entity import ArithmeticCombinator, DeciderCombinator  # type: ignore
import numpy as np

from blueprints import LUT_constant_combinator, to_int32_t
from all_signals import SIGNAL_LIST
from factorio_rng import build_generation_matrix


def compute_observation_matrix(num_observations: int):
    """Compute the observation matrix for the given number of observations.

    The returned matrix M is of shape (num_observations, 96) and computes the following:
    Given the RNG state s^{(t)} at time t: M @ s^{(t)} = o with elements o_i = o^{(t - i)}
    where o^{(t)} is the observed topmost bit of the RNG call at time t.
    Also 0 <= i < num_observations.
    """
    single_forward = build_generation_matrix(taps=(1,))
    single_LFSR_fwd = [
        single_forward[:, :32],
        single_forward[:, 32:64],
        single_forward[:, 64:],
    ]

    # Need a somewhat custom inverse, as the 32x32 matrices are not invertible
    # the LFSRs are not full rank, but have rank 31, 29, 28 respectively
    # I.e. the LSB are always linearly dependent on the more significant bits
    def reverse_LFSR_mat(mat, rank):
        sub_mat = mat[:rank, :rank]
        sub_inv = sub_mat.inv(iszerofunc=lambda x: x % 2 == 0)
        mat = mat.copy()
        mat *= 0
        mat[:rank, :rank] = sub_inv % 2
        return mat

    single_LFSR_bwd = [
        reverse_LFSR_mat(single_LFSR_fwd[0], 31),
        reverse_LFSR_mat(single_LFSR_fwd[1], 29),
        reverse_LFSR_mat(single_LFSR_fwd[2], 28),
    ]

    # Compute the observation matrix
    observation_mat = np.zeros((num_observations, 32 * 3), dtype=int)
    for mat_idx in range(3):
        bwd = np.array(single_LFSR_bwd[mat_idx], dtype=int)

        # Iterate over 0, -1, -2, -3, ... to get the previous states observations
        transition_mat = np.eye(32, dtype=int)
        for step in range(0, num_observations):
            # Only the first bit is observed, copy the first row of the transition matrix
            observation_mat[-step-1, 32 * mat_idx : 32 * (mat_idx + 1)] = transition_mat[0]
            transition_mat = (bwd @ transition_mat) % 2

    return observation_mat


def compute_inverse_observation_matrix():
    """Compute the inverse observation matrix for the given number of observations.

    The returned matrix M^{-1} is of shape (96, 88) and computes the following:
    Given the observed bits o^{(t-1)} to o^{(t-88)} of the RNG call at time t:
    s^{(t-1)} = M^{-1} @ o
    """
    observation_mat = compute_observation_matrix(88)
    non_empty_cols = np.where(np.any(observation_mat, axis=0))[0]
    to_invert = observation_mat[:, non_empty_cols]
    to_invert = np.hstack([to_invert, np.eye(88, dtype=int)])

    # Invert the matrix
    used_as_pivot = np.zeros(88, dtype=bool)
    for col_idx in range(88):
        # Find all rows with a 1 in the current column
        cancel_mask = to_invert[:, col_idx] == 1
        # Find a new pivot row
        pivot = np.where(cancel_mask & ~used_as_pivot)[0][0]
        used_as_pivot[pivot] = True
        cancel_mask[pivot] = False
        # Eliminate all other rows
        to_invert[cancel_mask] ^= to_invert[pivot]

    # Swap the rows to obtain a diagonal matrix on the left
    for col_idx in range(88):
        pivot = np.argmax(to_invert[:, col_idx])
        to_invert[[col_idx, pivot]] = to_invert[[pivot, col_idx]]

    # Split the matrix into the inverse (and the identity omitted)
    inverse = to_invert[:, 88:]

    # Spread back into the full matrix
    full_inverse = np.zeros((96, 88), dtype=int)
    for i, col_idx in enumerate(non_empty_cols):
        full_inverse[col_idx] = inverse[i]
    return full_inverse


def test_matrices():
    """Test the correctness of the observation and inverse observation matrices."""
    import matplotlib.pyplot as plt

    obs_mat = compute_observation_matrix(88)
    inv_obs_mat = compute_inverse_observation_matrix()

    res = (inv_obs_mat @ obs_mat) % 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].matshow(obs_mat, aspect="equal")
    ax[1].matshow(inv_obs_mat, aspect="equal")
    ax[2].matshow(res, aspect="equal")
    ax[0].set_title("Observation Matrix $M$")
    ax[1].set_title("Inverse Observation Matrix $M^{-1}$")
    ax[2].set_title("$M^{-1} M$")

    # col_checksum = np.sum(inv_obs_mat, axis=0)
    print(inv_obs_mat.T[0].shape)

    observations = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
    state = (inv_obs_mat @ observations) % 2
    registers = state.reshape((3, 32)).astype(np.uint32)
    print(registers)
    reg_values = np.sum(registers * (1 << (31 - np.arange(32, dtype=np.uint32))), axis=1)
    print(reg_values)
    print("{:08x} {:08x} {:08x}".format(*reg_values))

    plt.show()
    exit()


class SingleRNGUnit(Group):
    """Uses the recycler to observe 2 RNG calls using a recipe with 2 items which
    have a 50% chance to be produced, e.g. repair packs."""

    # The Blueprint string which will be modified to contain the linear equation from above
    base_string = (
        "0eNq9k91uo0AMhd/F1xABLVGDtE/SrdAATmMtzFDPkG2KePf1AGHz19WmF1UuIox97PPZ9"
        "FDUHbZM2kHWAzlsIDuJBbBHtmQ0ZOk62TxuNmn6kMRP6ToA1I4coYXsuQetGpRKxvJQ1sh"
        "S2Borr31lD++QxasogANkYbqKhgAqktTp9VMADTEbhsxxh7PwIdddU4hSFg/Bor9V1oWkL"
        "bK73SSdmiSr9KqJEStMFebWqfJXbulDFOMASqMdmzovcKf25MfooSQuO3I5alXUWB0nO4a"
        "lpFr6bomtE7lXrWr/PI86BcKZiMzqDq2P74ldJ5mD79y0ipXzPeFnlyTrCIbhCkByAoDxr"
        "UMr5sNyJ//XCKJzBHN+vqVaiqxPshOTaW+kK/TghhcZkJXd5dq4/NhlNi46/jJ89fBvXgu"
        "YL5G7IBRcoBznCwvSEzste/RnG/03yId7QC63FN8FMoAl4yy6tG0VcdjKBUrLN7EpA0pcG"
        "25Gy2dWfoyBzruMo+FFfmebuLD3eGKvwlJOXcyZRoCNcp86jG58kbeWPGv+3eFI4fTp+dO"
        "vgdjo8BUVh793iNfXPzk9rlTO8cb60jv9Rd/oD2sRF49UhvOpf8ni2kf/ANkW7YI="
    )

    def __init__(
        self,
        id,
        columns,
        results=("electronic-circuit", "iron-gear-wheel"),
        position=(0, 0),
    ):
        super().__init__(id, position=position)
        self.load_from_string(self.base_string)

        # Name the relevant entities
        index_map = {
            0: "recycler",
            1: "inserter",
            2: "out-chest",
            3: "in-chest",
            4: "decider-1",
            5: "decider-2",
        }
        for i, name in index_map.items():
            self.entities[i].id = name

        # Set the decider data
        for result, column, decider in zip(results, columns.T, self.entities[4:6]):
            # Check if no item was produced, in this case, we have a 1 bit in the MSB
            decider.conditions = [DeciderInput(result) == 0]
            signals = [sig for c_val, sig in zip(column, SIGNAL_LIST) if c_val == 1]
            decider.outputs = [DeciderOutput(signal, False, 1) for signal in signals]

        # Do the internal wire connections
        self.add_circuit_connection(
            "green", "decider-1", "decider-2", "output", "output"
        )
        self.add_circuit_connection("red", "out-chest", "decider-2")
        self.add_circuit_connection("red", "decider-1", "decider-2")


class FilteredLUT(Group):
    def __init__(
        self, id, signal_values_lut: dict[str, int], out_signal: str, position=(0, 0)
    ):
        super().__init__(id, position=position)
        condition = DeciderInput("signal-each", {"red"}) != 0
        condition &= DeciderInput("signal-each", {"green"}) != 0
        self.filter = DeciderCombinator(id="filter", position=(0, -1))
        self.filter.conditions = condition
        self.filter.outputs = [
            DeciderOutput(signal=out_signal, copy_count_from_input=True, networks={"red": False, "green": True})
        ]
        self.constants = LUT_constant_combinator(
            "constants", position=(0, 0), values=signal_values_lut
        )
        self.entities.extend([self.filter, self.constants])

        # Connect the entities
        self.add_circuit_connection("green", "constants", "filter")


class DecoderUnit(Group):
    """Takes the sums of all the rows from the single RNG units, computes the XOR
    between them (e.g. only the last bit) and combines the corresponding signals
    to form the 3 32-bit internal RNG states."""

    def __init__(self, id, position=(0, 0)):
        super().__init__(id, position=position)

        # Compute the sum of the cols as XOR (we get the sum -> take only last bit)
        self.combiner = ArithmeticCombinator(
            id="combiner",
            position=(0, 0),
            control_behavior={
                "arithmetic_conditions": {
                    "first_signal": {"name": "signal-each", "type": "virtual"},
                    "operation": "AND",
                    "second_constant": 1,
                    "output_signal": {"name": "signal-each", "type": "virtual"},
                }
            },
        )
        self.entities.append(self.combiner)

        # Compute the merged state using the FilteredLUTs
        self.LUTs = []
        for output_idx, output_signal in enumerate(
            ["signal-A", "signal-B", "signal-C"]
        ):
            signals = SIGNAL_LIST[output_idx * 32 : (output_idx + 1) * 32]
            row_values = {
                sig: to_int32_t(1 << (31 - sig_idx))
                for sig_idx, sig in enumerate(signals)
            }
            self.LUTs.append(
                FilteredLUT(
                    f"lut-{output_idx}",
                    row_values,
                    output_signal,
                    position=(output_idx, -2),
                )
            )
        self.entities.extend(self.LUTs)

        # Connect the entities
        self.add_circuit_connection(
            "red", "combiner", ("lut-0", "filter"), "output", "input"
        )
        for i in range(2):
            a = (f"lut-{i}", "filter")
            b = (f"lut-{i+1}", "filter")
            self.add_circuit_connection("red", a, b)
            self.add_circuit_connection("red", a, b, "output", "output")


class RNGReaderSingleTick(Group):
    def __init__(self, id, position=(0, 0)):
        super().__init__(id, position=position)
        mat = compute_inverse_observation_matrix()

        # Create the SingleRNGUnits
        self.rng_units = [
            SingleRNGUnit(
                f"rng-unit-{i}", mat[:, 2 * i : 2 * i + 2], position=(i * 2, 0)
            )
            for i in range(mat.shape[1] // 2)
        ]
        self.entities.extend(self.rng_units)

        # Create the DecoderUnit to combine the results into the 3 32-bit states
        self.entities.append(DecoderUnit("decoder", position=(-5, 0)))
        self.add_circuit_connection(
            "green",
            ("rng-unit-0", "decider-2"),
            ("decoder", "combiner"),
            "output",
            "input",
        )

        # Connect the units
        for i, rng_unit in enumerate(self.rng_units):
            if i == 0:
                continue
            a = f"rng-unit-{i-1}"
            b = f"rng-unit-{i}"
            self.add_circuit_connection("green", (a, "out-chest"), (b, "out-chest"))
            self.add_circuit_connection("red", (a, "inserter"), (b, "inserter"))
            self.add_circuit_connection(
                "green", (a, "decider-1"), (b, "decider-2"), "output", "output"
            )


if __name__ == "__main__":
    from pyperclip import copy  # type: ignore

    # test_matrices()
    bp = Blueprint()
    bp.label = "RNG Reader Single Tick"
    bp.description = "Reads out the RNG state in a single tick."
    # bp.entities.append(RNGReaderSingleTick("rng-reader"))
    bp.entities.append(DecoderUnit("decoder"))
    copy(bp.to_string())
