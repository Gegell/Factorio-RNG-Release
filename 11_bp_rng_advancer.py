r"""Blueprint generation for RNG advancer.

This blueprint will take the 3 states of the internal LFSR registers of the
RNG and produce the future N values of that register.

This is achieved by basically implementing matrix multiplication
    $$T^i * s^{(t)} = s^{(t+i)}$$
where $T$ is the transition matrix of the LFSR, $s^{(t)}$ is the state of the LFSR
at time $t$, and $s^{(t+i)}$ is the state of the LFSR at time $t+i$.
Note that this all happens in the finite field $GF(2)$, meaning that multiplication
= AND and addition = XOR.

We can write the matrix multiplication A*x = b as
    $$b = \sum_{i=0}^{n-1} A_i * x_i$$
where x_k is the k-th bit of x and A_i is the i-th column of A.
Thus our implementation will look as follows:
1. In parallel take the A_i columns of the transition matrix where x_i = 1. (1-tick)
2. XOR all the columns together using a tree structure. (log2(32) = 5-ticks)

To obtain the future N values of the LFSR, we simply multiply the state of the LFSR
by the transition matrices $T^i$ for $i = 1, 2, ..., N$.
-> These are stored in constant combinators, and are what is the main part of the blueprint.
"""

from math import ceil, log2
from blueprints import to_int32_t
from register import IntegerRegister
from rngs import ExtractedRNG
import numpy as np

from draftsman.classes.group import Group  # type: ignore
from draftsman.entity import DeciderCombinator, ArithmeticCombinator  # type: ignore
from draftsman.blueprintable import Blueprint  # type: ignore

from blueprints import LUT_constant_combinator
from all_signals import SIGNAL_LIST


def compute_skip_matrices(N: int, skip: int = 0) -> np.ndarray:
    """Compute the first N transition matrices of the RNG.
    Defaults to starting with the 0-th transition matrix, i.e. identity matrix.
    If skip > 0, then start with the skip-th transition matrix instead.
    """
    matrices = np.zeros((N, 3, 32), dtype=np.uint32)
    for column in range(32):
        start_bit = 1 << column
        rng = ExtractedRNG(
            registers=(
                IntegerRegister(32, start_bit),
                IntegerRegister(32, start_bit),
                IntegerRegister(32, start_bit),
            )
        )
        for _ in range(skip):
            rng.step()
        for step in range(N):
            matrices[step][0][column] = rng.state[0].get_int()
            matrices[step][1][column] = rng.state[1].get_int()
            matrices[step][2][column] = rng.state[2].get_int()
            rng.step()

    return matrices


def test():

    def to_bit_matrix(in_):
        """Convert a matrix of integers with shape D into a Dx32 matrix of bits."""
        matrix = np.atleast_1d(in_)
        bits = np.ones(32, dtype=np.uint32) << np.arange(32)
        return (matrix[..., None] & bits).astype(bool)

    def show_bit_matrix(matrix):
        import matplotlib.pyplot as plt

        plt.matshow(matrix[-1:0:-1, -1:0:-1], cmap="gray")
        plt.show()

    mat = compute_skip_matrices(3)
    bit_mat = to_bit_matrix(mat[0, 0, :])

    state = to_bit_matrix(0x0175D1C8).astype(np.uint32).flatten()
    advanced = (bit_mat @ state) & 1
    next_state = np.sum(advanced << np.arange(32), axis=0)
    expected = 0x5D1C9769
    print(state)
    print(f"{next_state:08x} == {expected:08x} ? {next_state == expected}")

    show_bit_matrix(bit_mat.T)


def ordinal_suffix(n) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n}-st"
    if n % 10 == 2 and n % 100 != 12:
        return f"{n}-nd"
    if n % 10 == 3 and n % 100 != 13:
        return f"{n}-rd"
    return f"{n}-th"


class ColumnMultiplier(Group):
    def __init__(self, id, skip_matrices, matrix_idx, column_idx, position=(0, 0)):
        """Initialize the column multiplier with the given columns."""
        assert 0 <= matrix_idx < skip_matrices.shape[1]
        assert 0 <= column_idx < skip_matrices.shape[2]

        super(ColumnMultiplier, self).__init__(id, position=position)
        self.column_idx = column_idx
        self.matrix_idx = matrix_idx
        self.columns = skip_matrices[:, matrix_idx, column_idx].astype(np.int32)
        self._add_entities()

    def _add_entities(self):
        """Add the entities required for this group."""
        cc = LUT_constant_combinator(
            id="cc",
            position=(0, 0),
            values=self.columns,
            allow_quality=True,
            description=(
                f"Columns {self.column_idx} of the {ordinal_suffix(self.matrix_idx)} transition matrices."
                f" Includes all RNG steps from 0 to {len(self.columns)-1}."
            ),
        )
        filter_dc = DeciderCombinator(
            id="filter",
            player_description=(
                f"Returns the result from multiplying the {ordinal_suffix(self.column_idx)} column of all"
                f" transition matrices with the {ordinal_suffix(self.column_idx)} LFSR state bit."
            ),
            position=(0, -1),
            control_behavior={
                "decider_conditions": {
                    "conditions": [
                        {
                            "first_signal": SIGNAL_LIST[31 - self.column_idx],
                            "constant": 0,
                            "comparator": "!=",
                            "first_signal_networks": {"red": True, "green": False},
                        }
                    ],
                    "outputs": [
                        {
                            "signal": "signal-everything",
                            "copy_count_from_input": True,
                            "networks": {"red": False, "green": True},
                        }
                    ],
                }
            },
        )
        self.entities.extend([cc, filter_dc])
        self.add_circuit_connection("green", 0, 1)


class MatrixMultiplier(Group):
    def __init__(self, id, skip_matrices, matrix_idx, position=(0, 0)):
        """Initialize the matrix multiplier with the given matrices."""
        super(MatrixMultiplier, self).__init__(id, position=position)
        self.skip_matrices = skip_matrices
        self.matrix_idx = matrix_idx
        self._add_entities()

    def _add_entities(self):
        """Add the entities required for this blueprint."""
        cols = self.skip_matrices.shape[2]
        for column_idx in range(cols):
            y = 3 - (column_idx % 2) * 3
            x = column_idx // 2
            self.entities.append(
                ColumnMultiplier(
                    id=f"col-{column_idx}",
                    skip_matrices=self.skip_matrices,
                    matrix_idx=self.matrix_idx,
                    column_idx=column_idx,
                    position=(x, y),
                )
            )
        for i in range(2, cols):
            cur = f"col-{i}"
            prev = f"col-{i-2}"
            self.add_circuit_connection("red", (cur, "filter"), (prev, "filter"))
        self.add_circuit_connection(
            "red",
            (f"col-{cols-1}", "filter"),
            (f"col-{cols-2}", "filter"),
        )

    def get_output(self, column_idx):
        """Get the outputting entity of the given column index."""
        return (self.id, f"col-{column_idx}", "filter")

    def get_input(self):
        cols = self.skip_matrices.shape[2]
        return (self.id, f"col-{cols-1}", "filter")


class XorTree(Group):
    """A tree of XOR gates to XOR all the columns of a matrix."""

    def __init__(self, id, num_inputs=32, position=(0, 0)):
        super(XorTree, self).__init__(id, position=position)
        self.num_inputs = num_inputs
        self._add_entities()

    def _add_entities(self):
        """Add the entities required for this blueprint."""
        num_combinators = (self.num_inputs + 1) // 2
        num_layers = int(ceil(log2(num_combinators)))

        # Add the XOR gates of the tree
        for layer in range(num_layers + 1):
            step_size = 2**layer
            for i in range(0, num_combinators, step_size):
                x = i + step_size // 2
                y = -2 * layer
                self.entities.append(
                    ArithmeticCombinator(
                        id=f"ac-{layer}-{i}",
                        position=(x, y),
                        player_description="XOR the two inputs.",
                        control_behavior={
                            "arithmetic_conditions": {
                                "first_signal": "signal-each",
                                "first_signal_networks": {"red": True, "green": False},
                                "operation": "XOR",
                                "second_signal": "signal-each",
                                "second_signal_networks": {"red": False, "green": True},
                                "output_signal": "signal-each",
                            }
                        },
                    )
                )

        # Connect the XOR tree
        for layer in range(1, num_layers + 1):
            step_size = 2**layer
            for i in range(0, num_combinators, step_size):
                this = f"ac-{layer}-{i}"
                prev_left = f"ac-{layer-1}-{i}"
                prev_right = f"ac-{layer-1}-{i+step_size//2}"
                self.add_circuit_connection("red", prev_left, this, "output", "input")
                self.add_circuit_connection("green", prev_right, this, "output", "input")

    def get_input(self, input_idx: int):
        assert 0 <= input_idx < self.num_inputs
        color = ["red", "green"][input_idx % 2]
        entity = f"ac-0-{input_idx//2}"
        return color, (self.id, entity)


class BitExtractor(Group):
    """Given a signal, spread the set bits across 32 different signals."""

    def __init__(self, id, input_signal, position=(0, 0)):
        super(BitExtractor, self).__init__(id, position=position)
        self.input_signal = input_signal
        self._add_entities()

    def _add_entities(self):
        """Add the entities required for this blueprint."""
        values = [to_int32_t(1 << (31 - i)) for i in range(32)]
        lut = LUT_constant_combinator(id="lut-cc", position=(0, 0), values=values)
        applicator = ArithmeticCombinator(
            id="applicator",
            position=(0, -1),
            player_description="Extract the set bits of the input signal.",
            control_behavior={
                "arithmetic_conditions": {
                    "first_signal": self.input_signal,
                    "first_signal_networks": {"red": True, "green": False},
                    "operation": "AND",
                    "second_signal": "signal-each",
                    "second_signal_networks": {"red": False, "green": True},
                    "output_signal": "signal-each",
                }
            },
        )
        self.entities.extend([lut, applicator])

        # Connect the entities
        self.add_circuit_connection("green", "lut-cc", "applicator")

    def get_output(self):
        """Get the outputting entity of the given output index."""
        return (self.id, "applicator")

    def get_input(self):
        return (self.id, "applicator")


class LFSRAdvancer(Group):
    """Advances the state of a single LFSR using the above described matrix multiplication method."""

    def __init__(self, id, skip_matrices, matrix_idx, input_signal, position=(0, 0)):
        """Initialize the LFSR advancer with the given matrices."""
        super(LFSRAdvancer, self).__init__(id, position=position)
        self.skip_matrices = skip_matrices
        self.matrix_idx = matrix_idx
        self.input_signal = input_signal
        self._add_entities()

    def _add_entities(self):
        """Add the entities required for this blueprint."""
        num_cols = self.skip_matrices.shape[2]
        bit_extractor = BitExtractor(
            id=f"bit-extractor-{self.matrix_idx}",
            input_signal=self.input_signal,
            position=(16, -2),
        )
        mat_mul = MatrixMultiplier(
            id=f"matrix-{self.matrix_idx}",
            skip_matrices=self.skip_matrices,
            matrix_idx=self.matrix_idx,
            position=(0, -3),
        )
        xor_tree = XorTree(id=f"xor-tree-{self.matrix_idx}", num_inputs=num_cols, position=(0, -6))
        self.entities.extend([bit_extractor, mat_mul, xor_tree])

        # Connect the bit extractor to the matrix multiplier
        out_entity = bit_extractor.get_output()
        in_entity = mat_mul.get_input()
        self.add_circuit_connection("red", out_entity, in_entity, "output", "input")

        # Connect the matrix multiplier to the XOR tree
        for i in range(num_cols):
            out_entity = mat_mul.get_output(i)
            color, in_entity = xor_tree.get_input(i)
            self.add_circuit_connection(color, out_entity, in_entity, "output", "input")


class RNGAdvancer(Group):
    def __init__(self, id, skip_matrices, position=(0, 0)):
        """Initialize the RNG advancer with the given matrices."""
        super(RNGAdvancer, self).__init__(id, position=position)
        self.skip_matrices = skip_matrices
        self.input_signals = ["signal-A", "signal-B", "signal-C"]
        self._add_entities()

    def _add_entities(self):
        """Add the entities required for this blueprint."""
        num_LFSRs = self.skip_matrices.shape[1]
        for i in range(num_LFSRs):
            advancer = LFSRAdvancer(
                id=f"lfsr-{i}",
                input_signal=self.input_signals[i],
                skip_matrices=self.skip_matrices,
                matrix_idx=i,
                position=(i * (16 + 1), 0),
            )
            self.entities.append(advancer)


if __name__ == "__main__":
    # test()
    from pyperclip import copy  # type: ignore

    STEPS_TO_SIMULATE = 1000
    skip_matrices = compute_skip_matrices(STEPS_TO_SIMULATE, skip=1)

    bp = Blueprint()
    bp.label = "RNG Advancer"
    # bp.entities.append(MatrixMultiplier("matrix-0", skip_matrices, 0, position=(0, -3)))
    # bp.entities.append(XorTree("xor-tree"))
    # bp.entities.append(LFSRAdvancer("rng-advancer", skip_matrices, 0, "signal-A"))
    bp.entities.append(RNGAdvancer("rng-advancer", skip_matrices))
    copy(bp.to_string())
