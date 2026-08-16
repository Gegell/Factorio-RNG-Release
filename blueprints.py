from typing import Optional
from draftsman.blueprintable import Blueprint  # type: ignore
from draftsman.constants import Direction  # type: ignore
from draftsman.entity import ArithmeticCombinator, ConstantCombinator, Lamp  # type: ignore
from draftsman.data import signals  # type: ignore
import pyperclip  # type: ignore

from all_signals import LUT_constant_combinator, get_signal_list


def to_uint32_t(n):
    """Convert a number to an unsigned 32-bit integer."""
    return n & 0xFFFFFFFF


def to_int32_t(n):
    """Convert a number to a signed 32-bit integer."""
    n = n & 0xFFFFFFFF
    return (n ^ 0x80000000) - 0x80000000


def single_bit_mask_ac(
    bp: Optional[Blueprint], run_len=16, gap=2, direction=Direction.NORTH
):
    """A row of arithmetic combinators that will extract a single bit from each applied signal."""
    if bp is None:
        bp = Blueprint()
        bp.label = "Single Bit Mask"
        bp.description = single_bit_mask_ac.__doc__
    for i in range(32):
        bp.entities.append(
            ArithmeticCombinator(
                position=rotate_position(
                    (run_gap_position(i, run_len, gap), 0), direction
                ),
                direction=direction,
                control_behavior={
                    "arithmetic_conditions": {
                        "first_signal": {"type": "virtual", "name": "signal-each"},
                        "operation": "AND",
                        "second_constant": to_int32_t(1 << i),
                        "output_signal": {"type": "virtual", "name": "signal-each"},
                    }
                },
            )
        )
    return bp


def run_gap_position(index, run_len, gap):
    """Get the position of an entity in a run with gaps."""
    group = index // run_len
    return (index % run_len) + group * (gap + run_len)


def rotate_position(position, direction):
    """Rotate a position around (0,0) to a new position."""
    x, y = position
    if direction == Direction.NORTH:
        return (x, y)
    elif direction == Direction.EAST:
        return (y, -x)
    elif direction == Direction.SOUTH:
        return (-x, -y)
    elif direction == Direction.WEST:
        return (-y, x)
    else:
        raise ValueError(f"Invalid direction {direction}")


def single_bit_mask_cc(run_len=4, gap=2, signal="signal-D"):
    """A row of constant combinators that will output a single bit."""
    bp = Blueprint()
    bp.label = "Single Bit Mask"
    bp.description = single_bit_mask_cc.__doc__
    for i in range(32):
        bp.entities.append(
            ConstantCombinator(
                position=rotate_position(
                    (run_gap_position(i, run_len, gap), 0), Direction.NORTH
                ),
                direction=Direction.NORTH,
                control_behavior={
                    "filters": [
                        {
                            "signal": {"type": "virtual", "name": signal},
                            "count": to_int32_t(1 << i),
                            "index": 1,
                        }
                    ]
                },
            )
        )
    return bp


def vectorized_LFSR_block_acs(a, b, c, d):
    """Collection of combinators to build LFSR block with parameters a,b,c,d. See ExtractedRNG."""
    bp = Blueprint()
    bp.label = "LFSR Block"
    bp.description = f"LFSR Block with parameters a={a}, b={b}, c={c}, d={d}."

    total_bits = 32  # In taus88 this is equivalent to parameter w
    reg_width = a + c  # In taus88 this is equivalent to parameter q
    alternating_mask = 0x5555_5555
    bit_mask_a = ((1 << (reg_width - a)) - 1) << (total_bits - reg_width)
    bit_mask_b = (1 << (total_bits - (c - b))) - 1
    combinators = [
        ((0, 0), a, ">>"),
        ((0, 2), b, "<<"),
        ((0, 4), c, "<<"),
        ((2, 0), alternating_mask & bit_mask_a, "AND"),
        ((2, 1), (alternating_mask << 1) & bit_mask_a, "AND"),
        ((2, 2), alternating_mask & bit_mask_b, "AND"),
        ((2, 3), (alternating_mask << 1) & bit_mask_b, "AND"),
        ((2, 4), alternating_mask, "AND"),
        ((2, 5), alternating_mask << 1, "AND"),
    ]

    for pos, const, op in combinators:
        bp.entities.append(
            ArithmeticCombinator(
                position=pos,
                direction=Direction.EAST,
                control_behavior={
                    "arithmetic_conditions": {
                        "first_signal": {"type": "virtual", "name": "signal-each"},
                        "operation": op,
                        "second_constant": to_int32_t(const),
                        "output_signal": {"type": "virtual", "name": "signal-each"},
                    }
                },
            )
        )

    return bp


def remove_all_circuit_connections(bp: Blueprint) -> Blueprint:
    """Remove all circuit connections from a blueprint."""
    bp.remove_circuit_connections()
    return bp


def get_all_signals() -> list[str]:
    """Get a list of all signal names."""
    return (
        signals.item
        + signals.fluid
        + [s for s in signals.virtual if s not in signals.pure_virtual]
    )


def all_signals_cc(group_count=1) -> Blueprint:
    """Row of all signals in constant combinators."""
    bp = Blueprint()
    bp.label = "All Signals"
    bp.description = all_signals_cc.__doc__
    all_signals = get_all_signals()
    signal_groups = [all_signals[o::group_count] for o in range(group_count)]
    for y, group in enumerate(signal_groups):
        for x, cc_contents in enumerate(
            group[i : i + 20] for i in range(0, len(group), 20)
        ):
            bp.entities.append(
                ConstantCombinator(
                    position=(-x, y),
                    direction=Direction.EAST,
                    control_behavior={
                        "filters": [
                            {
                                "signal": signal,
                                "count": i + 1 + (x * 20),
                                "index": i + 1,
                            }
                            for i, signal in enumerate(cc_contents)
                        ]
                    },
                )
            )
            if x > 0:
                bp.add_circuit_connection("green", -1, -2)
    return bp


def all_pows_of_2(start_signal="signal-0") -> Blueprint:
    """Row of all powers of 2 in constant combinators."""
    bp = Blueprint()
    bp.label = "All Powers of 2"
    bp.description = all_pows_of_2.__doc__
    all_signals = get_all_signals()
    start_idx = all_signals.index(start_signal)
    signals = all_signals[start_idx : start_idx + 32]
    assert len(signals) == 32
    for i, signal_group in enumerate(
        signals[i : i + 20] for i in range(0, len(signals), 20)
    ):
        bp.entities.append(
            ConstantCombinator(
                position=(-i // 20, 0),
                direction=Direction.EAST,
                control_behavior={
                    "filters": [
                        {
                            "signal": signal,
                            "count": to_int32_t(2 ** (i * 20 + j)),
                            "index": j + 1,
                        }
                        for j, signal in enumerate(signal_group)
                    ]
                },
            )
        )
        if i > 0:
            bp.add_circuit_connection("green", -1, -2)
    return bp


def recenter_coords(bp: Blueprint) -> Blueprint:
    """Recenter the coordinates of a blueprint such that the AABB is approximately centered on (0,0)."""
    bbox = bp.get_world_bounding_box()
    c_x = int((bbox.bot_right[0] + bbox.top_left[0]) / 2)
    c_y = int((bbox.bot_right[1] + bbox.top_left[1]) / 2)
    bp.translate(-c_x, -c_y)
    return bp


def lamp_matrix(signal_count: int = 96) -> Blueprint:
    """A matrix of lamps that displays each signals binary representation."""
    signals = get_all_signals()[:signal_count]
    bp = Blueprint()
    bp.label = "Lamp Matrix"
    bp.description = lamp_matrix.__doc__

    # Initial bit extraction combinators
    single_bit_mask_ac(bp, run_len=32, gap=0, direction=Direction.SOUTH)

    for i, signal in enumerate(signals):
        for bit in range(32):
            bp.entities.append(
                Lamp(
                    position=(
                        -run_gap_position(bit, run_len=32 * 3, gap=0),
                        -2 - run_gap_position(i, run_len=32 * 3, gap=0),
                    ),
                    control_behavior={
                        "circuit_enabled": True,
                        "circuit_condition": {
                            "first_signal": signal,
                            "constant": 0,
                            "comparator": "!=",
                        },
                    },
                )
            )
            if i > 0:
                bp.add_circuit_connection("green", -1, -33)
            else:
                bp.add_circuit_connection("green", -1, -33, side_2="output")

    return recenter_coords(bp)


if __name__ == "__main__":
    # blueprint_str = pyperclip.paste()
    # with open("small_bp.txt", "r") as f:
    #     blueprint_str = f.read()
    # bp = Blueprint(blueprint_str)
    # bp = remove_all_circuit_connections(bp)

    # bp = single_bit_mask_ac()
    # bp = single_bit_mask_cc()
    # bp = vectorized_LFSR_block_acs(12, 6, 19, 13)
    # bp = vectorized_LFSR_block_acs(4, 23, 25, 7)
    # bp = vectorized_LFSR_block_acs(17, 8, 11, 21)
    # bp = all_signals_cc()
    # bp = all_pows_of_2()
    # bp = lamp_matrix(signal_count=96)
    bp = Blueprint()
    bp.entities.append(
        LUT_constant_combinator(
            id="cc", position=(0, 0), values=[10 * i + 100 for i in range(500)]
        )
    )

    pyperclip.copy(bp.to_string())
