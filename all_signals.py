import enum
from functools import cache, reduce
from operator import or_
from itertools import product
from typing import Union
from pyperclip import copy  # type: ignore

from draftsman.data import signals, items  # type: ignore
from draftsman.blueprintable import Blueprint  # type: ignore
from draftsman.entity import ConstantCombinator  # type: ignore
from draftsman.signatures import SignalFilter, Sections, Section, SignalID  # type: ignore

ordered_types = [
    "item",
    "fluid",
    "recipe",
    "entity",
    "space-location",
    "asteroid-chunk",
    "quality",
    "virtual",
]

blacklist = {
    "copy-paste-tool-recycling",
    "cut-paste-tool-recycling",
}


def get_order(sig: str, typ: str) -> str:
    """get the order"""
    if typ == "item" and (item := items.raw.get(sig)):
        return item.get("order", "")
    return signals.raw[sig].get("order", "")


def get_type(signal: str) -> str:
    """Get the type of a signal."""
    types = signals.type_of.get(signal, [])
    return min(types, key=ordered_types.index) if types else "unknown"


def is_recycling(signal: str) -> bool:
    """Check if a signal is a recycling signal."""
    return signals.raw[signal].get("category", "") == "recycling"


class FilterFlags(enum.IntFlag):
    """Flags for filtering signals."""

    NONE = 0
    HIDDEN = enum.auto()  # Signals hidden in the game
    RECYCLING = enum.auto()  # Recycling recipe signals
    CORPSES = enum.auto()  # Corpse and rail-remnants
    DUPLICATES = enum.auto()  # Join separate item, entity, recipe signals into one

    VIRTUAL_NUMBER = enum.auto()  # 0-9
    VIRTUAL_LETTER = enum.auto()  # A-Z
    VIRTUAL_OTHER = enum.auto()  # colors, punctuation, pictograms, etc.
    VIRTUAL_PURE = enum.auto()  # each, any and every virtual signals
    PARAMETERS = enum.auto()  # parameter signals

    VIRTUAL = VIRTUAL_LETTER | VIRTUAL_NUMBER | VIRTUAL_OTHER
    SPECIAL = PARAMETERS | VIRTUAL_PURE

    ALL = HIDDEN | RECYCLING | CORPSES | DUPLICATES | VIRTUAL | SPECIAL

    @staticmethod
    def from_verbosity(verbosity: int) -> "FilterFlags":
        """Get the filter flags for a verbosity level. 0 = only common, 6 = all of the signals (cursed)."""
        flags = [
            FilterFlags.VIRTUAL,
            FilterFlags.RECYCLING,
            FilterFlags.HIDDEN,
            FilterFlags.CORPSES,
            FilterFlags.DUPLICATES,
            FilterFlags.SPECIAL,
        ]
        assert 0 <= verbosity <= len(flags), f"verbosity must be between 0 and {len(flags)}"
        return reduce(or_, flags[:verbosity], FilterFlags.NONE)


@cache
def get_signal_list(
    include: FilterFlags = FilterFlags.NONE,
) -> list[SignalID]:
    skip_flags = FilterFlags.ALL & ~include
    signal_list: list[SignalID] = []
    # The original raw dict is ordered as in the game, just iterate over that instead of sorting.
    for sig in signals.raw:
        signal = signals.raw[sig]
        types = signals.type_of.get(sig, ["unknown"])
        item = items.raw.get(sig, signal)
        recycle = signal.get("category", "") == "recycling"

        if sig in blacklist:
            continue
        matches = FilterFlags.NONE
        if (signal.get("hidden", False) and item.get("hidden", False)) and not recycle:
            matches |= FilterFlags.HIDDEN
        if recycle:
            matches |= FilterFlags.RECYCLING
        if "parameter" in sig:
            matches |= FilterFlags.PARAMETERS
        if signal.get("type", "") in {"corpse", "rail-remnants"}:
            matches |= FilterFlags.CORPSES
        if "signal-0" <= sig <= "signal-9":
            matches |= FilterFlags.VIRTUAL_NUMBER
        if "signal-A" <= sig <= "signal-Z":
            matches |= FilterFlags.VIRTUAL_LETTER
        if sig in signals.pure_virtual:
            matches |= FilterFlags.VIRTUAL_PURE
        if "virtual" in types and not (matches & FilterFlags.VIRTUAL) and (sig != "signal-any-quality"):
            matches |= FilterFlags.VIRTUAL_OTHER

        # If any of the skip flags are matched, skip this signal
        if matches & skip_flags:
            continue

        if not (skip_flags & FilterFlags.DUPLICATES) or "barrel" in sig:
            signal_list.extend([SignalID(name=sig, type=typ, quality="normal") for typ in types])
        else:
            signal_list.append(SignalID(name=sig, type=get_type(sig), quality="normal"))

    # Sort the recycling signals to the end
    signal_list.sort(key=lambda s: is_recycling(s.name))

    return signal_list


SIGNAL_LIST = [signal.name for signal in get_signal_list()]


# Signal type "name [type [quality]]"
signalT = Union[str, tuple[str, str], tuple[str, str, str], SignalID]


def to_signalID(signal: signalT) -> SignalID:
    """Convert a signal to a SignalID."""
    if isinstance(signal, SignalID):
        return signal
    if isinstance(signal, str):
        return SignalID(name=signal, type=get_type(signal), quality="normal")
    if len(signal) == 2:
        return SignalID(name=signal[0], type=signal[1], quality="normal")
    return SignalID(name=signal[0], type=signal[1], quality=signal[2])  # type: ignore # mypy sucks at counting???


def normalize_signal(signal: signalT) -> tuple[str, str, str]:
    """Normalize a signal to a tuple of (name, type, quality)."""
    sigID = to_signalID(signal)
    return (sigID.name, sigID.type, sigID.quality)


SignalValueListT = list[int] | dict[str, int] | dict[tuple[str, str], int] | dict[tuple[str, str, str], int]


def LUT_constant_combinator(
    id, position, values: SignalValueListT, max_verbosity=1, allow_quality=False, **kwargs
) -> ConstantCombinator:
    """Create a constant combinator with the given values."""
    assert 0 <= max_verbosity <= 6, "max_verbosity must be between 0 and 6"
    # Note: "any" does not work together with other qualities
    qualities = ["normal", "uncommon", "rare", "epic", "legendary"] + ["quality-unknown"]
    if not isinstance(values, dict):
        signal_list: list[SignalID] = []
        for i in range(max_verbosity):
            signal_list = get_signal_list(include=FilterFlags.from_verbosity(i))
            if len(values) <= len(signal_list):
                break
            if allow_quality and len(values) <= len(signal_list) * len(qualities):
                signal_list = [
                    SignalID(name=sig.name, type=sig.type, quality=q) for sig, q in product(signal_list, qualities)
                ]
                break
        else:
            hint = " Consider using higher verbosity." if max_verbosity < 6 else ""
            hint += f" Consider using allow_quality=True. ({len(qualities)}x)" if not allow_quality else ""
            num_available = len(signal_list) * (len(qualities) if allow_quality else 1)
            raise ValueError(f"Too many values ({len(values)}) for the signal list ({num_available}).{hint}")
        values = {normalize_signal(signal): value for signal, value in zip(signal_list, values)}
    else:
        values = {normalize_signal(signal): value for signal, value in values.items()}

    max_section_size = 1000
    filters = [
        SignalFilter(
            index=(i % max_section_size) + 1,
            name=signal,
            quality=quality,
            comparator="=",
            count=value,
            type=typ,
        )
        for i, ((signal, typ, quality), value) in enumerate(values.items())
    ]

    slices = [slice(i, i + max_section_size) for i in range(0, len(filters), max_section_size)]
    section_list = [Section(index=i + 1, filters=filters[sl]) for i, sl in enumerate(slices)]

    return ConstantCombinator(
        id=id,
        position=position,
        control_behavior={"sections": Sections(sections=section_list)},
        **kwargs,
    )


if __name__ == "__main__":
    # Create a constant combinator with all signals
    # and their corresponding values from 1 to len(signal_list)

    signal_list = get_signal_list()
    all_qualities = [
        SignalID(name=sig.name, type=sig.type, quality=q) for sig, q in product(signal_list, signals.quality)
    ]
    values = {k: i for i, k in enumerate(signal_list, 1)}

    bp = Blueprint()
    bp.entities.append(
        LUT_constant_combinator(
            id="constant_combinator",
            position=(0, 0),
            values=list(i for i in range(0, 12990)),
            max_verbosity=6,
            allow_quality=True,
        )
    )

    copy(bp.to_string())
