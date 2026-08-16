"""Reimplementations of the RNG used in the binary and the suspected original RNG."""

from abc import abstractmethod
from typing import Any, Generic, TypeVar
from register import TRegister

rng_state_t = tuple[TRegister, TRegister, TRegister]
"""Type alias for the state of the RNGs."""

TRng = TypeVar("TRng", bound="BaseRNG")
"""Type alias for the RNGs."""


class BaseRNG(Generic[TRegister]):
    """Base class for the RNGs."""

    def __init__(self, registers: rng_state_t[TRegister]) -> None:
        """Initialize the RNG with the given state.

        Args:
            registers (rng_state_t[TRegister]): The initial state of the RNG.
        """
        self.state: rng_state_t[TRegister] = registers
        self.steps: int = 0

    @abstractmethod
    def step(self) -> None:
        """Advance the RNG state a single step."""
        raise NotImplementedError

    def get_register(self) -> TRegister:
        """Return the current state as a single register."""
        return self.state[0] ^ self.state[1] ^ self.state[2]

    def get_int(self) -> Any:
        """Return the current state as a single integer."""
        return self.get_register().get_int()

    def print_state(self) -> None:
        """Print the state of the RNG."""
        print(f"a: {self.state[0]}")
        print(f"b: {self.state[1]}")
        print(f"c: {self.state[2]}")


class ExtractedRNG(BaseRNG[TRegister]):
    """Reimplementation of the RNG calls from the disassembled binary."""

    def _single_LFSR(
        self, value: TRegister, a: int, b: int, c: int, d: int
    ) -> TRegister:
        bit_mask = (1 << d) - 1
        return (((value << a) ^ (value >> b)) & bit_mask) ^ (value >> c) ^ (value << a)

    def step(self) -> None:
        """Advance the RNG state a single step."""
        self.steps += 1
        self.state = (
            self._single_LFSR(self.state[0], 12, 6, 19, 13),
            self._single_LFSR(self.state[1], 4, 23, 25, 7),
            self._single_LFSR(self.state[2], 17, 8, 11, 21),
        )


class Taus88RNG(BaseRNG[TRegister]):
    """Reimplementation of the Taus88 RNG from boost.

    ```c++
    typedef xor_combine_engine<
        xor_combine_engine<
            linear_feedback_shift_engine<uint32_t, 32, 31, 13, 12>, 0,
            linear_feedback_shift_engine<uint32_t, 32, 29, 2, 4>, 0>, 0,
        linear_feedback_shift_engine<uint32_t, 32, 28, 3, 17>, 0> taus88;
    ```
    """

    def _word_mask(self, word_size: int) -> int:
        return (1 << word_size) - 1

    def _single_LFSR(
        self, value: TRegister, w: int, k: int, q: int, s: int
    ) -> TRegister:
        word_mask = self._word_mask(w)
        b = (((value << q) ^ value) & word_mask) >> (k - s)
        mask = (word_mask << (w - k)) & word_mask
        return ((value & mask) << s) ^ b

    def step(self) -> None:
        """Advance the RNG state a single step."""
        self.steps += 1
        self.state = (
            self._single_LFSR(self.state[0], 32, 31, 13, 12),
            self._single_LFSR(self.state[1], 32, 29, 2, 4),
            self._single_LFSR(self.state[2], 32, 28, 3, 17),
        )
