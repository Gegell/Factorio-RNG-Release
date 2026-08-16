"""Simple class to represent a bit vector register in sympy."""

from abc import abstractmethod
from typing import Any, Callable, Optional, Type, TypeVar, Union
from sympy import (
    And,
    Expr,
    Or,
    Not,
    Piecewise,
    Xor,
    S,
    symbols,
)
from sympy.logic.boolalg import BooleanFunction, Boolean
import numpy as np


TRegister = TypeVar("TRegister", bound="Register")


class Register:
    @abstractmethod
    def __init__(self, width: int, value: Optional[int] = None) -> None: ...

    @abstractmethod
    def __invert__(self: TRegister) -> TRegister: ...

    @abstractmethod
    def __and__(self: TRegister, other: Any) -> TRegister: ...

    @abstractmethod
    def __or__(self: TRegister, other: Any) -> TRegister: ...

    @abstractmethod
    def __xor__(self: TRegister, other: Any) -> TRegister: ...

    @abstractmethod
    def __rshift__(self: TRegister, amount: int) -> TRegister: ...

    @abstractmethod
    def __lshift__(self: TRegister, amount: int) -> TRegister: ...

    @abstractmethod
    def set_value(self, value: int) -> None: ...

    @abstractmethod
    def get_int(self) -> Any: ...


class IntegerRegister(Register):
    def __init__(self, width: int, value: Optional[int] = None) -> None:
        assert width > 0
        self.width = width
        self._value = 0
        if value is not None:
            self.set_value(value)

    def _word_mask(self) -> int:
        return 2**self.width - 1

    def __invert__(self) -> "IntegerRegister":
        new_reg = IntegerRegister(self.width)
        new_reg._value = ~self._value & self._word_mask()
        return new_reg

    def _apply(
        self, other: Union[int, "IntegerRegister"], op: Callable[[int, int], int]
    ) -> "IntegerRegister":
        if isinstance(other, IntegerRegister):
            assert self.width == other.width
            other = other._value
        other &= self._word_mask()
        return IntegerRegister(self.width, op(self._value, other))

    def __and__(self, other: Any) -> "IntegerRegister":
        return self._apply(other, lambda a, b: a & b)

    def __or__(self, other: Any) -> "IntegerRegister":
        return self._apply(other, lambda a, b: a | b)

    def __xor__(self, other: Any) -> "IntegerRegister":
        return self._apply(other, lambda a, b: a ^ b)

    def __rshift__(self, amount: int) -> "IntegerRegister":
        return IntegerRegister(self.width, self._value >> amount)

    def __lshift__(self, amount: int) -> "IntegerRegister":
        return IntegerRegister(self.width, self._value << amount)

    def set_value(self, value: int) -> None:
        self._value = value & self._word_mask()

    def get_int(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"IntegerRegister({self.width}, {self._value})"

    def __str__(self) -> str:
        return f"r[{self.width}] = {self._value:0{self.width}b}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntegerRegister):
            return False
        return self.width == other.width and self._value == other._value

    def __hash__(self) -> int:
        return hash((self.width, self._value))

    def __getitem__(self, key: int) -> int:
        assert 0 <= key < self.width
        return (self._value >> key) & 1

    def __setitem__(self, key: int, value: bool) -> None:
        assert 0 <= key < self.width
        new_bit = int(value)
        self._value ^= (self._value & (1 << key)) ^ (new_bit << key)


class SymbolicRegister(Register):
    def __init__(
        self, width: int, name: str = "r", value: Optional[int] = None
    ) -> None:
        assert width > 0
        self.width = width
        self._bits: list[Boolean] = list(symbols(f"{name}(:{width})"))[::-1]
        if value is not None:
            self.set_value(value)

    def __getitem__(self, key: int) -> Boolean:
        assert 0 <= key < self.width
        return self._bits[-key - 1]

    def __setitem__(self, key: int, value: Boolean) -> None:
        assert 0 <= key < self.width
        self._bits[-key - 1] = value

    def set_value(self, value: int) -> None:
        assert 0 <= value < 2**self.width
        bit_str = f"{value:0{self.width}b}"[-self.width :]
        self._bits = [S.One if bit == "1" else S.Zero for bit in bit_str]

    def __len__(self) -> int:
        return self.width

    def __iter__(self):
        return iter(self._bits)

    def __repr__(self) -> str:
        return f"Register({self.width}, {self._bits})"

    def __str__(self) -> str:
        return f"r[{self.width}] = {self._bits}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicRegister):
            return False
        return self.width == other.width and all(
            a == b for a, b in zip(self._bits, other._bits)
        )

    def __hash__(self) -> int:
        return hash((self.width, tuple(self._bits)))

    def _apply_to_all(
        self, other: Any, op: Type[BooleanFunction]
    ) -> "SymbolicRegister":
        if isinstance(other, SymbolicRegister):
            assert self.width == other.width
            bits = other._bits
        elif isinstance(other, int):
            bit_str = f"{other:0{self.width}b}"[: self.width]
            bits = [S.One if bit == "1" else S.Zero for bit in bit_str]
        else:
            return NotImplemented

        new_reg = SymbolicRegister(self.width)
        new_reg._bits = [op(a, b) for a, b in zip(self._bits, bits)]
        return new_reg

    def __invert__(self) -> "SymbolicRegister":
        new_reg = SymbolicRegister(self.width)
        new_reg._bits = [Not(a) for a in self._bits]
        return new_reg

    def __and__(self, other: Any) -> "SymbolicRegister":
        return self._apply_to_all(other, And)

    def __or__(self, other: Any) -> "SymbolicRegister":
        return self._apply_to_all(other, Or)

    def __xor__(self, other: Any) -> "SymbolicRegister":
        return self._apply_to_all(other, Xor)

    def __rshift__(self, amount: int) -> "SymbolicRegister":
        assert isinstance(amount, int)
        new_reg = SymbolicRegister(self.width)
        new_reg._bits = [S.Zero] * amount + self._bits[:-amount]
        return new_reg

    def __lshift__(self, amount: int) -> "SymbolicRegister":
        assert isinstance(amount, int)
        new_reg = SymbolicRegister(self.width)
        new_reg._bits = self._bits[amount:] + [S.Zero] * amount
        return new_reg

    def get_int(self) -> Expr:
        """Returns the current state as a single integer."""
        equations = [
            Piecewise((2**i, bit), (0, True)) for i, bit in enumerate(self._bits[::-1])
        ]
        return sum(equations, S.Zero)


class ProbabilisticRegister(Register):
    def __init__(self, width: int, value: Optional[int] = None) -> None:
        assert width > 0
        self.width = width
        self._bits = np.ones(width) / 2
        if value is not None:
            self.set_value(value)

    def __getitem__(self, key: int) -> float:
        assert 0 <= key < self.width
        return self._bits[-key - 1]

    def __setitem__(self, key: int, value: float) -> None:
        assert 0 <= key < self.width
        self._bits[-key - 1] = value

    def set_value(self, value: int) -> None:
        assert 0 <= value < 2**self.width
        bit_str = f"{value:0{self.width}b}"[-self.width :]
        self._bits = np.array([1.0 if bit == "1" else 0.0 for bit in bit_str])

    def __len__(self) -> int:
        return self.width

    def __iter__(self):
        return iter(self._bits)

    def __repr__(self) -> str:
        return f"ProbabilisticRegister({self.width}, {self._bits})"

    def __str__(self) -> str:
        return f"r[{self.width}] = {self._bits}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProbabilisticRegister):
            return False
        return self.width == other.width and all(
            a == b for a, b in zip(self._bits, other._bits)
        )

    def __hash__(self) -> int:
        return hash((self.width, tuple(self._bits)))

    def _cast_to_prob_reg(self, other: Any) -> "ProbabilisticRegister":
        if isinstance(other, ProbabilisticRegister):
            assert self.width == other.width
            return other
        elif isinstance(other, int):
            new_reg = ProbabilisticRegister(self.width)
            new_reg.set_value(other)
            return new_reg
        else:
            return NotImplemented

    def __and__(self, other: Any) -> "ProbabilisticRegister":
        other_reg = self._cast_to_prob_reg(other)
        new_reg = ProbabilisticRegister(self.width)
        new_reg._bits = self._bits * other_reg._bits
        return new_reg

    def __or__(self, other: Any) -> "ProbabilisticRegister":
        other_reg = self._cast_to_prob_reg(other)
        new_reg = ProbabilisticRegister(self.width)
        new_reg._bits = 1.0 - (1.0 - self._bits) * (1.0 - other_reg._bits)
        return new_reg

    def __xor__(self, other: Any) -> "ProbabilisticRegister":
        other_reg = self._cast_to_prob_reg(other)
        new_reg = ProbabilisticRegister(self.width)
        new_reg._bits = (
            self._bits + other_reg._bits - 2.0 * self._bits * other_reg._bits
        )
        return new_reg

    def __invert__(self) -> "ProbabilisticRegister":
        new_reg = ProbabilisticRegister(self.width)
        new_reg._bits = 1.0 - self._bits
        return new_reg

    def __rshift__(self, amount: int) -> "ProbabilisticRegister":
        assert isinstance(amount, int)
        new_reg = ProbabilisticRegister(self.width)
        new_reg._bits = np.concatenate((np.zeros(amount), self._bits[:-amount]))
        return new_reg

    def __lshift__(self, amount: int) -> "ProbabilisticRegister":
        assert isinstance(amount, int)
        new_reg = ProbabilisticRegister(self.width)
        new_reg._bits = np.concatenate((self._bits[amount:], np.zeros(amount)))
        return new_reg

    def get_int(self) -> int:
        """Returns the most likely value of the register as a single integer."""
        return sum(2**i for i, bit in enumerate(self._bits) if bit > 0.5)

    def entropy(self) -> float:
        """Returns the entropy of the register."""
        return -sum(
            (
                p * np.log2(p) + (1 - p) * np.log2(1 - p)
                for p in self._bits
                if 0.0 < p < 1.0
            ),
            0.0,
        )


def main() -> None:
    regA = SymbolicRegister(32, "A")
    regB = SymbolicRegister(32, "B")

    print(regA)
    print(regB << 2)

    smallReg = SymbolicRegister(4, "s")
    # smallReg.set_value(0b1010)
    print(smallReg._bits)
    print(smallReg.get_int())

    smallProbReg = ProbabilisticRegister(4)
    print(smallProbReg.entropy())
    smallProbReg._bits[-1] = 0.4
    print(smallProbReg.entropy())


if __name__ == "__main__":
    main()
