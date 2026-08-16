"""Miscellaneous experiments and utilities for analyzing Factorio's RNG.

This module builds symbolic generation matrices for the extracted RNG, solves
for internal state from observed outputs, renders GF(2) matrix diagnostics, and
simulates recipe-result rolls to estimate the information exposed by crafting.
Its main entry point runs one state-recovery I did previously and predicts some outputs.
"""

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import floor, log2
import sys
from typing import Iterable, Optional, Type
from sympy import GF, Matrix, Symbol, matrix2numpy
from sympy.polys.domainmatrix import DomainMatrix
from sympy.logic.boolalg import Boolean
from register import SymbolicRegister
from PIL import Image
from pathlib import Path

from rngs import BaseRNG, ExtractedRNG, TRng, Taus88RNG, rng_state_t


def build_generation_matrix(taps: Iterable[int] = (0, 1, 2), top_k_bits=None) -> Matrix:
    """Build the matrix that describes the RNG generation process."""
    taps = list(taps)
    start_rng = ExtractedRNG(
        registers=(
            SymbolicRegister(32, "a_"),
            SymbolicRegister(32, "b_"),
            SymbolicRegister(32, "c_"),
        )
    )
    symbol_pos = {
        symbol: (reg_idx * 32 + bit_idx)
        for reg_idx, reg in enumerate(start_rng.state)
        for bit_idx, symbol in enumerate(reg._bits)
    }
    if top_k_bits is None:
        top_k_bits = 32

    equations: list[Boolean] = []
    for tap_count in taps:
        rng = deepcopy(start_rng)
        for _ in range(tap_count):
            rng.step()
        equations.extend((rng.get_register()._bits[:top_k_bits]))

    matrix = [[0 for _ in range(96)] for _ in range(len(equations))]
    for y, equation in enumerate(equations):
        for symbol in equation.atoms(Symbol):
            matrix[y][symbol_pos[symbol]] = 1
    return Matrix(matrix)


def render_binary_matrix_as_image(mat: Matrix, name: str) -> None:
    """Render the given binary matrix as a black & white image."""
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(matrix2numpy(mat * 255, int)).convert("L")
    img.save(name)


def has_solution(mat: DomainMatrix, output: DomainMatrix) -> bool:
    """Return true if the given equation system has a solution."""
    return mat.rank() == mat.hstack(output).rank()


def analyze_equation_system(mat: DomainMatrix, output: DomainMatrix) -> None:
    """Print some information about the given equation system & generate some images for debugging."""
    print("Matrix shape: ", mat.shape)
    print("Output shape: ", output.shape)
    print("Matrix domain:", mat.domain)
    print("Output domain:", output.domain)
    print("Matrix rank:  ", mat.rank())

    render_binary_matrix_as_image(mat.to_dense().to_Matrix(), "analysis/equations.png")

    print("Null space:")
    nullspace = mat.nullspace()
    print(nullspace.shape)
    print(nullspace)
    if min(nullspace.shape) > 0:
        render_binary_matrix_as_image(
            nullspace.to_dense().to_Matrix(), "analysis/nullspace.png"
        )

    print("Row space:")
    rowspace = mat.rowspace()
    print(rowspace.shape)
    # print(rowspace)
    render_binary_matrix_as_image(
        rowspace.to_dense().to_Matrix(), "analysis/rowspace.png"
    )

    print("Reduced row echelon form:")
    rref_matrix, rref_pivots = mat.rref()
    print(rref_matrix.shape)
    print(rref_pivots)
    print(set(range(min(rref_matrix.shape))) - set(rref_pivots))
    # print(rref_matrix)
    render_binary_matrix_as_image(
        rref_matrix.to_dense().to_Matrix(), "analysis/rref.png"
    )

    print("LU-decomposition:")
    lower, upper, swaps = mat.lu()
    print(lower.shape)
    print(upper.shape)
    # print(swaps)
    render_binary_matrix_as_image(lower.to_dense().to_Matrix(), "analysis/lower.png")
    render_binary_matrix_as_image(upper.to_dense().to_Matrix(), "analysis/upper.png")

    # Pseudo-inverse
    print("Pseudo-inverse:")
    identity = mat.eye(mat.shape[0], mat.domain)
    augmented_mat = mat.hstack(identity)
    augmented_rref_mat, _ = augmented_mat.rref()
    pseudo_inverse = augmented_rref_mat[:, mat.shape[0] :]
    print(pseudo_inverse.shape)
    render_binary_matrix_as_image(
        augmented_mat.to_dense().to_Matrix(), "analysis/augmented.png"
    )
    render_binary_matrix_as_image(
        augmented_rref_mat.to_dense().to_Matrix(), "analysis/augmented_rref.png"
    )
    render_binary_matrix_as_image(
        pseudo_inverse.to_dense().to_Matrix(), "analysis/pseudo_inverse.png"
    )

    # Has solution?
    print("Has at least 1 solution: ", has_solution(mat, output))


def get_state(observed_ints: list[int]) -> rng_state_t:
    """Return the RNG state that produces the 3 given outputs."""
    mat = build_generation_matrix()
    domain_mat = DomainMatrix.from_Matrix(mat)
    domain_mat = domain_mat.convert_to(GF(2))

    # Join the 3 output bits into a single 96 bit vector
    bits_str = "".join(f"{x:032b}" for x in observed_ints[:3])
    output = DomainMatrix.from_list([[int(x)] for x in bits_str], GF(2))

    # analyze_equation_system(domain_mat, output)

    augmented = domain_mat.hstack(output)
    rref_mat, pivots = augmented.rref()

    rref_mat, solution = rref_mat[:, :96], rref_mat[:, 96:]
    print("Solution:")
    print(solution.to_dense().to_Matrix())

    # Convert the solution into a state
    state: rng_state_t = (
        SymbolicRegister(32, "a"),
        SymbolicRegister(32, "b"),
        SymbolicRegister(32, "c"),
    )
    for eq_idx, bit_idx in enumerate(pivots):
        bit = 31 - (bit_idx % 32)
        register = state[bit_idx // 32]
        sol = solution.to_Matrix()[eq_idx, 0]
        register[bit] = sol
    return state


def compare_sequence(rng_data: list[int], rng: TRng) -> bool:
    """Compare the given RNG data with the output of the given RNG."""
    rng_clone = deepcopy(rng)
    for i, x in enumerate(rng_data):
        val = int(rng_clone.get_int())
        if x != val:
            print(f"Failed at {i}: Should be {x} but is {val}")
            return False
        rng_clone.step()
    return True


def main(given_rng_data: Optional[list[int]] = None) -> None:
    """Reverse the Factorio RNG, given some RNG data. Main entry point."""
    render_binary_matrix_as_image(
        build_generation_matrix(taps=(1,)), "analysis/skip_1.png"
    )

    rng = ExtractedRNG(
        registers=(
            SymbolicRegister(32, "a"),
            SymbolicRegister(32, "b"),
            SymbolicRegister(32, "c"),
        )
    )

    print("Simulate a single step of the RNG:")
    print("Initial state:")
    rng.print_state()
    all_symbols = rng.get_int().free_symbols
    print("After 1 step:")
    rng.step()
    rng.print_state()
    dropped_symbols = all_symbols - rng.get_int().free_symbols
    print("Dropped variables: " + ", ".join(sorted(map(str, dropped_symbols))))

    if not given_rng_data:
        print()
        print("Generate some synthetic data:")
        rng.state[0].set_value(0x0123_4567)
        rng.state[1].set_value(0x89AB_CDEF)
        rng.state[2].set_value(0xFEDC_BA98)
        rng_data = []
        for _ in range(10):
            rng_data.append(int(rng.get_int()))
            print(
                "  {:08x} {:08x} {:08x}".format(
                    *(int(reg.get_int()) for reg in rng.state)
                )
            )
            rng.step()
    else:
        rng_data = given_rng_data

    print()
    print("RNG data:")
    for x in rng_data:
        print(f"  {x:>08x}")

    # Extract the RNG state from the data
    print()
    rng.state = get_state(rng_data)

    # Find the initial state by brute forcing all possible remaining assignments
    print()
    print(f"Goal: {rng_data[0]:>8x}")
    print("Final state:")
    rng.print_state()
    int_eq = rng.get_int()
    free_symbols = int_eq.free_symbols
    print("Free symbols:", ", ".join(sorted(map(str, free_symbols))))
    print("Integer equation:", int_eq)

    print()
    print("Trying all possible assignments:")
    for assignment in product([0, 1], repeat=len(free_symbols)):
        res = int_eq.subs(zip(free_symbols, assignment))
        matches = res == rng_data[0]
        if not matches:
            continue
            pass
        register_values = [
            int(reg.get_int().subs(zip(free_symbols, assignment))) for reg in rng.state
        ]
        print(
            "Trying {} = {:>08x} - {} - Internal: {:08x} {:08x} {:08x}".format(
                assignment, int(res), matches, *register_values
            )
        )
        fully_specified_rng = ExtractedRNG(
            registers=(
                SymbolicRegister(32, value=register_values[0]),
                SymbolicRegister(32, value=register_values[1]),
                SymbolicRegister(32, value=register_values[2]),
            )
        )
        matches_all = compare_sequence(rng_data, fully_specified_rng)
        if matches_all:
            print("Found a match!")
            break
    else:
        print("No match found!")
        exit(1)

    print()
    print("Simulating the next 30 steps:")
    rng = deepcopy(fully_specified_rng)
    for it in range(-len(rng_data) + 1, 31):
        value = int(rng.get_int())
        print(
            "{:>4d}  {:08x} ^ {:08x} ^ {:08x} = {value:08x}  ({value:>10d})".format(
                it, *(int(reg.get_int()) for reg in rng.state), value=value
            )
        )
        rng.step()
        if it == 0:
            print("  {:~^59s}".format(" Prediction starts here "))

    print()

    repair_pack_recycle = Recipe(
        "repair-pack-recycling",
        [
            RecipeResult("electronic-circuit", 2 / 4),
            RecipeResult("iron-gear-wheel", 2 / 4),
        ],
    )
    simulate_recipe_results(
        repair_pack_recycle, deepcopy(fully_specified_rng), craft_count=50
    )
    simulate_until(deepcopy(fully_specified_rng), 0, 200)

    exit()

    print()
    print("Predicting the outputs of multiple train station recycle operations:")
    ts_recycle = Recipe(
        "train-stop-recycling",
        [
            RecipeResult("electronic-circuit", 5 / 4),
            RecipeResult("iron-plate", 6 / 4),
            RecipeResult("iron-stick", 6 / 4),
            RecipeResult("steel-plate", 3 / 4),
        ],
    )
    simulate_recipe_results(ts_recycle, deepcopy(fully_specified_rng), craft_count=100)

    # simulate_until(deepcopy(fully_specified_rng), 0xD88CBB9A)

    print()
    print(
        "Predicting the number of crafting cycles to gain enough knowledge to rebuild rng state:"
    )
    rng = deepcopy(fully_specified_rng)
    uranium_recipe = Recipe(
        "uranium-processing",
        [
            RecipeResult("uranium-235", 0.007),
            RecipeResult("uranium-238", 0.993),
        ],
    )
    simulate_recipe_results(uranium_recipe, rng, bit_count=96, suppress_no_data=True)


def sim_solve(mat: Matrix) -> Matrix:
    """Simulate the solving of the given matrix as is done by the solver I implemented using circuit networks."""
    mat = mat.copy()
    used_rows = []
    for column in range(mat.shape[1]):
        # For the pivot find the first row not previously used with a 1 in the column
        for pivot_row in range(mat.shape[0]):
            if pivot_row in used_rows:
                continue
            if mat[pivot_row, column] != 0:
                used_rows.append(pivot_row)
                # Eliminate all other 1s in the column
                for other_row in range(mat.shape[0]):
                    if other_row == pivot_row:
                        continue
                    if mat[other_row, column] != 0:
                        # Add the pivot row to the other row
                        mat[other_row, :] = (mat[other_row, :] + mat[pivot_row, :]) % 2
                break
    return mat


def simulate_until(rng: BaseRNG, state: int, _max_steps: int = int(1e6)) -> None:
    """Simulate the given RNG until it reaches the target state."""
    for step in range(_max_steps):
        print(
            "{:>5d}  {:08x} ^ {:08x} ^ {:08x} = {:08x}".format(
                step, *(int(reg.get_int()) for reg in rng.state), int(rng.get_int())
            )
        )
        if int(rng.get_int()) == state:
            print("Found the specified state!")
            return
        rng.step()
    else:
        print(
            f"Could not find the specified state without exceeding the step limit {_max_steps}."
        )


@dataclass
class RecipeResult:
    name: str
    result_count: float
    output_prob: float = 1.0


@dataclass
class Recipe:
    """A simple representation of the in game recipes."""

    name: str
    results: list[RecipeResult]


def simulate_recipe_results(
    recipe: Recipe,
    rng: BaseRNG,
    craft_count=None,
    bit_count=None,
    suppress_no_data=False,
    _max_crafts=int(1e6),
):
    """Simulate the results of crafting the given recipe."""
    if craft_count is None and bit_count is None:
        craft_count = 10

    start_step = rng.steps
    total_bits_observed = 0
    craft_attempts = 0

    def rng_print(message):
        num_steps = rng.steps - start_step
        last_state = int(rng.get_int())
        msb_nibble = (last_state >> 28) & 0xF
        print(f"{num_steps: >4d}  {last_state:08x} {msb_nibble:04b}...  {message}")

    for _ in range(_max_crafts):
        if craft_count is not None and craft_attempts >= craft_count:
            break
        if bit_count is not None and total_bits_observed >= bit_count:
            break

        # Simulate the crafting process
        craft_attempts += 1
        for result in recipe.results:
            if result.output_prob < 1:
                sample = rng.get_int()
                is_generated = sample / 2**32 < result.output_prob
                if not is_generated:
                    rng_print(f"{result.name:20s} skipped (failed output prob check)")
                    rng.step()
                    continue
                else:
                    rng_print(f"{result.name:20s} prob check success")
                rng.step()
            fractional = result.result_count - int(result.result_count)
            if fractional != 0:
                sample = rng.get_int()
                roll = sample / 2**32
                extra_output = 1 if roll < fractional else 0
                produced = int(result.result_count) + extra_output
                if fractional < 0.5:
                    observed_bit_type = 0
                    observed_bits = floor(-log2(fractional)) if extra_output else 0
                elif fractional > 0.5:
                    observed_bit_type = 1
                    observed_bits = (
                        floor(-log2(1 - fractional)) if not extra_output else 0
                    )
                else:
                    observed_bit_type = 1 - extra_output
                    observed_bits = 1
                total_bits_observed += observed_bits
                if not (suppress_no_data and observed_bits == 0):
                    rng_print(
                        f"{result.name:20s} produced {produced} -> MSB = 0b{f'{observed_bit_type}'*observed_bits}..."
                    )
                rng.step()
    else:
        print("Crafting simulation ended without reaching the goal.", file=sys.stderr)
        return


if __name__ == "__main__":
    factorio_rng_data = [0x0495497C, 0xA4F9C488, 0x6BA307A3]
    main(factorio_rng_data)
    # symbol_reg = SymbolicRegister(32, "a")
    # rng = Taus88RNG((symbol_reg, symbol_reg, symbol_reg))
    # symbol_reg = rng._single_LFSR(symbol_reg, 32, 29, 2, 4)
    # print(symbol_reg)

    # mat = build_generation_matrix(taps=range(14), top_k_bits=7)
    # render_binary_matrix_as_image(mat, "analysis/top_7_bits.png")
    # render_binary_matrix_as_image(sim_solve(mat), "analysis/sim_top_7_bits.png")
