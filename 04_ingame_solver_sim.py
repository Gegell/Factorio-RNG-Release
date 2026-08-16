from sympy import Matrix
from sympy.polys.domains import GF
from sympy.polys.matrices import DomainMatrix
from factorio_rng import (
    analyze_equation_system,
    build_generation_matrix,
    render_binary_matrix_as_image,
)
from rngs import ExtractedRNG, rng_state_t
from register import SymbolicRegister
from functools import reduce
from operator import xor
from copy import deepcopy


def get_indices_to_add():
    symbolic_rng = ExtractedRNG(
        registers=(
            SymbolicRegister(32, "a_"),
            SymbolicRegister(32, "b_"),
            SymbolicRegister(32, "c_"),
        )
    )
    symbolic_rng.step()
    indices = []
    for state in symbolic_rng.state:
        indices.append([])
        for bit_eq in state._bits:
            indices[-1].append(
                [31 - int(symbol.name.split("_")[1]) for symbol in bit_eq.free_symbols]
            )
    return indices


def get_equations_to_solve(values=None):
    U235_THRESHOLD = int(2**32 * 0.007)
    U238_THRESHOLD = int(2**32 * 0.993)

    BIT_ADDITION_INDICES = get_indices_to_add()

    values = values or [0x01234567, 0x89ABCDEF, 0xFEDCBA98]

    simulation_rng = ExtractedRNG(
        registers=(
            SymbolicRegister(32, value=values[0]),
            SymbolicRegister(32, value=values[1]),
            SymbolicRegister(32, value=values[2]),
        )
    )
    state = [[1 << i for i in reversed(range(32))] for _ in range(3)]

    def step(state):
        new_state = [[0] * 32 for _ in range(3)]
        for reg_idx, indices in enumerate(BIT_ADDITION_INDICES):
            for bit_index, bit_indices in enumerate(indices):
                new_state[reg_idx][bit_index] = reduce(
                    xor, (state[reg_idx][i] for i in bit_indices)
                )
        return new_state

    eqs = []

    crafts = 0
    while len(eqs) < 96:
        u235_value = int(simulation_rng.get_int())
        gets_u235 = u235_value <= U235_THRESHOLD
        if gets_u235:
            for i in range(7):
                eqs.append((state[0][i], state[1][i], state[2][i], 0))
        state = step(state)
        simulation_rng.step()

        # for s in state:
        #     print(s)

        u238_value = int(simulation_rng.get_int())
        gets_u238 = u238_value <= U238_THRESHOLD
        if not gets_u238:
            for i in range(7):
                eqs.append((state[0][i], state[1][i], state[2][i], 1))
        state = step(state)
        simulation_rng.step()
        # print()
        # for s in state:
        #     print(s)

        crafts += 1
        if gets_u235 or not gets_u238:
            print(
                f"{gets_u238:1d} {gets_u235:1d}  {u238_value:08x} {u235_value:08x} {simulation_rng.steps:2d}"
            )

    true_eqs = [f"{eq[0]:032b}{eq[1]:032b}{eq[2]:032b}{eq[3]}" for eq in eqs]
    mat = Matrix([list(map(int, eq)) for eq in true_eqs])
    render_binary_matrix_as_image(mat, "crafting_equations.png")
    mat_t = Matrix.hstack(mat[:, :32].T, mat[:, 32:64].T, mat[:, 64:96].T)
    render_binary_matrix_as_image(mat_t, "crafting_equations_transposed.png")
    return mat


if __name__ == "__main__":
    values = [0xFE065C2C, 0xE303068B, 0xD01C79C5]

    print(get_indices_to_add())

    mat = get_equations_to_solve(values)
    domain_mat = DomainMatrix.from_Matrix(mat)
    domain_mat = domain_mat.convert_to(GF(2))
    analyze_equation_system(domain_mat[:, :-1], domain_mat[:, -1])

    # Solve the system
    rref_mat, pivots = domain_mat.rref()
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

    print("State:")
    print(state[0])
    print(state[1])
    print(state[2])
