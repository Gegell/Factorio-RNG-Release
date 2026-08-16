# Factorio RNG Breaking – Appendix
This repository contains the code and additional data used to break the RNG in Factorio.
I talk about the theoretical aspects in the main writeup on my [blog](https://gegell.github.io/posts/factorio-rng).

This appendix contains both some scripts I used to generate parts of the machine
and some theoretical foundation in addition to the main writeup.
Files which act as entrypoints to individual topics are prefixed with a number.

To run the scripts, first install the dependencies with `pip install -r requirements.txt`.

> [!NOTE]
> **!!Version 2.0 only!!**
> 
> The repo and most of the research was conducted in Factorio version 1.0 and 2.0.
> In version 2.1 there were breaking changes to the RNG calling and the where the RNG state is used in (and as of the 16.08.2026) is still undergoing changes.
> As such the provided world & blueprints currently only work in version 2.0 and will not work in 2.1.

## Theory
- [`00_equality_proof.ipynb`](./00_equality_proof.ipynb) uses symbolic registers
  to prove that Boost's `taus88` implementation and the reverse-engineered game
  code perform the same RNG state transition.
- [`01_factorio_scrap_recycle.ipynb`](./01_factorio_scrap_recycle.ipynb)
  processes Factorio recipe data and ranks recipes which expose information
  about the RNG state through probabilistic or fractional outputs.
- [`02_recycler_skipping_math.ipynb`](./02_recycler_skipping_math.ipynb)
  derives and verifies the integer-only calculation needed to skip to a target
  number of RNG calls with multiple recyclers and scrap productivity. The final
  form is organized for translation into circuit-network combinators.

Note that the scripts all rely on [`register.py`](./register.py) and [`rngs.py`](./rngs.py) which contain the RNG reimplementations.
Additionally a cluster of utility functions for analysis of the transition matrices and some demo prediction code is given in the [`factorio_rng.py`](./factorio_rng.py).

## Blueprints
Many parts of the machine rely on large lookup tables.
To avoid having to manually copy those values into the game I used [factorio-draftsman](https://github.com/redruin1/factorio-draftsman)
to generate blueprint strings which can be imported into the game.

- [`10_bp_rng_reader_intra_tick.py`](./10_bp_rng_reader_intra_tick.py) generates
  a reader which performs many recycler RNG samples in the same tick and uses
  an inverse observation matrix to reconstruct the three internal LFSR states.
- [`11_bp_rng_advancer.py`](./11_bp_rng_advancer.py) generates an RNG-state
  advancer. It stores transition-matrix columns in constant combinators and
  uses circuit-network matrix multiplication to calculate future LFSR states.

## Reversing
The following are the scripts I've used in conjunction with [Binary Ninja](https://binary.ninja/) to reverse engineer the call tree and RNG type propagation from the binary.
Note that to run the scripts one requires a paid version of binary ninja as the free version does not support the Python API.
Additionally it requires an older version of binary ninja (<=5.0.7290) as newer versions broke the API, and I haven't bothered to find a workaround yet.[^broken]
Lastly, they were not designed to be run standalone but rather inside binary ninja as snippets.
- [`20_bn_tag_factorio_rng_references.py`](./20_bn_tag_factorio_rng_references.py) tags all instructions which reference the RNG state structure in the binary, propagating those to their callers and so on.
  This script appears to be broken in the latest version (5.2.8722) as `bv.get_code_refs_for_type(...)` now appears to also return *indirect* references and I've not yet found a workaround for that.
- [`21_bn_dump_factorio_rng_call_tree.py`](./21_bn_dump_factorio_rng_call_tree.py) is to be called after the references have been tagged with the previous script.
  It dumps the call tree as a dot graph, which can be visualized with [Graphviz](https://graphviz.org/).

[^broken]: `bv.get_type_refs_for_type` in newer versions returns both direct and indirect references. The Factorio `Map` object is counted as an indirect reference to the RNG objects (which makes sense), this however means it flags essentially all code as potentially referencing the RNG. If you have a license, you can downgrade to an older version via their portal, which is very nice :).

## Ingame testing
I've also added both a blueprint string containing multiple RNG manipulation machines and a savegame showing some of the progression of the machines over time.
**Note that these variants only work in 2.0 not 2.1!**
- [`30_rng_breaker_blueprint_book.txt`](./30_rng_breaker_blueprint_book.txt) contains a blueprint book.
  Note that the sub-tick version might require some care when building / replacing the input combinators to the recyclers after the fact such that they get triggered in the correct order.
- [`Breaking PRNG release.zip`](./Breaking%20PRNG%20release.zip) contains a savegame with the machines in action.
  Copy the savegame into your Factorio saves folder to have a predefined setup ready to go.
