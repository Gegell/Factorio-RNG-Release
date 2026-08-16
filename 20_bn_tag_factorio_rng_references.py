#
#
from collections import Counter, defaultdict
import re
from binaryninja import (
    HighLevelILAssign,
    HighLevelILAdd,
    HighLevelILDerefField,
    HighLevelILVarInit,
    HighLevelILVarPhi,
    InstructionTextTokenType,
    TypeReferenceType,
    Variable,
    ILException,
)

# Define a couple of emojis in unicode encoding, as otherwise the syntax highlighter of the snippets breaks.
dice = "\U0001f3b2"  # dice
qmark = "\u2753"  # red question mark
cmark = "\u2714\ufe0f"  # green check mark
cross = "\u274c"  # red cross
dexclmark = "\u203c\ufe0f"  # double exclamation mark

unknown = "unknown"
nohlil = "no HLIL available"
maxrecurse = "max recursion reached"


# Record some direct reference offsets to RandomGenerators in structs
# these can be used if only the offset is available instead of the full
# reversed code
def get_member_from_type_reference_source(trs):
    return bv.types[trs.name].member_at_offset(trs.offset)


known_rng_refs = {}
unique_offsets = {}
for trs in bv.get_type_refs_for_type("RandomGenerator"):
    if trs.ref_type != TypeReferenceType.DirectTypeReferenceType:
        continue
    try:
        member = get_member_from_type_reference_source(trs)
        known_rng_refs.setdefault(trs.name, []).append(member)
        unique_offsets.setdefault(member.offset, []).append((trs.name, member))
    except ValueError:
        pass
    except AttributeError as e:
        print(trs.name, trs.offset, e)
        pass
unique_offsets = {k: v[0] for k, v in unique_offsets.items() if len(v) == 1 and k >= 0x100}


# Attempt to decode the given instruction wrt. which RNG gets used,
# and further explore the callers of this function
def get_instr_type(instr, _max_depth=3):
    if _max_depth < 0:
        return maxrecurse

    # Compare with HLIL string against known generator strings
    instr = instr.non_ssa_form
    str_instr = str(instr)
    MapRNGs = [
        "unsafeRandom",
        "generalRandomGenerator",
        "aiRandomGenerator",
        "entitiesRandomGenerator",
        "mapRandomGenerator",
        "triggersRandomGenerator",
    ]
    for map_rng in MapRNGs:
        if map_rng in str_instr:
            return f"Map.{map_rng}"

    if "TriggerContext::getRNG(" in str_instr or "getTriggersRandomGenerator" in str_instr:
        return "Map.triggersRandomGenerator | GlobalContext.randomGenerator"

    if "asteroidsRandomGenerator" in str_instr:
        return "SpacePlatform.asteroidsRandomGenerator"

    if re.search(r"global(_1)?->(randomGenerator|__offset\(0x238\))", str_instr):
        return "GlobalContext.randomGenerator"
    elif "this->random" in str_instr:
        this_var = [v for v in instr.vars if v.name == "this"][0]
        str_type = str(this_var.type)
        if "GlobalContext" in str_type:
            return "GlobalContext.randomGenerator"
        elif "LightningMeshGenerator" in str_type:
            return "LightningMeshGenerator.random"
        elif "CargoBayConnectableDrawLogic" in str_type:
            return "CargoBayConnectableDrawLogic.random"
        elif "SelectorCombinatorControlBehavior" in str_type:
            return "SelectorCombinatorControlBehavior.random"
        elif "SoundRandomizer" in str_type:
            return "SoundRandomizer.randomGenerator"

    for tok in instr.tokens:
        if tok.type != InstructionTextTokenType.IntegerToken:
            continue
        offset = unique_offsets.get(tok.value, None)
        if offset:
            cls_name, member = offset
            return f"{cls_name}.{member.name} via offset {tok.value}"

    # Check if the instruction is a pointer assignment
    if isinstance(instr, HighLevelILVarInit) and str(instr.dest.type) == "class RandomGenerator*":
        if isinstance(instr.src, HighLevelILAdd):
            left, right = instr.src.operands
            cont = True
            if isinstance(left, HighLevelILDerefField):
                base, offset = left, right
            elif isinstance(right, HighLevelILDerefField):
                base, offset = right, left
            else:
                cont = False
            if cont:
                offsets = offset.possible_values
                cont = offsets.count > 0
            if cont:
                cls_name = str(base.expr_type.target).removeprefix("class ").removeprefix("struct ")
                members = known_rng_refs.get(cls_name, [])
                types = set()
                for o in offset.possible_values.values:
                    for member in members:
                        if member.offset == o:
                            types.add(f"{cls_name}.{member.name}")
                            break
                    else:
                        types.add(f"{cls_name}.unknown[{o}]")
                if types:
                    return " | ".join(sorted(types))

    # Check if the instruction contains a reference to a generator
    instr = instr.ssa_form
    for ssa_var in instr.vars:
        if "RandomGenerator" not in str(ssa_var.type):
            continue
        # If not the first occurrence, try again at the definition point
        def_site = getattr(ssa_var, "def_site", None)
        if def_site and (instr != def_site):
            if isinstance(def_site, HighLevelILVarPhi):
                types = set()
                for branch in def_site.src:
                    sub_types = get_instr_type(branch.def_site, _max_depth - 1).split(" | ")
                    types |= set(sub_types)
                return " | ".join(sorted(types))
            elif isinstance(def_site, HighLevelILAssign):
                return get_instr_type(def_site.src, _max_depth - 1)

            return get_instr_type(def_site, _max_depth - 1)

        # Check if the generator was passed into the function
        var = ssa_var if isinstance(ssa_var, Variable) else ssa_var.var
        if var.is_parameter_variable:
            # TODO: Track across function border
            fn = var.function
            params = fn.parameter_vars.vars
            return f"parameter[{params.index(var)}] = {var.name}"

    return unknown


def get_emoji(remark):
    if maxrecurse in remark:
        return dexclmark
    if unknown in remark or nohlil in remark:
        return qmark
    return cmark


bv.remove_tag_type("Includes RNG")
bv.create_tag_type("Includes RNG", dice)
bv.remove_tag_type("RNG Call")
bv.create_tag_type("RNG Call", dice)

references = list(bv.get_code_refs_for_type("RandomGenerator"))
seen_fns = defaultdict(Counter)
merged_fn_types = defaultdict(set)
seen_addresses = set()
ref_stats = Counter()
for ref in references:
    addr = ref.address
    fn = ref.function
    try:
        if ref.hlil:
            instr = ref.hlil.instr
            addr = instr.address
            typ = get_instr_type(instr)
        else:
            typ = nohlil
    except ILException:
        typ = nohlil
    if addr not in seen_addresses:
        emoji = get_emoji(typ)
        ref_stats[emoji] += 1
        seen_fns[fn][emoji] += 1
        merged_fn_types[fn] |= set(typ.split(" | "))
        bv.add_tag(addr, "RNG Call", f"{emoji} {typ}")
        seen_addresses.add(addr)

total_fn_stats = Counter()
for fn, fn_stats in seen_fns.items():
    success = fn_stats[cmark]
    total = fn_stats.total()
    emoji = cmark if success == total else cross
    total_fn_stats[emoji] += 1
    merged_type = " | ".join(merged_fn_types[fn])
    fn.add_tag("Includes RNG", f"{emoji} {success}/{total} refs associated; {merged_type}")

print("Reference stats:", ref_stats)
print("Function stats:", total_fn_stats)
