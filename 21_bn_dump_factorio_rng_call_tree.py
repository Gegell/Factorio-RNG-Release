# dump factorio rng call tree
#
import re
from binaryninja import demangle


def recurse_callers(entry_points, max_depth=2):
    seen = set()
    parents = {}

    def recursive_search(fn, depth_remaining):
        if fn in seen or depth_remaining <= 0:
            return
        callers = fn.callers
        for caller in callers:
            recursive_search(caller, depth_remaining - 1)
        parent_set = parents.setdefault(fn, set())
        parent_set |= callers
        seen.add(fn)

    for fn in entry_points:
        recursive_search(fn, max_depth)
    return parents


def find_largest_bracket_spans(string):
    stack = []
    opening = "<[({`"
    closing = ">])}'"
    assoc = {o: c for o, c in zip(opening, closing)}
    for idx, char in enumerate(string):
        if char in opening:
            stack.append((idx, char))
        elif char in closing:
            open_idx, open_char = stack.pop()
            assert assoc[open_char] == char, "Inconsistent brackets encountered!"
            if not stack:
                yield (open_idx, idx, open_char)


def replace_spans_with(spans, string, repl=""):
    idx = 0
    new_string = ""
    for start, end, _ in spans:
        new_string += string[idx : start + 1] + repl
        idx = end
    new_string += string[idx:]
    return new_string


def demangle_lambda_invoker(fn_name):
    mat = re.search(
        r"lambda_invoker.*registerLoader@V(\w+)@.*InstanceLoader.*MapDeserialiser",
        fn_name,
    )
    if not mat:
        print(f"Could not demangle lambda_invoker {fn_name}")
    return f"InstanceLoader.lambda<{mat[1]}>"


def get_better_name(fn):
    typ, name = demangle.demangle_ms(fn.arch, fn.name)
    if isinstance(name, list):
        name = demangle.get_qualified_name(name)
    if typ is None and "lambda_invoker" in name:
        name = demangle_lambda_invoker(name)

    # Apply name shortening to keep the graph more manageable.
    if name.startswith("EntityOrderHelpers::EntityOrderList"):
        name = re.sub("<(.*?)>", "<...>", name, count=1)
    elif name.startswith("ConstructionManager::findRobotFor"):
        mat = re.search(r"<(\w+?),`([\w:]+)'", name)
        for_what = mat[1]
        job = mat[2]
        name = f"ConstructionManager::findRobotFor<{for_what}, {job}>"
    elif name.startswith("std::_Func_impl_no_alloc"):
        span = list(find_largest_bracket_spans(name))[0]
        name = name[span[0] : span[1]]
    elif name.startswith("std::vector<class MapTick") and ("FlameParticle" in name):
        name = replace_spans_with(
            find_largest_bracket_spans(name), name, "FireFlame::FlameParticle"
        )
    elif name.startswith(
        "std::vector<class std::unique_ptr<class AsteroidCollectorArm"
    ):
        name = replace_spans_with(
            find_largest_bracket_spans(name),
            name,
            "std::unique_ptr<AsteroidCollectorArm>",
        )
    elif name.startswith("std::unique_ptr<class AsteroidCollectorArm"):
        name = replace_spans_with(
            find_largest_bracket_spans(name), name, "AsteroidCollectorArm"
        )
    elif "AsteroidCollector" in name and "MapDeserialiser" in name:
        name = "lambda<MapDeserialiser<AsteroidCollectorArm>>???"
    elif "AsteroidCollectorArm" in name and "_Emplace_reallocate" in name:
        name = "_Emplace_reallocate<std::unique_ptr<AsteroidCollector>>"
    elif name.startswith("std::variant<struct SegmentedUnitAI"):
        name = "std::variant<struct SegmentedUnitAI::*>"

    name = name.replace("struct ", "")
    name = name.replace("class ", "")

    return name


def write_dot(path, callers):
    fn_names = {}
    for child, parents in callers.items():
        if child not in fn_names:
            fn_names[child] = get_better_name(child)
        for parent in parents:
            if parent not in fn_names:
                fn_names[parent] = get_better_name(parent)

    with open(path, "w") as f:
        f.write("digraph {\n")
        f.write("    node[shape=box, style=filled, colorscheme=set310];\n")
        f.write("    rankdir=LR;\n")
        cmap = dict()
        for fn, name in fn_names.items():
            tags = fn.get_function_tags(tag_type="Includes RNG")
            if tags:
                tag = tags[0].data
                types = tag.split("; ")[1].split(" | ")
                map_type = "other"
                for typ in types:
                    if typ.startswith("Map."):
                        map_type = typ.split(" ")[0]
                        break
                if map_type not in cmap:
                    cmap[map_type] = len(cmap) + 1
                col = cmap[map_type]
                f.write(f'    "{name}"[fillcolor={col}]\n')
        for child, parents in callers.items():
            for parent in parents:
                con = f'"{fn_names[parent]}" -> "{fn_names[child]}"'
                f.write(f"    {con};\n")

        # Write the legend subgraph
        f.write("    subgraph cluster_legend {\n")
        f.write("        label=Legend;\n")
        f.write("        bgcolor=gray80;\n")
        for name, col in cmap.items():
            f.write(f'        "legend_{col}"[label="{name}", fillcolor={col}]\n')
        f.write("    }\n")

        f.write("}\n")


def get_rng_fns():
    rng_names = [
        "RandomGenerator::uniformInteger",
        "RandomGenerator::uniformDouble",
        "RandomGenerator::getInt",
        "RandomGenerator::normal",
        "RandomGenerator::uniformFloat",
        "RandomGenerator::uniformSignedInteger",
        "RandomGenerator::testChance",
    ]
    rng_fns = []
    for name in rng_names:
        rng_fns.extend(bv.get_functions_by_name(name))
    return rng_fns


rng_fns = get_rng_fns()
callers = recurse_callers(rng_fns, 5)
write_dot("C:/Users/Alex/Desktop/rng_callers.dot", callers)
