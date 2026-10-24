"""Verilog parsing: AST -> directed, scoped instance/net connections.

Module 2 changes vs. Module 1:
  * every connection now carries a direction ("input" / "output" / "unknown")
  * connections are scoped by the module they live in, so two modules that both
    have a net called `a` no longer get merged into one node
  * primary inputs / outputs of every module are collected (used later for the
    "distance to output" feature)
"""

import os
import re

from pyverilog.vparser.parser import parse
from pyverilog.vparser.ast import (
    InstanceList, Instance, Identifier, Pointer, Partselect,
    ModuleDef, Decl, Input, Output, Inout,
)

# Verilog built-in gate primitives: by language rule, port 0 is the output.
PRIMITIVE_GATES = {
    "and", "nand", "or", "nor", "xor", "xnor", "not", "buf",
    "bufif0", "bufif1", "notif0", "notif1",
}

# Fallback port-name guesses for cells whose definition is not in the file
# (standard-cell style netlists). Anything that matches neither -> "unknown".
_OUTPUT_NAME = re.compile(r"^(q|qn|qb|q_n|y|z|zn|o|out|dout|co|sum)\d*$", re.I)
_INPUT_NAME = re.compile(
    r"^(a|b|c|d|i|in|din|clk|clock|ck|cp|rst|reset|rn|sn|set|clr|clear|en|enable|sel|cin|e)\d*$",
    re.I,
)


def parse_verilog(file_path):
    """Parse a Verilog file into an AST. Returns None if parsing fails."""
    print(f"Parsing: {file_path}")
    try:
        include_dir = [os.path.dirname(os.path.abspath(file_path))]
        ast, _ = parse([file_path], preprocess_include=include_dir, debug=False)
        return ast
    except Exception as e:
        print(f"❌ Failed to parse {file_path}: {e}")
        return None


def _decl_direction(node):
    if isinstance(node, Input):
        return "input"
    if isinstance(node, Output):
        return "output"
    return "inout"


def _module_ports(moddef):
    """Return ({port_name: direction}, [port names in declaration order])."""
    directions, order = {}, []
    portlist = getattr(moddef, "portlist", None)
    for p in (getattr(portlist, "ports", None) or []):
        first = getattr(p, "first", None)  # ANSI style: Ioport(first=Input/Output..)
        if first is not None:
            order.append(first.name)
            directions[first.name] = _decl_direction(first)
        else:  # non-ANSI style: plain Port(name)
            order.append(getattr(p, "name", None))
    for item in moddef.items:  # non-ANSI style: `input a, b;` declarations
        if isinstance(item, Decl):
            for d in item.list:
                if isinstance(d, (Input, Output, Inout)):
                    directions[d.name] = _decl_direction(d)
    return directions, order


def _resolve_direction(module, port_name, position, module_info):
    """Decide whether a port is an input or an output of the instantiated cell."""
    if module.lower() in PRIMITIVE_GATES:
        return "output" if position == 0 else "input"
    if module in module_info:  # cell defined in the same file: use real declarations
        directions, order = module_info[module]
        if port_name:
            return {"input": "input", "output": "output"}.get(directions.get(port_name), "unknown")
        if position < len(order) and order[position] is not None:
            return {"input": "input", "output": "output"}.get(directions.get(order[position]), "unknown")
        return "unknown"
    if port_name:  # last resort: guess from the port name
        if _OUTPUT_NAME.match(port_name):
            return "output"
        if _INPUT_NAME.match(port_name):
            return "input"
    return "unknown"


def extract_netlist(ast):
    """Return a dict describing the netlist.

    {
      "connections": [(scope, instance, module, port, signal, direction), ...],
      "primary_inputs":  {(scope, net_name), ...},
      "primary_outputs": {(scope, net_name), ...},
    }
    `scope` is the name of the module the instance lives in.
    """
    result = {"connections": [], "primary_inputs": set(), "primary_outputs": set()}
    if ast is None:
        return result

    moddefs = [d for d in ast.description.definitions if isinstance(d, ModuleDef)]
    module_info = {m.name: _module_ports(m) for m in moddefs}

    for m in moddefs:
        directions, _ = module_info[m.name]
        for name, d in directions.items():
            if d == "input":
                result["primary_inputs"].add((m.name, name))
            elif d == "output":
                result["primary_outputs"].add((m.name, name))

        anon = 0
        for child in m.items:
            if not isinstance(child, InstanceList):
                continue
            for inst in child.instances:
                if not isinstance(inst, Instance):
                    continue
                inst_name = inst.name
                if not inst_name:  # unnamed primitive, e.g.  and (o, a, b);
                    inst_name = f"{inst.module}_anon{anon}"
                    anon += 1
                for pos, port in enumerate(inst.portlist or []):
                    if port.argname is None or port.argname == "":
                        continue
                    signal = _signal_name(port.argname)
                    if not signal:
                        continue
                    port_name = port.portname if port.portname else f"pin{pos}"
                    direction = _resolve_direction(inst.module, port.portname, pos, module_info)
                    result["connections"].append(
                        (m.name, inst_name, inst.module, port_name, signal, direction)
                    )
    return result


def _signal_name(node):
    """Get a useful signal name from common PyVerilog expression nodes."""
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, (Pointer, Partselect)):
        return _signal_name(node.var)
    name = getattr(node, "name", None)
    return name if isinstance(name, str) else None
