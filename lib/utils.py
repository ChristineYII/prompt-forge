import ast

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def parse_tool_signature(sig: str, description: str) -> dict:
    """
    Parse a Python-style function signature into a JSON Schema tool dict.

    Accepts either bare signatures like:
        lookup_order(order_id: str) -> dict
    or full def statements:
        def lookup_order(order_id: str) -> dict

    Parameters with defaults are treated as optional (omitted from 'required').
    Raises ValueError with a human-readable message on parse failure.
    """
    sig = sig.strip()
    if not sig.startswith("def "):
        sig = "def " + sig
    # Append a minimal body so ast.parse accepts it
    if not sig.rstrip().endswith(":"):
        sig += ": ..."

    try:
        tree = ast.parse(sig)
    except SyntaxError as e:
        raise ValueError(f"Invalid function signature — {e}") from e

    func = tree.body[0]
    args = func.args
    n_args = len(args.args)
    n_defaults = len(args.defaults)

    properties: dict = {}
    required: list = []

    for i, arg in enumerate(args.args):
        if arg.arg == "self":
            continue
        param_type = "string"
        if arg.annotation:
            param_type = _TYPE_MAP.get(ast.unparse(arg.annotation), "string")
        properties[arg.arg] = {"type": param_type, "description": ""}
        if i < (n_args - n_defaults):
            required.append(arg.arg)

    return {
        "name": func.name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
