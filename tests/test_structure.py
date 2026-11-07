"""The layout rules every module keeps (see CONTRIBUTING.md), checked from the source
so they cannot drift: steps read like a table of contents, figures go through
agulhas.plotting, and the pipeline map lists every step."""

import ast
import pathlib

import pytest

from agulhas import pipeline

ROOT = pathlib.Path(pipeline.__file__).parent
REPO = ROOT.parent
MODULES = sorted(ROOT.rglob("*.py"))
STEP_MODULES = sorted({s.module for s in pipeline.STEPS + pipeline.TOOLS})
MAX_MAIN_LINES = 40
MAX_FUNCTION_LINES = 80


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def step_path(module):
    return ROOT.joinpath(*module.split(".")).with_suffix(".py")


def functions(path):
    return [n for n in ast.walk(tree(path)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def imports(path):
    names = set()
    for n in ast.walk(tree(path)):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module)
    return names


@pytest.mark.parametrize("module", STEP_MODULES)
def test_a_step_starts_with_its_title_and_ends_in_a_short_main(module):
    path = step_path(module)
    doc = ast.get_docstring(tree(path))
    assert doc and doc.startswith("Agulhas Region — "), f"{module}: docstring title"
    main = [f for f in functions(path) if f.name == "main"]
    assert len(main) == 1, f"{module}: one main()"
    args = main[0].args
    assert [a.arg for a in args.args] == ["argv"] and len(args.defaults) == 1, f"{module}: main(argv=None)"
    length = main[0].end_lineno - main[0].lineno + 1
    assert length <= MAX_MAIN_LINES, f"{module}: main() is {length} lines"
    assert path.read_text(encoding="utf-8").rstrip().endswith('if __name__ == "__main__":\n    main()')


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_function_is_too_long(path):
    long = [(f.name, f.end_lineno - f.lineno + 1) for f in functions(path)
            if f.end_lineno - f.lineno + 1 > MAX_FUNCTION_LINES]
    assert not long, f"split these: {long}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_figures_go_through_the_shared_plotting_module(path):
    """agulhas.plotting sets the headless backend before pyplot is used, and only it does."""
    if path.name == "plotting.py":
        return
    used = imports(path)
    if "matplotlib.pyplot" in used:
        assert "agulhas.plotting" in used, "import agulhas.plotting (headless backend, save_figure)"
    calls = [n for n in ast.walk(tree(path)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "use" and getattr(n.func.value, "id", "") == "matplotlib"]
    assert not calls, "matplotlib.use belongs in agulhas.plotting"


def test_the_pipeline_map_lists_every_step():
    text = (REPO / "docs" / "pipeline.md").read_text(encoding="utf-8")
    missing = [s.name for s in pipeline.STEPS + pipeline.TOOLS if f"`{s.name}`" not in text]
    assert not missing, f"add to docs/pipeline.md: {missing}"


def test_the_architecture_page_lists_every_shared_module():
    text = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    shared = [p.name for p in ROOT.glob("*.py") if p.name not in ("__init__.py", "__main__.py")]
    missing = [name for name in shared if f"`{name}`" not in text]
    assert not missing, f"add to docs/architecture.md: {missing}"
