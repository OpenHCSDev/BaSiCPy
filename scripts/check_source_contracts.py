"""Light source-only checks; deliberately do not import numerical backends."""

from __future__ import annotations

import ast
import configparser
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_source_parses_on_running_interpreter(self):
        for path in (ROOT / "src").rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(), filename=str(path))

    def test_inverse_dct_uses_public_owner(self):
        path = ROOT / "src/basicpy/tools/dct_tools.py"
        tree = ast.parse(path.read_text())
        owner = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "JaxDCT"
        )
        inverse = next(
            node for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "idctnd"
        )
        self.assertEqual(ast.unparse(inverse.body[0].value.func), "jax.scipy.fft.idctn")
        self.assertFalse((ROOT / "src/basicpy/tools/_jax_idct.py").exists())
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("jax._"), path)
                    self.assertNotIn("_jax_idct", node.module or "", path)

    def test_python314_dependency_contract(self):
        config = configparser.ConfigParser()
        config.read(ROOT / "setup.cfg")
        dependencies = config["options"]["install_requires"].split()
        self.assertEqual(config["options"]["python_requires"], ">=3.11")
        self.assertIn("jax>=0.9.2,<0.10", dependencies)
        self.assertIn("scipy>=1.16.2", dependencies)
        self.assertIn("pydantic>=2.12.0,<3.0.0", dependencies)
        self.assertFalse(any(dep.startswith("hyperactive") for dep in dependencies))
        self.assertIn("autotune", config["options.extras_require"])

    def test_supported_pydantic_owner_api(self):
        for relative in ("src/basicpy/basicpy.py", "src/basicpy/_jax_routines.py"):
            tree = ast.parse((ROOT / relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ("dict", "json", "__fields__"), relative)
                if isinstance(node, ast.ClassDef):
                    self.assertNotEqual(node.name, "Config", relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
