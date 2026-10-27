import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('krea_code_test', Path(__file__).resolve().parents[1] / 'realtime/model_code.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelCodeTests(unittest.TestCase):
    def create_model(self, root, value):
        folder = Path(root) / 'transformer'
        folder.mkdir()
        (folder / 'attention.py').write_text(f'VALUE = {value!r}\n')
        (folder / 'model.py').write_text('from .attention import VALUE\n')
        (folder / 'causal_model.py').write_text('from .model import VALUE\nclass CausalWanModel:\n    marker = VALUE\n')

    def test_clean_import_resolves_transitive_siblings_and_isolates_models(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.create_model(a, 'first')
            self.create_model(b, 'second')
            first = module.load_transformer_class(a)
            second = module.load_transformer_class(b)
            self.assertEqual(first.marker, 'first')
            self.assertEqual(second.marker, 'second')
            self.assertIsNot(first, second)
            self.assertIs(first, module.load_transformer_class(a))

    def test_missing_sibling_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as root:
            self.create_model(root, 'first')
            (Path(root) / 'transformer/attention.py').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'attention.py'):
                module.load_transformer_class(root)
