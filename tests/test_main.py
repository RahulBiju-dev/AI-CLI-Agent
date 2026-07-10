import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Mock expensive startup dependencies only while importing this test subject;
# restoring sys.modules prevents this module from poisoning later test imports.
with patch.dict(sys.modules, {
    'ollama': unittest.mock.MagicMock(),
    'agent.core': unittest.mock.MagicMock(),
    'agent.web': unittest.mock.MagicMock(),
    'rich': unittest.mock.MagicMock(),
    'rich.console': unittest.mock.MagicMock(),
}):
    import main
from main import _get_modelfile_path

class TestMain(unittest.TestCase):
    def test_get_modelfile_path_default(self):
        # Remove _MEIPASS if it exists to test default behavior
        if hasattr(sys, '_MEIPASS'):
            del sys._MEIPASS

        path = _get_modelfile_path()
        expected = os.path.join(os.path.dirname(os.path.abspath(main.__file__)), 'Modelfile')
        self.assertEqual(path, expected)

    def test_get_modelfile_path_meipass(self):
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path(temporary) / 'Modelfile'
            expected.touch()
            with patch.object(sys, '_MEIPASS', temporary, create=True):
                path = _get_modelfile_path()
            self.assertEqual(path, str(expected))

if __name__ == '__main__':
    unittest.main()
