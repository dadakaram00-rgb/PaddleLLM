import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before import
sys.modules['paddle'] = MagicMock()
sys.modules['paddleocr'] = MagicMock()
sys.modules['paddlenlp'] = MagicMock()
sys.modules['paddlenlp.transformers'] = MagicMock()

import unittest
from unittest.mock import patch
import kie_pipeline

class TestKIEPipeline(unittest.TestCase):
    def test_functions_exist(self):
        self.assertTrue(hasattr(kie_pipeline, 'find_model_dir'))
        self.assertTrue(hasattr(kie_pipeline, 'load_ocr_model'))
        self.assertTrue(hasattr(kie_pipeline, 'load_llm_model'))
        self.assertTrue(hasattr(kie_pipeline, 'extract_text_from_image'))
        self.assertTrue(hasattr(kie_pipeline, 'perform_kie'))

    def test_find_model_dir(self):
        with patch('os.walk') as mock_walk:
            # Structure: (root, dirs, files)
            # Create fake structure
            mock_walk.return_value = [
                ('/models', ['ocr'], []),
                ('/models/ocr', ['whl'], []),
                ('/models/ocr/whl', ['det', 'rec'], []),
                ('/models/ocr/whl/det', ['en'], []),
                ('/models/ocr/whl/det/en', [], ['inference.pdmodel']),
                ('/models/ocr/whl/rec', ['de', 'en'], []),
                ('/models/ocr/whl/rec/de', [], ['inference.pdmodel']),
                ('/models/ocr/whl/rec/en', [], ['inference.pdmodel']),
            ]

            # Test simple find
            det = kie_pipeline.find_model_dir('/models', 'det')
            self.assertEqual(det, '/models/ocr/whl/det/en')

            # Test preferred language
            rec = kie_pipeline.find_model_dir('/models', 'rec', preferred_lang='de')
            self.assertEqual(rec, '/models/ocr/whl/rec/de')

            # Test missing
            cls = kie_pipeline.find_model_dir('/models', 'cls')
            self.assertIsNone(cls)

if __name__ == '__main__':
    unittest.main()
