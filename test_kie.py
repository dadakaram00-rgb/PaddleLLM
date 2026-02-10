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
        # We removed find_model_dir, so check if load_ocr_model and others exist
        self.assertTrue(hasattr(kie_pipeline, 'load_ocr_model'))
        self.assertTrue(hasattr(kie_pipeline, 'load_llm_model'))
        self.assertTrue(hasattr(kie_pipeline, 'extract_text_from_image'))
        self.assertTrue(hasattr(kie_pipeline, 'perform_kie'))

    @patch('kie_pipeline.PaddleOCR')
    def test_load_ocr_model_default(self, mock_paddleocr):
        # Mock PaddleOCR class
        mock_instance = MagicMock()
        mock_paddleocr.return_value = mock_instance

        ocr = kie_pipeline.load_ocr_model()

        # Verify it was called with expected default args
        mock_paddleocr.assert_called_once_with(
            use_angle_cls=True,
            lang='de',
            use_gpu=False,
            show_log=False
        )
        self.assertEqual(ocr, mock_instance)

if __name__ == '__main__':
    unittest.main()
