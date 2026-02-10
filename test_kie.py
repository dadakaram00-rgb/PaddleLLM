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
        self.assertTrue(hasattr(kie_pipeline, 'load_ocr_model'))
        self.assertTrue(hasattr(kie_pipeline, 'load_llm_model'))
        self.assertTrue(hasattr(kie_pipeline, 'extract_text_from_image'))
        self.assertTrue(hasattr(kie_pipeline, 'perform_kie'))

    @patch('kie_pipeline.PaddleOCR')
    def test_load_ocr_model_default(self, mock_paddleocr):
        mock_instance = MagicMock()
        mock_paddleocr.return_value = mock_instance
        ocr = kie_pipeline.load_ocr_model()
        mock_paddleocr.assert_called_once()
        self.assertEqual(ocr, mock_instance)

    @patch('kie_pipeline.AutoTokenizer')
    @patch('kie_pipeline.AutoModelForCausalLM')
    def test_load_llm_auto_success(self, mock_model, mock_tokenizer):
        mock_tokenizer.from_pretrained.return_value = "tokenizer"
        mock_model.from_pretrained.return_value = MagicMock()

        tok, mod = kie_pipeline.load_llm_model()
        self.assertEqual(tok, "tokenizer")

    def test_load_llm_fallback(self):
        # We need to mock the import of Qwen2Tokenizer inside the function
        # This is tricky because it's inside the function scope.
        # However, we can mock sys.modules to ensure 'paddlenlp.transformers' has Qwen2Tokenizer

        with patch('kie_pipeline.AutoTokenizer') as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("Auto failed")

            with patch.dict(sys.modules, {'paddlenlp.transformers': MagicMock()}):
                # Ensure Qwen2Tokenizer is available on the mock module
                mock_qwen = MagicMock()
                sys.modules['paddlenlp.transformers'].Qwen2Tokenizer = mock_qwen
                mock_qwen.from_pretrained.return_value = "qwen_tokenizer"

                with patch('kie_pipeline.AutoModelForCausalLM') as mock_model:
                     mock_model.from_pretrained.return_value = MagicMock()

                     tok, mod = kie_pipeline.load_llm_model()
                     self.assertEqual(tok, "qwen_tokenizer")

if __name__ == '__main__':
    unittest.main()
