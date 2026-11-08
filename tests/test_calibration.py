import unittest

from qwen_quant_lab.calibration import render_calibration_examples


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "|".join(item["content"] for item in messages)


class CalibrationTests(unittest.TestCase):
    def test_invalid_examples_are_dropped(self):
        rendered, stats = render_calibration_examples(
            [
                {"messages": [{"role": "user", "content": "hello"}]},
                {"messages": []},
                {"not_messages": True},
            ],
            FakeProcessor(),
        )
        self.assertEqual(rendered, [{"text": "hello"}])
        self.assertEqual(stats, {"processed": 1, "dropped": 2})

    def test_chat_template_kwargs_are_forwarded(self):
        processor = FakeProcessor()
        render_calibration_examples(
            [{"messages": [{"role": "user", "content": "hi"}]}],
            processor,
            {"enable_thinking": False},
        )
        self.assertEqual(
            processor.kwargs,
            {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False},
        )


if __name__ == "__main__":
    unittest.main()

