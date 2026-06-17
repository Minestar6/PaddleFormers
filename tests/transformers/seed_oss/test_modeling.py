# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import tempfile
import unittest

import paddle

from paddleformers.cli.utils.llm_utils import get_lora_target_modules
from paddleformers.datasets.template.template import TEMPLATES
from paddleformers.transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from paddleformers.transformers import SeedOssConfig, SeedOssForCausalLM, SeedOssModel


class SeedOssModelTest(unittest.TestCase):
    def get_config(self):
        return SeedOssConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_dropout=0.0,
            residual_dropout=0.0,
            use_cache=False,
        )

    def test_config_default_rope_parameters(self):
        config = SeedOssConfig()
        self.assertEqual(config.model_type, "seed_oss")
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertEqual(config.rope_parameters["rope_theta"], 10000000.0)

    def test_auto_config_and_model_registration(self):
        config = self.get_config()
        model = AutoModel.from_config(config)
        self.assertIsInstance(model, SeedOssModel)

        model_class = AutoModelForCausalLM._get_model_class_from_config(
            None,
            None,
            {"architectures": ["SeedOssForCausalLM"], "model_type": "seed_oss"},
        )
        self.assertIs(model_class, SeedOssForCausalLM)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/config.json", "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f)
            loaded_config = AutoConfig.from_pretrained(tmpdir)
        self.assertIsInstance(loaded_config, SeedOssConfig)

    def test_causal_lm_forward(self):
        paddle.seed(2026)
        config = self.get_config()
        model = SeedOssForCausalLM(config)
        model.eval()

        input_ids = paddle.randint(0, config.vocab_size, shape=[2, 5], dtype="int64")
        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(outputs.logits.shape, [2, 5, config.vocab_size])

    def test_lora_targets_and_template_registration(self):
        class Model:
            config = self.get_config()

        self.assertEqual(
            get_lora_target_modules(Model()),
            [
                ".*q_proj.*",
                ".*k_proj.*",
                ".*v_proj.*",
                ".*o_proj.*",
                ".*gate_proj.*",
                ".*up_proj.*",
                ".*down_proj.*",
            ],
        )
        self.assertIn("seed_oss", TEMPLATES)
        self.assertEqual(TEMPLATES["seed_oss"].suffix, ["<|end_of_text|>"])


if __name__ == "__main__":
    unittest.main()
