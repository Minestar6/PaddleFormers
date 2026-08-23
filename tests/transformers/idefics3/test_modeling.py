# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import unittest

import paddle

from paddleformers.transformers import Idefics3Config, Idefics3ForConditionalGeneration, Idefics3Model
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin, ModelTesterPretrainedMixin


class Idefics3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=3,
        image_token_id=4,
        pad_token_id=0,
        hidden_size=16,
        vocab_size=32,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def get_config(self):
        return Idefics3Config(
            image_token_id=self.image_token_id,
            pad_token_id=self.pad_token_id,
            scale_factor=2,
            text_config={
                "model_type": "llama",
                "hidden_size": self.hidden_size,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": self.pad_token_id,
            },
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
            },
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        image_hidden_states = paddle.zeros([self.batch_size, 1, config.text_config.hidden_size], dtype="float32")
        return config, image_hidden_states

    def prepare_config_and_inputs_for_common(self):
        config, image_hidden_states = self.prepare_config_and_inputs()
        input_ids = paddle.to_tensor([[1, self.image_token_id, 5]], dtype="int64").expand([self.batch_size, -1])
        labels = paddle.to_tensor([[-100, -100, 5]], dtype="int64").expand([self.batch_size, -1])
        attention_mask = paddle.ones(input_ids.shape, dtype="int64")

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_hidden_states": image_hidden_states,
            "labels": labels,
        }
        return config, inputs_dict


class Idefics3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics3ForConditionalGeneration`.
    """

    base_model_class = Idefics3Model
    all_model_classes = (Idefics3Model, Idefics3ForConditionalGeneration)
    all_generative_model_classes = {Idefics3ForConditionalGeneration: {Idefics3Model, "idefics3"}}
    max_new_tokens = 3

    def setUp(self):
        self.model_tester = Idefics3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics3Config, has_text_modality=False)

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 2]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        if config is not None and config.image_token_id < config.text_config.vocab_size:
            logits_processor_kwargs["bad_words_ids"].append([config.image_token_id])
        return logits_processor_kwargs

    def _greedy_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        return model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **logits_processor_kwargs,
            **inputs_dict,
        )

    def _beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        return model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

    def _sample_generate(
        self,
        model,
        inputs_dict,
        num_return_sequences,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        paddle.seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        return model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **logits_processor_kwargs,
            **inputs_dict,
        )

    def prepare_config_and_inputs_for_generate(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {k: v[:batch_size, ...] if isinstance(v, paddle.Tensor) else v for k, v in inputs_dict.items()}
        inputs_dict.pop("labels", None)
        config.text_config.eos_token_id = None
        config.text_config.forced_eos_token_id = None
        return config, inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_config_dict = config.to_dict()
        base_config = Idefics3Config(**base_config_dict)

        self.assertEqual(base_config.vocab_size, base_config.text_config.vocab_size)
        base_config.vocab_size = 55
        self.assertEqual(base_config.vocab_size, 55)
        self.assertEqual(base_config.text_config.vocab_size, 55)

    def test_model_forward_with_precomputed_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Idefics3Model(config)
        model.eval()

        with paddle.no_grad():
            outputs = model(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                image_hidden_states=inputs_dict["image_hidden_states"],
            )

        self.assertEqual(outputs.last_hidden_state.shape, [self.model_tester.batch_size, 3, config.text_config.hidden_size])
        self.assertEqual(outputs.image_hidden_states.shape, [self.model_tester.batch_size, 1, config.text_config.hidden_size])

    def test_conditional_generation_forward_with_labels(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Idefics3ForConditionalGeneration(config)
        model.eval()

        with paddle.no_grad():
            outputs = model(**inputs_dict)

        self.assertEqual(outputs.logits.shape, [self.model_tester.batch_size, 3, config.text_config.vocab_size])
        self.assertIsNotNone(outputs.loss)

    def test_image_token_count_mismatch_raises(self):
        config = self.model_tester.get_config()
        model = Idefics3Model(config)

        input_ids = paddle.to_tensor([[1, self.model_tester.image_token_id, self.model_tester.image_token_id]], dtype="int64")
        image_hidden_states = paddle.zeros([1, 1, config.text_config.hidden_size], dtype="float32")

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, image_hidden_states=image_hidden_states)


@unittest.skip("Idefics3 tiny checkpoint is not available yet.")
class Idefics3IntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = Idefics3ForConditionalGeneration
    hf_remote_test_model_path = "PaddleFormers/tiny-random-idefics3"
    paddlehub_remote_test_model_path = "PaddleFormers/tiny-random-idefics3"

    def test_model_from_pretrained_hf_hub(self):
        super().test_model_from_pretrained_hf_hub()

    def test_model_from_pretrained_paddle_hub(self):
        super().test_model_from_pretrained_paddle_hub()

    def test_model_from_config_paddle_hub(self):
        super().test_model_from_config_paddle_hub()

    def test_model_from_pretrained_with_cache_dir(self):
        super().test_model_from_pretrained_with_cache_dir()

    def test_pretrained_save_and_load(self):
        super().test_pretrained_save_and_load()


if __name__ == "__main__":
    unittest.main()
