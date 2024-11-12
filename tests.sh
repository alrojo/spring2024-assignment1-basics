#!/bin/bash

# Part 2
pytest tests/test_train_bpe.py --disable-warnings
pytest tests/test_tokenizer.py --disable-warnings

# Part 3
pytest -k test_rmsnorm --disable-warnings
pytest -k test_gelu --disable-warnings
pytest -k test_positionwise_feedforward --disable-warnings
pytest -k test_softmax_matches_pytorch --disable-warnings
pytest -k test_scaled_dot_product_attention --disable-warnings
pytest -k test_4d_scaled_dot_product_attention --disable-warnings
pytest -k test_multihead_self_attention --disable-warnings
pytest -k test_transformer_block --disable-warnings
pytest -k test_transformer_lm --disable-warnings

# Part 4
pytest -k test_cross_entropy --disable-warnings
pytest -k test_adamw --disable-warnings
pytest -k test_get_lr_cosine_schedule --disable-warnings
pytest -k test_gradient_clipping --disable-warnings

# Part 5
pytest -k test_get_batch --disable-warnings
pytest -k test_checkpointing --disable-warnings
