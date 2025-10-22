"""Simple test script for ROSA implementation."""

import torch

from decoder_pytorch import ROSA, model_summary


def test_rosa_forward():
    """Test basic forward pass and generation."""
    print("Testing ROSA implementation...")

    # Create small ROSA model
    model = ROSA(
        num_tokens=256,
        dim=128,
        depth=4,
        rosa_state_cap=8192,
        k_candidates=1,
        channels=1,
        max_seq_len=128,
    )

    print("\nModel summary:")
    model_summary(model, max_depth=2, show_param_shapes=False)

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 256, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Test logits output
    logits = model(input_ids, return_loss=False)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 256), f"Expected (2, 32, 256), got {logits.shape}"

    # Test loss computation
    loss = model(input_ids, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is inf"

    # Test backward pass
    loss.backward()
    print("Backward pass: OK")

    # Test generation
    prompt = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=32,
            temperature=1.0,
            min_p=0.1,
        )
    print(f"\nPrompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    assert generated.shape[0] == 1, "Batch size should be 1"
    assert generated.shape[1] == 32, f"Should generate 32 tokens, got {generated.shape[1]}"

    print("\n✓ All tests passed!")


def test_rosa_multi_channel():
    """Test multi-channel ROSA."""
    print("\n" + "=" * 50)
    print("Testing Multi-channel ROSA...")

    model = ROSA(
        num_tokens=256,
        dim=128,
        depth=4,
        rosa_state_cap=8192,
        channels=2,  # Multi-channel
        max_seq_len=128,
    )

    print("\nModel summary:")
    model_summary(model, max_depth=2, show_param_shapes=False)

    # Test forward pass
    input_ids = torch.randint(0, 256, (2, 32))
    logits = model(input_ids, return_loss=False)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 32, 256)

    loss = model(input_ids, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()

    print("\n✓ Multi-channel tests passed!")


def test_rosa_soft_mixing():
    """Test soft pointer mixing."""
    print("\n" + "=" * 50)
    print("Testing Soft Pointer Mixing...")

    model = ROSA(
        num_tokens=256,
        dim=128,
        depth=4,
        rosa_state_cap=8192,
        k_candidates=4,  # Soft mixing
        temperature=1.0,
        max_seq_len=128,
    )

    print("\nModel summary:")
    model_summary(model, max_depth=2, show_param_shapes=False)

    # Test forward pass
    input_ids = torch.randint(0, 256, (2, 32))
    logits = model(input_ids, return_loss=False)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 32, 256)

    loss = model(input_ids, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()

    # Test generation
    prompt = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        generated = model.generate(prompt, max_length=32)
    print(f"Generated shape: {generated.shape}")

    print("\n✓ Soft mixing tests passed!")


if __name__ == "__main__":
    test_rosa_forward()
    test_rosa_multi_channel()
    test_rosa_soft_mixing()
    print("\n" + "=" * 50)
    print("All ROSA tests passed successfully! ✓")
    print("=" * 50)
