import pytest
import torch
from bvh_rssm.utils.rng import RNGStateStore, restore_rng_states


class TestRNGStateStore:
    def test_capture_returns_bytes_tensor(self):
        store = RNGStateStore()
        state = store.capture()
        assert isinstance(state, torch.ByteTensor)

    def test_len_increments_on_capture(self):
        store = RNGStateStore()
        assert len(store) == 0
        store.capture()
        assert len(store) == 1
        store.capture()
        assert len(store) == 2

    def test_restore_reproduces_same_samples(self):
        """After restoring RNG state, torch.randn produces identical output."""
        store = RNGStateStore()
        store.capture()  # index 0

        # Generate reference samples at this RNG state
        store.restore(0)
        ref = torch.randn(10)

        # Generate again after restoring — must match
        store.restore(0)
        again = torch.randn(10)

        assert torch.equal(ref, again)

    def test_clear_resets_length(self):
        store = RNGStateStore()
        store.capture()
        store.capture()
        store.clear()
        assert len(store) == 0

    def test_different_states_produce_different_samples(self):
        """Two separately captured states should produce different samples."""
        store = RNGStateStore()
        store.capture()     # state 0
        torch.randn(1)      # advance RNG
        store.capture()     # state 1

        store.restore(0)
        sample_0 = torch.randn(10)
        store.restore(1)
        sample_1 = torch.randn(10)

        assert not torch.equal(sample_0, sample_1)


class TestRestoreRngStates:
    def test_context_manager_restores_original_state(self):
        """After context exits, original RNG state is restored."""
        original_state = torch.get_rng_state()
        torch.randn(5)  # advance RNG
        pre_context_state = torch.get_rng_state()

        states = [original_state]
        with restore_rng_states(states):
            torch.randn(100)  # advance RNG inside context

        # After context, RNG should be at pre_context_state, not original_state
        post_context_state = torch.get_rng_state()
        assert torch.equal(post_context_state, pre_context_state)

    def test_context_manager_yields_states(self):
        """Context manager yields the states list for use inside the block."""
        states = [torch.get_rng_state()]
        with restore_rng_states(states) as s:
            assert s is states

    def test_counterfactual_replay_pattern(self):
        """Core use case: replay with same z_t samples but different action."""
        store = RNGStateStore()

        # Simulate factual trajectory: store RNG state, then draw z_t
        store.capture()
        z_factual = torch.randn(8)  # z_t for factual

        # Simulate counterfactual: restore RNG state, draw z_t again
        store.restore(0)
        z_counterfactual = torch.randn(8)

        assert torch.equal(z_factual, z_counterfactual)
