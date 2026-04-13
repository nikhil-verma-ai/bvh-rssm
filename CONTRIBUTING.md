# Contributing to BVH-RSSM

## Setup

```bash
git clone https://github.com/nikhil-verma-ai/bvh-rssm.git
cd bvh-rssm
pip install -e ".[dev,training]"
make test
```

`make test` runs the full unit test suite (`pytest tests/`). All 271+ tests should pass.

## Adding an Environment

All environments subclass `ShiftWrapper` from `bvh_rssm/envs/wrappers.py`.

**Minimum interface:**

```python
from bvh_rssm.envs.wrappers import ShiftWrapper
import gymnasium as gym

class MyEnv(ShiftWrapper):
    def __init__(self, seed: int = 0):
        base_env = gym.make("SomeEnv-v4")
        super().__init__(base_env, shift_rate=0.01, shift_type="abrupt", seed=seed)

    def _apply_shift(self, progress: float = 1.0) -> None:
        """Called by ShiftWrapper when a shift event fires. Modify env dynamics here."""
        self.unwrapped.some_param = new_value

    def _is_interventionist(self, action) -> bool:
        """Return True if this action constitutes a causal intervention."""
        return False  # most envs return False

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["oracle_tau"] = self._compute_oracle_tau()  # REQUIRED
        return obs, reward, terminated, truncated, info

    def _compute_oracle_tau(self) -> int:
        """Return steps until next shift. Must be >= 0."""
        ...
```

**Registering the environment:**

Add to `bvh_rssm/envs/__init__.py`:

```python
from bvh_rssm.envs.my_env import MyEnv
```

**Required unit tests** (create `tests/unit/test_my_env.py`):

```python
import numpy as np
from bvh_rssm.envs.my_env import MyEnv

def test_oracle_tau_in_info():
    env = MyEnv(seed=0)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "oracle_tau" in info
    assert isinstance(info["oracle_tau"], int)
    assert info["oracle_tau"] >= 0

def test_shift_changes_dynamics():
    env = MyEnv(seed=0)
    env.reset()
    env._apply_shift()
    # assert that some env parameter changed
```

## Running Validation

```bash
# Fast smoke test (~10 seconds)
python3 scripts/train.py --fast

# Full validation — skip P1 if checkpoint exists (~25 minutes)
python scripts/validate.py --skip-p1

# Full validation — save P1 checkpoint for future --skip-p1 runs (~2 hours)
python scripts/validate.py --save-p1 checkpoints/my_run.pt
```

Results write to `validation_report.json`. To reproduce reference results:
```bash
python scripts/validate.py --skip-p1
# Checkpoint at checkpoints/p1_joint.pt (100k P1 steps, joint τ training, seed=42)
```

## Running Tests

```bash
make test              # all tests
make test-unit         # unit tests only (no MuJoCo required)
make test-env          # environment integration tests (requires MuJoCo)
```

CI runs `make test-unit` only (no MuJoCo in CI).

## Type Checking

```bash
python -m pyright
```

Strict mode is enforced on `bvh_rssm/networks/`, `bvh_rssm/training/losses.py`,
`bvh_rssm/training/replay_buffer.py`, and `bvh_rssm/utils/`. New code in these modules
must pass pyright strict with zero errors.

## Commit Conventions

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature or environment |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Test additions or fixes |
| `perf:` | Performance improvement |
| `refactor:` | Code restructure, no behaviour change |

Examples:
```
feat: add SensorDrift environment with deterministic oracle tau
fix: correct free-bits clamping in kl_loss
docs: update ARCHITECTURE.md with Phase 3 details
test: add unit tests for HazardHead competing risks
```
