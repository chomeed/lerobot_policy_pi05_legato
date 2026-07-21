from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_legato")
@dataclass
class PI05LegatoConfig(PI05Config):
    """
    Configuration class for the PI05Legato policy.

    Inherits from PI05Config and adds LEGATO-specific parameters that control
    the per-chunk-position weight schedule (warmup mask) and the guided flow.

    See `Legato-kinetix/src/model_legato.py: ModelConfig` for the reference
    values these mirror.
    """

    # --- LEGATO weight-schedule -----------------------------------------------
    # Per example a `warmup` and a `ramp` length are sampled. The per-position
    # weight is 0 (anchored to the previous action chunk) for the first
    # `warmup` positions, rises over the next `ramp` positions, then is 1
    # (freshly generated). At inference `warmup` == executed steps of the
    # previous chunk.
    warmup_min: int = 5
    warmup_max: int = 8
    warmup_sampling: str = "bell"  # "bell" | "exp" | "uniform"

    ramp_min: int = 4
    ramp_max: int = 8
    ramp_sampling: str = "uniform"  # "bell" | "exp" | "uniform"

    # Ramp shape from anchored (0) to fresh (1). "cosine" is the smooth ease
    # recommended for real-robot deployment; "step" reproduces the Kinetix sim.
    weight_shape: str = "cosine"  # "step" | "linear" | "cosine"

    # NOTE: kappa = (1 - w) / dt in the guided loss uses dt = 1 /
    # num_inference_steps (inherited from PI05Config), so it stays consistent
    # with the number of Euler steps actually run at inference — no separate
    # flow-step field.

    # TODO(legato): any additional LEGATO hyperparameters go here.

    def __post_init__(self):
        super().__post_init__()

        for name in ("warmup_sampling", "ramp_sampling"):
            val = getattr(self, name)
            if val not in ("bell", "exp", "uniform"):
                raise ValueError(f"Invalid {name}: {val!r} (expected bell|exp|uniform)")
        if self.weight_shape not in ("step", "linear", "cosine"):
            raise ValueError(f"Invalid weight_shape: {self.weight_shape!r} (expected step|linear|cosine)")
        if not (0 <= self.warmup_min <= self.warmup_max):
            raise ValueError(f"Require 0 <= warmup_min <= warmup_max, got {self.warmup_min}, {self.warmup_max}")
        if not (1 <= self.ramp_min <= self.ramp_max):
            raise ValueError(f"Require 1 <= ramp_min <= ramp_max, got {self.ramp_min}, {self.ramp_max}")
        # The weight curve must reach w=1 (fresh) inside the chunk, else there
        # are no freshly generated positions to supervise.
        if self.warmup_max + self.ramp_max >= self.chunk_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must exceed warmup_max + ramp_max "
                f"({self.warmup_max} + {self.ramp_max} = {self.warmup_max + self.ramp_max}); "
                "otherwise the weight curve never reaches 1 (no fresh positions)."
            )
