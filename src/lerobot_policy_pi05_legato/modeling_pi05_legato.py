"""
LEGATO variant of PI05.

LEGATO layers a *guided flow* on top of the pi05 action expert: the first
`warmup` positions of the action chunk are anchored to the previously executed
chunk (weight w=0) and smoothly handed over to freshly generated actions
(w=1), which gives temporally consistent, "legato" chunk-to-chunk transitions.

Class map (installed lerobot 0.6.1, `lerobot.policies.pi05.modeling_pi05`):
    PI05Policy                    -> LeRobot policy wrapper; holds .model
      PI05Pytorch                 -> flow-matching module; holds embed_suffix /
                                     forward (loss) / sample_actions / denoise_step
        PaliGemmaWithExpertModel  -> inner VLM + action-expert transformer
                                     (only a transformer `forward`)

The methods LEGATO changes (embed_suffix, forward, sample_actions,
denoise_step) live on PI05Pytorch, so the LEGATO child subclasses PI05Pytorch
(NOT PaliGemmaWithExpertModel). PI05LegatoPolicy just swaps in this model.

NOTE — flow-time convention differs from Legato-kinetix. Installed pi05 uses
    x_t = t * noise + (1 - t) * actions,  u_t = noise - actions,
    inference t: 1 -> 0 with dt = -1/num_steps,  x_t += dt * v_t
whereas Legato-kinetix uses t: 0 (noise) -> 1 (data). The reference LEGATO
formulas (kappa, guided_x_t, u_t) must be re-derived for pi05's convention
before implementing — do not copy the signs verbatim.

Everything below is a TEMPLATE: each override documents the intended LEGATO
change, keeps the parent signature, and currently delegates to super() (so the
package imports and behaves as vanilla pi05) with `# TODO(legato)` insertion
points. Nothing LEGATO-specific is implemented yet.
"""

import torch
import torch.nn.functional as F

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    clone_past_key_values,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.utils.constants import ACTION

from .configuration_pi05_legato import PI05LegatoConfig
from . import utils


# =============================================================================
# Weight schedule helpers (config-aware wrappers over utils; see utils.py for
# the PyTorch port of Legato-kinetix build_weight_curve / sample_* family)
# =============================================================================
def sample_warmup(
    config: PI05LegatoConfig,
    bsize: int,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a per-example warmup length (B,) per config.warmup_sampling."""
    return utils.sample_length(
        config.warmup_min,
        config.warmup_max,
        (bsize,),
        config.warmup_sampling,
        device=device,
        generator=generator,
    )


def sample_ramp(
    config: PI05LegatoConfig,
    bsize: int,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a per-example ramp length (B,) per config.ramp_sampling."""
    return utils.sample_length(
        config.ramp_min,
        config.ramp_max,
        (bsize,),
        config.ramp_sampling,
        device=device,
        generator=generator,
    )


def build_weight_curve(
    config: PI05LegatoConfig,
    warmup: torch.Tensor | int,
    ramp: torch.Tensor | int = 1,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Per-position weight curve for given `warmup`/`ramp` using config.weight_shape
    (scalar -> (chunk,), or (B,) -> (B, chunk)).
    """
    return utils.build_weight_curve(
        config.chunk_size, warmup, ramp, config.weight_shape, device=device, dtype=dtype
    )


def sample_weight_curve(
    config: PI05LegatoConfig,
    bsize: int,
    device=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample per-example warmup + ramp and build the weight curve.
    Returns (w (B, chunk), warmup (B,), ramp (B,)).
    """
    warmup = sample_warmup(config, bsize, device=device, generator=generator)
    ramp = sample_ramp(config, bsize, device=device, generator=generator)
    w = build_weight_curve(config, warmup, ramp, device=device, dtype=torch.float32)
    return w, warmup, ramp


# =============================================================================
# LEGATO flow-matching model
# =============================================================================
class PI05PytorchLEGATO(PI05Pytorch):
    """
    PI05Pytorch with the LEGATO guided flow and per-position weight schedule.

    Recommended conditioning (see design note): concatenate the per-position
    weight `w` as an extra input channel to the action tokens (grow
    action_in_proj by +1, zero-init the new column so pretrained pi05 loads as
    a no-op). Keep the flow timestep in adaRMS as pi05 already does.
    """

    def __init__(self, config: PI05LegatoConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        # TODO(legato): grow the action input projection to carry the weight
        # channel and zero-init the extra column for checkpoint compatibility:
        #
        #   old = self.action_in_proj
        #   new = nn.Linear(config.max_action_dim + 1, old.out_features)
        #   with torch.no_grad():
        #       new.weight[:, :config.max_action_dim].copy_(old.weight)
        #       new.weight[:, config.max_action_dim:].zero_()
        #       new.bias.copy_(old.bias)
        #   self.action_in_proj = new

    # -- embedding ------------------------------------------------------------
    def embed_suffix(self, noisy_actions, timestep, weight=None):
        """
        pi05.embed_suffix + LEGATO weight conditioning.

        Each per-position weight w[i] is written into the first *padding* slot
        of the action vector (index original_action_dim). pi05 pads actions to
        max_action_dim and ignores dims [original_action_dim:] in the loss, so
        this conditions every action token on its schedule value with no change
        to action_in_proj and full checkpoint compatibility.

        Args:
            noisy_actions: (B, H, max_action_dim)  -- H == chunk_size
            weight:        (B, H, 1) in [0, 1]; None -> all-fresh (w=1), i.e.
                           unguided vanilla pi05 behavior.
        """
        bsize, horizon = noisy_actions.shape[:2]
        original_action_dim = self.config.output_features[ACTION].shape[0]

        # Unguided default: w = 1 everywhere (fully fresh, no anchoring).
        if weight is None:
            weight = noisy_actions.new_ones(bsize, horizon, 1)

        assert noisy_actions.shape == (bsize, horizon, self.config.max_action_dim), (
            f"noisy_actions must be (B, H, max_action_dim={self.config.max_action_dim}), "
            f"got {tuple(noisy_actions.shape)}"
        )
        assert weight.shape == (bsize, horizon, 1), (
            f"weight must be (B, H, 1)=({bsize}, {horizon}, 1), got {tuple(weight.shape)}"
        )
        assert original_action_dim < self.config.max_action_dim, (
            f"no free padding slot for the weight channel: original_action_dim="
            f"{original_action_dim} must be < max_action_dim={self.config.max_action_dim}"
        )

        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # --- LEGATO: inject the weight into the first padding slot ---
        model_actions = noisy_actions.clone()
        model_actions[..., original_action_dim : original_action_dim + 1] = weight.to(
            model_actions.dtype
        )

        # Fuse timestep + action information using an MLP
        def action_proj_func(a):
            return self.action_in_proj(a)

        action_emb = self._apply_checkpoint(action_proj_func, model_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2] # (B, H, D)
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    # -- training loss --------------------------------------------------------
    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> torch.Tensor:
        """Do a full training forward pass and compute the LEGATO guided-flow loss.

        Copied from PI05Pytorch.forward; LEGATO changes are marked `# Note (LEGATO)`.
        In training the anchor for low-weight positions is the ground-truth
        `actions` itself (native continuation), so no separate previous chunk
        is needed here — only inference uses the real previous chunk.
        """
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Note (LEGATO): sample a per-example weight curve (warmup + ramp).
        bsize = actions.shape[0]
        w, _, _ = sample_weight_curve(self.config, bsize, device=actions.device)  # (B, chunk)
        w = w.unsqueeze(-1)  # (B, chunk, 1)
        # Note (LEGATO): anchor dropout. With prob `fresh_prob`, force the whole
        # example fully fresh (w == 1) so the model also learns unconditional
        # generation for the cold-start first chunk. Otherwise warmup >= warmup_min
        # means the leading positions are never supervised at w == 1. w==1 makes
        # kappa=0 (u_t unchanged) and x_t stays the plain noisy sample, i.e. a
        # vanilla pi05 flow example (still tagged w=1 in the weight channel).
        if self.config.fresh_prob > 0.0:
            fresh = torch.rand(bsize, device=actions.device) < self.config.fresh_prob  # (B,)
            w = torch.where(fresh[:, None, None], torch.ones_like(w), w)
        # Note (LEGATO): dt from num_inference_steps keeps kappa consistent with
        # the number of Euler steps actually run at inference.
        dt = 1.0 / self.config.num_inference_steps
        kappa = (1.0 - w) / dt
        # Note (LEGATO): pi05 base target is (noise - actions); the guided target
        # scales it by (1 - kappa * t), where pi05 time t == reference (1 - t_ref).
        # NOTE: this is (1 - kappa*t), NOT the reference's (1 + kappa*(1-t_ref)).
        # The Legato-kinetix reference has a sign error that is masked in its
        # hard-step sim (only w in {0,1} occur, and only the fresh w=1 / kappa=0
        # positions are executed). For a ramp (0 < w < 1) the wrong sign makes
        # the guided Euler integration overshoot `data`; the minus sign lands on
        # `data` exactly for every w (verified by simulating the discrete scheme).
        u_t = u_t * (1.0 - kappa * time_expanded)
        # Note (LEGATO): anchor low-weight positions to the data, generate the rest.
        x_t = (1.0 - w) * actions + w * x_t

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        # Note (LEGATO): pass the guided state and weight schedule into embed_suffix.
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time, weight=w)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs) -> torch.Tensor:
        """Do a full inference forward with the LEGATO guided flow.

        Copied from PI05Pytorch.sample_actions; LEGATO changes are marked
        `# Note (LEGATO)`. The RTC branch is intentionally dropped — LEGATO
        (native continuation) is the alternative to RTC.

        LEGATO inputs (via kwargs, all optional):
            prev_action_chunk: (B, chunk, max_action_dim) previously executed
                               chunk (already padded); None -> no anchoring.
            warmup:            # of already-executed positions to anchor (int).
            ramp:              hand-over length (int; default config.ramp_min).
        With no prev_action_chunk (or warmup=0) the weight is all-ones and this
        reduces exactly to vanilla pi05 sampling.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        # Note (LEGATO): build the fixed inference weight schedule.
        prev_action_chunk = kwargs.get("prev_action_chunk")
        warmup = int(kwargs.get("warmup", 0))
        ramp = int(kwargs.get("ramp", self.config.ramp_min))
        if prev_action_chunk is None:
            warmup = 0  # nothing to anchor to -> fully fresh (w == 1)
            prev_action_chunk = torch.zeros_like(noise)
        w = build_weight_curve(self.config, warmup, ramp, device=device)  # (chunk,)
        w = w.view(1, -1, 1).expand(bsize, -1, 1).contiguous()  # (B, chunk, 1)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            # Note (LEGATO): re-anchor every step -> guided_x_t = (1-w)*prev + w*x_t,
            # then integrate FROM the guided state (guided_x_t + dt*v_t), so the
            # anchoring does not leak away between steps.
            guided_x_t = (1.0 - w) * prev_action_chunk + w * x_t
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=guided_x_t,
                timestep=time_tensor,
                weight=w,  # Note (LEGATO): thread weight into embed_suffix
            )
            x_t = guided_x_t + dt * v_t

        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, weight=None):
        """Apply one denoising step at a given timestep.

        Copied from PI05Pytorch.denoise_step; the only LEGATO change is passing
        `weight` into embed_suffix (marked `# Note (LEGATO)`). The guided blend
        of x_t is done by the caller (sample_actions) so the integrator can step
        from the guided state.
        """
        # Note (LEGATO): pass the weight schedule into embed_suffix.
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep, weight=weight
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


# =============================================================================
# LEGATO policy wrapper
# =============================================================================
class PI05LegatoPolicy(PI05Policy):
    """PI05 policy whose flow model is the LEGATO guided-flow variant."""

    config_class = PI05LegatoConfig
    name = "pi05_legato"

    def __init__(self, config: PI05LegatoConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Swap the vanilla flow model for the LEGATO one, preserving the rtc
        # processor the parent already built. State-dict keys are unchanged
        # (until the action_in_proj surgery lands), so pi05 checkpoints load.
        self.model = PI05PytorchLEGATO(config, rtc_processor=self.rtc_processor)
        self.model.to(config.device)
        # TODO(legato): if inference needs a rolling previous-chunk buffer,
        # initialize it here and clear it in reset().

    def reset(self):
        super().reset()
        # TODO(legato): reset the previous-action-chunk buffer, if any.
