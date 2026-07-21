#!/usr/bin/env python
"""Validate the LEGATO ``sample_actions`` reimplementation on real data.

Idea
----
Take a trained checkpoint, run the policy's full inference path
(``predict_action_chunk`` -> ``PI05PytorchLEGATO.sample_actions``, i.e. the whole
flow-matching denoising loop) on a handful of episodes of a dataset, and report
the MSE between the fully-denoised predicted action chunk and the ground-truth
action chunk. This is a sanity check on the reimplemented sampler: a correctly
trained policy should reconstruct the demonstrated actions with low MSE.

Optionally the same is done for the un-finetuned ``lerobot/pi05_base`` weights
loaded into the *identical* LEGATO architecture, as a reference point (only the
weights differ, so the MSE gap isolates the effect of finetuning).

What space is the MSE in?
-------------------------
The primary metric is computed in the model's own (relative, if
``use_relative_actions``, then normalized) action space -- exactly the space the
flow operates in, so it is the most direct check of the sampler. A secondary
metric in the raw/absolute action space (via the post-processor) is also
reported for interpretability.

Two modes (``--mode``)
----------------------
fresh (default)
    No previous action chunk -> ``sample_actions`` should reduce to vanilla pi05
    (weight ``w == 1`` everywhere). NOTE: passing nothing does *not* achieve this
    -- ``warmup`` is forced to 0 but ``ramp`` defaults to ``config.ramp_min``,
    which makes ``build_weight_curve`` emit a leading ramp (e.g. w[0] = 0.5 for
    ramp=2), scaling the first action position toward the (zero) anchor. Passing
    ``warmup=0, ramp=1`` provably yields ``w == 1`` everywhere, so fresh mode uses
    those. This validates the core denoiser.

continuation (LEGATO guided flow)
    Teacher-forced: the previous "executed" chunk is the ground-truth action
    chunk itself (this is exactly the native-continuation anchor LEGATO trains
    on -- on a single demo trajectory the aligned previous chunk equals the
    current chunk on the overlap). We feed it as ``prev_action_chunk`` with a
    ``warmup`` / ``ramp`` schedule, so the first ``warmup`` positions are anchored
    to GT (w=0), then handed over across the ramp (0<w<1), then freshly generated
    (w=1). MSE is reported per region:
      * anchored (w=0):  should be ~0 -- the guided blend must land on the anchor.
      * ramp (0<w<1):    should be ~0 too -- validates the guided velocity
                         correction ``u_t*(1 - kappa*t)`` rides the flow onto the
                         data for every w (the disputed sign in the README).
      * fresh (w=1):     ordinary generation error (no prev influence there).
    This is teacher-forced (GT prefix), NOT a closed-loop rollout: it validates
    the guided-flow math that exists. The deployment rolling-buffer that anchors
    to the model's *own* previous chunk (in a different relative frame) is still a
    TODO in ``PI05LegatoPolicy``.

No GPU is required by design of the code, but flow inference is heavy -- run this
on a GPU box (``--device cuda``).

Examples
--------
    # validate the core sampler (fresh path) + baseline reference
    python scripts/validate_sample_actions.py \
        --policy-path /path/to/.../checkpoints/050000/pretrained_model \
        --dataset-repo-id chomeed/board_insertion_..._k30_relative_action \
        --n-episodes 10 --device cuda --compare-base

    # validate the LEGATO continuation (guided flow), warmup=6 ramp=4
    python scripts/validate_sample_actions.py \
        --policy-path /path/to/.../checkpoints/050000/pretrained_model \
        --dataset-repo-id chomeed/board_insertion_..._k30_relative_action \
        --n-episodes 10 --device cuda --mode continuation --warmup 6 --ramp 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

# Allow running straight from the repo without `pip install -e .` (src layout).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

from lerobot_policy_pi05_legato.configuration_pi05_legato import PI05LegatoConfig  # noqa: E402
from lerobot_policy_pi05_legato.modeling_pi05_legato import (  # noqa: E402
    PI05LegatoPolicy,
    build_weight_curve,
)
from lerobot_policy_pi05_legato.processor_pi05_legato import (  # noqa: E402
    make_pi05_legato_pre_post_processors,
)

BASE_REPO_ID = "lerobot/pi05_base"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset(repo_id: str, chunk_size: int, n_episodes: int) -> LeRobotDataset:
    """Load the first ``n_episodes`` episodes with an ``action`` chunk of length
    ``chunk_size`` (delta_timestamps), matching how training samples action chunks."""
    # Peek at fps to build the action delta_timestamps.
    meta = LeRobotDataset(repo_id, episodes=[0]).meta
    fps = meta.fps
    delta_timestamps = {ACTION: [i / fps for i in range(chunk_size)]}
    episodes = list(range(min(n_episodes, meta.total_episodes)))
    logging.info(
        f"Loading dataset '{repo_id}': episodes {episodes[0]}..{episodes[-1]} "
        f"({len(episodes)} eps), fps={fps}, action chunk={chunk_size}."
    )
    return LeRobotDataset(repo_id, episodes=episodes, delta_timestamps=delta_timestamps)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_finetuned(policy_path: str, device: str, num_steps: int | None) -> PI05LegatoPolicy:
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device
    if num_steps is not None:
        cfg.num_inference_steps = num_steps
    if not isinstance(cfg, PI05LegatoConfig):
        raise TypeError(
            f"Expected a pi05_legato checkpoint, got config type {type(cfg).__name__}. "
            "Point --policy-path at a LEGATO checkpoint's pretrained_model dir."
        )
    logging.info(f"Loading finetuned LEGATO policy from '{policy_path}' (num_steps={cfg.num_inference_steps}).")
    policy = PI05LegatoPolicy.from_pretrained(policy_path, config=cfg)
    return policy.to(device).eval()


def load_baseline(reference_cfg: PI05LegatoConfig, device: str) -> PI05LegatoPolicy:
    """Load un-finetuned pi05_base weights into an *identical* LEGATO architecture.

    Normalization lives in the processor (not the model) and the action/state
    projections are padded to ``max_action_dim`` independent of the dataset, so
    the base pi05 state_dict loads into the LEGATO model unchanged (LEGATO adds
    no new parameters). Only the weights differ from the finetuned run.
    """
    cfg = deepcopy(reference_cfg)
    cfg.pretrained_path = BASE_REPO_ID
    cfg.device = device
    logging.info(f"Loading baseline (no finetuning) weights '{BASE_REPO_ID}' into the LEGATO architecture.")
    # strict=False: tolerate incidental buffer/key differences between the base
    # checkpoint and the LEGATO wrapper; missing/unexpected keys are logged.
    policy = PI05LegatoPolicy.from_pretrained(BASE_REPO_ID, config=cfg, strict=False)
    return policy.to(device).eval()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    policy: PI05LegatoPolicy,
    dataset: LeRobotDataset,
    dataset_stats: dict,
    device: str,
    batch_size: int,
    num_workers: int,
    mode: str,
    warmup: int,
    ramp: int,
) -> dict:
    """Run full-denoising sample_actions over the dataset and accumulate action MSE.

    mode="fresh":        no anchor, w==1 everywhere (validates the core denoiser).
    mode="continuation": teacher-forced -- the ground-truth chunk is fed as the
                         LEGATO anchor with the given warmup/ramp schedule; MSE is
                         additionally broken down by weight region.

    Returns a dict with overall and per-episode MSE (normalized + raw space), and
    for continuation mode a per-region breakdown.
    """
    config = policy.config
    preprocessor, postprocessor = make_pi05_legato_pre_post_processors(config, dataset_stats=dataset_stats)
    original_action_dim = config.output_features[ACTION].shape[0]
    max_action_dim = config.max_action_dim
    pad_key = f"{ACTION}_is_pad"

    # Per-position weight schedule -> region membership (continuation mode only).
    region_pos: dict[str, torch.Tensor] = {}
    if mode == "continuation":
        w_curve = build_weight_curve(config, warmup, ramp).view(-1).cpu()  # (T,)
        region_pos = {
            "anchored (w=0)": (w_curve == 0),
            "ramp (0<w<1)": (w_curve > 0) & (w_curve < 1),
            "fresh (w=1)": (w_curve == 1),
        }
        logging.info(
            f"  continuation schedule: warmup={warmup}, ramp={ramp} -> "
            + ", ".join(f"{k}: {int(v.sum())} pos" for k, v in region_pos.items())
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # Accumulators keyed by episode index (-1 == aggregate over all).
    norm_se: dict[int, float] = {}
    raw_se: dict[int, float] = {}
    count: dict[int, int] = {}
    # Per-region accumulators (continuation mode): region -> [sum_se_norm, count].
    region_se: dict[str, float] = {k: 0.0 for k in region_pos}
    region_cnt: dict[str, int] = {k: 0 for k in region_pos}

    n_batches = len(loader)
    for bi, batch in enumerate(loader):
        # Keep a raw copy of the ground-truth action chunk (pre-normalization) and
        # the pad mask + episode ids before the preprocessor mutates/wraps the batch.
        raw_action = batch[ACTION][..., :original_action_dim].clone().float()  # (B, T, Da)
        pad = batch[pad_key].clone()  # (B, T) bool, True == padded (invalid)
        ep_ids = batch["episode_index"].clone()

        # Match the training-time image dtype handling (uint8 -> float in [0, 1]).
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

        batch = preprocessor(batch)
        target_norm = batch[ACTION][..., :original_action_dim].to(device).float()  # (B, T, Da)

        if mode == "fresh":
            # w == 1 everywhere: reduces to vanilla pi05 sampling.
            kwargs = {"warmup": 0, "ramp": 1}
        else:
            # Teacher-forced continuation: anchor = the (normalized) GT chunk itself,
            # padded to max_action_dim (padding slots are unused / overwritten by the
            # weight channel in embed_suffix). Shape must match the noise: (B, T, Dmax).
            prev = torch.zeros(*target_norm.shape[:2], max_action_dim, device=device, dtype=torch.float32)
            prev[..., :original_action_dim] = target_norm
            kwargs = {"prev_action_chunk": prev, "warmup": warmup, "ramp": ramp}

        # Fully-denoised predicted chunk.
        pred = policy.predict_action_chunk(batch, **kwargs)  # (B, T, Da), model space

        valid = (~pad).to(device).unsqueeze(-1).float()  # (B, T, 1)

        # --- normalized (model) space ---
        se_norm = ((pred - target_norm) ** 2) * valid  # (B, T, Da)

        # --- raw / absolute space (invert normalization + relative) ---
        pred_raw = postprocessor(pred.clone())  # -> cpu, unnormalized, absolute
        pred_raw = pred_raw[..., :original_action_dim].float()
        se_raw = ((pred_raw - raw_action) ** 2) * valid.cpu()  # (B, T, Da)

        n_valid = valid.expand(-1, -1, original_action_dim)  # (B, T, Da)

        # Per-sample sums, distributed to their episode bucket.
        se_norm_s = se_norm.sum(dim=(1, 2)).cpu()
        se_raw_s = se_raw.sum(dim=(1, 2)).cpu()
        n_s = n_valid.sum(dim=(1, 2)).cpu()
        for ep, sn, sr, nn in zip(ep_ids.tolist(), se_norm_s.tolist(), se_raw_s.tolist(), n_s.tolist(), strict=True):
            norm_se[ep] = norm_se.get(ep, 0.0) + sn
            raw_se[ep] = raw_se.get(ep, 0.0) + sr
            count[ep] = count.get(ep, 0) + int(nn)
            norm_se[-1] = norm_se.get(-1, 0.0) + sn
            raw_se[-1] = raw_se.get(-1, 0.0) + sr
            count[-1] = count.get(-1, 0) + int(nn)

        # Per-region breakdown (normalized space), masking padded positions.
        for region, pos in region_pos.items():
            sel = pos.to(device)  # (T,) bool
            reg_se = se_norm[:, sel, :]  # (B, n_pos, Da)
            reg_n = n_valid[:, sel, :]
            region_se[region] += float(reg_se.sum().cpu())
            region_cnt[region] += int(reg_n.sum().cpu())

        if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
            running = norm_se[-1] / max(count[-1], 1)
            logging.info(f"  batch {bi + 1}/{n_batches}  running normalized MSE={running:.6f}")

    per_episode = {
        ep: {
            "mse_norm": norm_se[ep] / max(count[ep], 1),
            "mse_raw": raw_se[ep] / max(count[ep], 1),
            "n_action_elems": count[ep],
        }
        for ep in sorted(k for k in count if k != -1)
    }
    result = {
        "overall": {
            "mse_norm": norm_se[-1] / max(count[-1], 1),
            "mse_raw": raw_se[-1] / max(count[-1], 1),
            "n_action_elems": count[-1],
        },
        "per_episode": per_episode,
    }
    if region_pos:
        result["per_region"] = {
            region: {
                "mse_norm": region_se[region] / max(region_cnt[region], 1),
                "n_action_elems": region_cnt[region],
            }
            for region in region_pos
        }
    return result


def _print_report(name: str, result: dict) -> None:
    ov = result["overall"]
    print(f"\n=== {name} ===")
    print(f"  overall  MSE(normalized)={ov['mse_norm']:.6f}   MSE(raw)={ov['mse_raw']:.6f}   "
          f"(n={ov['n_action_elems']} action elems)")
    if "per_region" in result:
        print("  per-region (normalized space):")
        for region, m in result["per_region"].items():
            print(f"    {region:<16} MSE(norm)={m['mse_norm']:.6f}   (n={m['n_action_elems']})")
    print("  per-episode:")
    for ep, m in result["per_episode"].items():
        print(f"    ep {ep:>4}:  MSE(norm)={m['mse_norm']:.6f}   MSE(raw)={m['mse_raw']:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Path to the trained LEGATO checkpoint (pretrained_model dir).")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id to validate on.")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to evaluate (default 10).")
    parser.add_argument("--device", default="cuda", help="torch device (default cuda).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Override the flow denoising steps (default: config value).")
    parser.add_argument("--mode", choices=["fresh", "continuation"], default="fresh",
                        help="fresh: w==1 core-denoiser check (default). continuation: teacher-forced LEGATO guided flow.")
    parser.add_argument("--warmup", type=int, default=None,
                        help="continuation warmup (# anchored positions). Default: config.warmup_min. Ignored in fresh mode.")
    parser.add_argument("--ramp", type=int, default=None,
                        help="continuation ramp (hand-over length). Default: config.ramp_min. Ignored in fresh mode.")
    parser.add_argument("--compare-base", action="store_true", help="Also evaluate the un-finetuned lerobot/pi05_base baseline.")
    args = parser.parse_args()

    init_logging()

    # ---- finetuned ----
    finetuned = load_finetuned(args.policy_path, args.device, args.num_inference_steps)
    dataset = build_dataset(args.dataset_repo_id, finetuned.config.chunk_size, args.n_episodes)
    dataset_stats = dataset.meta.stats

    # Resolve the schedule. Fresh mode is pinned to warmup=0, ramp=1 (provably w==1).
    if args.mode == "fresh":
        warmup, ramp = 0, 1
        if args.warmup not in (None, 0) or args.ramp not in (None, 1):
            logging.warning("--warmup/--ramp are ignored in fresh mode (pinned to warmup=0, ramp=1).")
    else:
        warmup = args.warmup if args.warmup is not None else finetuned.config.warmup_min
        ramp = args.ramp if args.ramp is not None else finetuned.config.ramp_min
        if warmup + ramp >= finetuned.config.chunk_size:
            logging.warning(
                f"warmup+ramp ({warmup}+{ramp}) >= chunk_size ({finetuned.config.chunk_size}): "
                "no fresh (w=1) positions to generate."
            )

    logging.info(f"Evaluating finetuned checkpoint sequentially (mode={args.mode})...")
    ft_result = evaluate(
        finetuned, dataset, dataset_stats, args.device,
        args.batch_size, args.num_workers, args.mode, warmup, ramp,
    )
    _print_report(f"finetuned  ({args.policy_path})", ft_result)

    # ---- baseline (optional), evaluated sequentially after the finetuned run ----
    if args.compare_base:
        reference_cfg = finetuned.config
        del finetuned
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        baseline = load_baseline(reference_cfg, args.device)
        logging.info(f"Evaluating pi05_base baseline sequentially (mode={args.mode})...")
        base_result = evaluate(
            baseline, dataset, dataset_stats, args.device,
            args.batch_size, args.num_workers, args.mode, warmup, ramp,
        )
        _print_report(f"baseline   ({BASE_REPO_ID}, no finetuning)", base_result)

        ft = ft_result["overall"]["mse_norm"]
        bl = base_result["overall"]["mse_norm"]
        print(f"\n>>> normalized-space action MSE: finetuned={ft:.6f}  base={bl:.6f}  "
              f"(finetuning {'reduced' if ft < bl else 'did NOT reduce'} MSE by {bl - ft:+.6f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
