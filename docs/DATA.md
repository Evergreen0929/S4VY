# Data

**No dataset is redistributed with this repository.** Several sources prohibit
redistribution outright or require a signed agreement before download. Obtain each one
from its own source, accept its license, and point `env.sh` at your local copy.

## Segmenter training mix

22 sources, 33,234 items, ~6.07M loader-visible frames. Each training sample draws 8-16
frames from one item. `exhaustive` marks sources whose ground truth labels every object in
the frame; the others label only a subset, and the no-object penalty is skipped for them.

| Group | Source | Items | exhaustive | How to obtain |
|---|---|---:|---|---|
| static / multi-view | Infinigen | 1,466 sub-scenes | yes | procedural; generate or download from the Infinigen project |
| static / multi-view | ScanNet++ v2 | 898 scenes | yes | **signed Terms of Use required; redistribution prohibited** |
| static / multi-view | RealEstate10K | 5,127 scenes | yes | **Google releases video ids + camera poses only; frames must be fetched by the user** |
| video / exhaustive | VIPSeg | 2,806 clips | yes | public release |
| video / exhaustive | Cityscapes-VPS | 394 clips | yes | **registration required; redistribution prohibited** |
| video / exhaustive | KITTI-STEP | 12 clips | yes | CC BY-NC-SA |
| video / exhaustive | JRDB-PanoTrack | 135 clips | yes | **registration required** |
| video / partial | YouTube-VOS | 3,471 clips | no | public release |
| video / partial | DAVIS | 60 clips | no | public release |
| video / partial | LVOS | 120 clips | no | public release |
| video / partial | MOSE | 1,496 clips | no | public release |
| video / partial | MeViS, SAIL-VOS, VOST, UVO, DynRep, PointOdyssey, MVS-Synth, Spring, SAV-X | — | no | see each project |
| pseudo-only | DL3DV, Matterport3D | — | no | **Matterport3D requires a signed EULA** |

ScanNet++ and RealEstate10K alone account for roughly 22% of the frames, and neither may
be redistributed. This is why the repository ships the recipe rather than the data.

## What the released recipe does ship

- `segomega/configs/*.yaml` — the exact subset list, sampling weights, and the
  exhaustive:non-exhaustive curriculum used for each run.
- `segomega/data/` — the loaders that turn each source into the common format.

Derived annotations we generated ourselves (SAM2-refined instance ids, pseudo labels) can
be released separately; the underlying RGB cannot.

## A note on the released s4vy recipe

The released configuration uses the clean 898-scene ScanNet++ split. The historical
checkpoint was trained on 907 scenes, which includes the 9 scenes that the IGGT benchmark
uses for evaluation. Numbers produced with that checkpoint on the ScanNet++ arm of IGGT
are therefore not a clean held-out measurement. The ScanNet arm is unaffected: ScanNet is
not in the training mix at all.

## Grounder and critic training data

Built from ScanRefer, our ScanNet++ annotations, Ref-DAVIS, LVOS and MeViS expressions
(~80k expressions). The chain-of-thought supervision is distilled from scene geometry by
`grounding/build_cot_data.py` and `grounding/gen_spp_cot.py`; the critic trains on paired
mask candidates rendered by `grounding/perframe_cards.py`.
