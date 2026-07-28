# 🧪 scout-matter

This README explains how to use **scout-matter**, our modified version of Microsoft's MatterGen diffusion model, extended with custom **guidance functions** to bias crystal generation. These include **coordination targeting** and **volume control**. This functionality is entirely **training-free**: no retraining is required when adding new guidance objectives.

---

## 📅 Quick Start

Example to generate structures with mean-coordination guidance:

```bash
mattergen-generate "results/Li-Co-O_guided_env" \
    --pretrained-name=chemical_system \
    --batch_size=50 \
    --properties_to_condition_on="{'chemical_system':'Li-Co-O'}" \
    --record_trajectories=False \
    --diffusion_guidance_factor=2.0 \
    --guidance="{'mean_coordination': {'mode': 'huber', 'Co-O': [6, 2.6]}}" \
    --diffusion_loss_weight="[0.01, 0.01, True]" \
    --print_loss=False \
    --self_rec_steps=3 \
    --back_step=2
```

---

## 🌍 Arguments Explained

| Argument                                                             | Type                   | Description                                                                       |
| -------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------- |
| `output_path`                                                        | `str`                  | Directory to save generated structures                                            |
| `pretrained_name`                                                    | `str`                  | Name of pretrained model from HuggingFace, check mattergen.md to see all available models (e.g., `chemical_system` to fix the system)               |
| `model_path`                                                         | `str`                  | Alternative to `pretrained_name`; path to local checkpoint                        |
| `batch_size`                                                         | `int`                  | Number of structures per batch                                                    |
| `num_batches`                                                        | `int`                  | Number of batches to generate                                                     |
| `properties_to_condition_on`                                         | `dict`                 | Conditioning properties when a finetuned model has been chosen, like `{'chemical_system':'Li-Co-O'}`                     |
| `diffusion_guidance_factor`                                          | `float`                | Strength of guidance correction applied to the classifier-free diffusion when a finetuned model has been chosen  (choice for guidance : `2.0`)                           |
| `guidance`                                                           | `dict`                 | Dictionary defining the training-free guidance  (see below)                                     |
| `diffusion_loss_weight`                                              | `[float, float, bool]` | `[g, k, normalize]` where:                                                        |
| └─ `g`: weight of forward guidance                                   |                        |                                                                                   |
| └─ `k`: weight of backward guidance                                  |                        |                                                                                   |
| └─ `normalize`: whether to normalize gradients in the guidance steps (recommended: `True`) |                        |                                                                                   |
| `print_loss`                                                         | `bool`                 | Save loss values during generation                                               |
| `self_rec_steps`                                                     | `int`                  | Number of self-recurrence steps                                                   |
| `back_step`                                                          | `int`                  | Number of backward guidance steps per backward guidance                                      |
| `algo`                                                               | `int`                 | `0` (Algo 2) = outer-loop correction; `1` (Algo 1) = inner-loop correction before forward pass; `2` (Algo 3) = inner-loop correction after forward pass |
| `record_trajectories`                                                | `bool`                 | Whether to record step-wise atomic positions                                      |
| `force_gpu`                                                          | `int`                  | Force use of specific GPU ID                                                      |

---

## 🔍 Guidance Dictionary Format

You can guide generation using one or more objectives. Each is passed via the `--guidance` argument.

### 🔮 Mean-Coordination Objective

The existing `mean_coordination` loss supports two definitions, selected by
`coordination_mode`.

#### Sigmoid-weighted coordination (default)

```bash
--guidance="{'mean_coordination': {
  'coordination_mode': 'soft_count',
  'mode': 'huber',
  'alpha': 3.0,
  'Cu-P': [4, 2.6],
  'Cu-Cu': [0, 2.9],
  'Cu-S': 1,
  'H-[Pd,Ni,Pt]': 2,
  '[Fe,Nd]-B': 6
}}"
```

- `coordination_mode: 'soft_count'`: optional; this is the backward-compatible
  default.
- `mode`: can be `l1`, `l2`, or `huber`
- `alpha`: sigmoid steepness in inverse angstroms; optional, with a default of `2.0`
- The loss compares the target with the mean sigmoid-weighted coordination of
  the selected central atoms.

#### k-th-neighbor boundary loss

```bash
--guidance="{'mean_coordination': {
  'coordination_mode': 'kth_neighbor',
  'margin': 0.05,
  'temperature': 0.10,
  'Co-O': [5, 2.42]
}}"
```

For every central `Co` atom, the current periodic `Co-O` distances are sorted.
For target coordination `k=5`, the loss pulls the fifth O neighbor inside
`r_cut - margin` and pushes the sixth O neighbor outside `r_cut + margin`.
The two penalties are smooth softplus functions of the selected distances.

- `coordination_mode: 'kth_neighbor'`: selects the new behavior without adding
  another registered loss name.
- `margin`: safety interval around the cutoff in angstroms; default `0.05`.
- `temperature`: softplus smoothing width in angstroms; default `0.10`. Smaller
  values approach a hinge loss.
- The resulting loss has units of angstroms, so its guidance weight is not
  numerically equivalent to the dimensionless sigmoid-count loss weight.
- The target coordination `k` must be a non-negative integer. For `k=0`, only
  the nearest neighbor is pushed outside the cutoff margin.
- `mode` and `alpha` belong to the sigmoid-weighted formulation and are ignored
  in `kth_neighbor` mode.
- A pair-specific override can be written as
  `[target_CN, cutoff_radius, margin, temperature]`, for example
  `'Co-O': [5, 2.42, 0.05, 0.10]`.
- Sorting is differentiable with respect to the currently selected distances
  except at exact distance ties, where the ranking changes. Autograd follows
  the ordering returned for that step.
- The same switch is accepted by the existing `environment`,
  `target_coordination_share`, and `target_coordination` aliases. In
  `kth_neighbor` mode they return the direct boundary penalty.

The following pair and group syntax is shared by both modes:

- `A-B`: `[target_CN, cutoff_radius]`
- `A-B`: `int`; in this case the cutoff is the sum of the covalent radii plus
  `0.5` angstrom.
- `A-[B,C,D]`: group environment target for `CN(A-[B,C,D])`, the total coordination of `A` by any species in the set.
  If no cutoff is supplied, the cutoff is the maximum default cutoff over all `A-B`,
  `A-C`, and `A-D` pairs.
- `[A,B,C]-D`: grouped-center target. All atoms of types `A`, `B`, and `C` are
  pooled as centers, their `D` neighbors are counted, and the mean is taken over
  all center atoms. Species with more atoms therefore have proportionally more weight.
- A group may appear on only one side. Keys such as `[A,B]-[C,D]` are rejected.
- Overlapping sides are valid. For example, `[A,B]-B` does not count a central
  `B` atom as its own neighbor; this self-interaction correction applies only to
  the `B` centers.
- Multiple atom-pair environments may be defined.

### 🎯 Target-Coordination Share Objective

```bash
--guidance="{'target_coordination_share': {'alpha': 2.0, 'Co-O': [5, 2.42, 0.5]}}"
```

- Guides the fraction of central atoms having the requested coordination.
- The optional third list value is the coordination-space tolerance `tau`.

### 🏢 Volume Objective

```bash
--guidance="{'volume': 80.0}"
```

- Tries to enforce a specific cell volume in Å³.

### 🏢 Volume-Per-Atom Objective

```bash
--guidance="{'volume_pa': 15.0}"
```

- Tries to enforce a specific cell volume per atom in Å³.

### 📊 Combine Multiple Objectives

```bash
--guidance="{'mean_coordination': {'mode': 'l1', 'Li-O': [4, 2.5]}, 'volume': 75.0}"
```

## 🔁 Multiple Guided Runs

Use the repository-root `multiple_runs.sh` to repeat guided generation. This is the
only runner implementation in the repository. It handles independent runs,
multiple batches per run, OOM recovery, and result aggregation.

There is one guidance interface: pass the complete top-level guidance dictionary
to `--guidance`. The same dictionary format is used by `mattergen-generate` and
supports single or combined objectives. The runner does not assemble guidance
from separate type or objective-specific options, and it does not accept
positional arguments or short aliases.

For example, this generates 22 `Ni-Pd-H` structures per batch with
`CN([Pd,Ni]-H) = 6`:

```bash
./multiple_runs.sh \
    --batch-size 22 --runs 50 --system Ni-Pd-H \
    --guidance "{'mean_coordination': {'mode':'huber', 'alpha':3, '[Pd,Ni]-H':6}}" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --normalize true --self-rec-steps 3 --back-step 2 \
    --algorithm 1 --gpu 2
```

The only alternative is a YAML input file:

```bash
./multiple_runs.sh --config examples/multiple_runs/mean_coordination.yaml
```

`--config` must be the only command-line option. YAML uses the long option names
with underscores instead of hyphens, and `guidance` is a YAML mapping rather
than a quoted Python dictionary:

```yaml
batch_size: 22
runs: 50
system: Ni-Pd-H
guidance:
  mean_coordination:
    mode: huber
    alpha: 3.0
    "[Pd,Ni]-H": 6
forward_weight: 0.01
backward_weight: 0.01
normalize: true
self_rec_steps: 3
back_step: 2
algorithm: 1
gpu: 2
```

Runnable YAML examples cover every canonical guidance configuration:

- [mean coordination, including grouped species](examples/multiple_runs/mean_coordination.yaml)
- [k-th-neighbor coordination](examples/multiple_runs/kth_neighbor.yaml)
- [target coordination share](examples/multiple_runs/target_coordination_share.yaml)
- [target volume](examples/multiple_runs/volume.yaml)
- [target volume per atom](examples/multiple_runs/volume_per_atom.yaml)
- [combined objectives](examples/multiple_runs/combined.yaml)

The main non-guidance settings are:

- `batch_size`, `num_batches`, `runs`, and `system`
- `forward_weight`, `backward_weight`, and `normalize`
- `self_rec_steps`, `back_step`, and `algorithm`
- `diffusion_guidance_factor`, `gpu`, and `gpu_memory_gb`
- `oom_retries`, `oom_backoff_percent`, `min_batch_size`, and `oom_wait_seconds`
- `base_dir`, `log_file`, and `dry_run`

Use `./multiple_runs.sh --help` for defaults and the corresponding CLI option
names. On an OOM failure, the script retries the same run with
`ceil(current_batch * oom_backoff_percent / 100)` structures per batch. It stops
when retries are exhausted or the next batch would be smaller than
`min_batch_size`. A non-OOM failure stops immediately.

Each run writes to `run_N/` under
`<base-dir>/results/<system>/<guidance>/<parameters>/<settings>/`. The settings
directory contains the combined `generated_crystals.extxyz` and `durations.csv`;
each run directory contains per-attempt logs. The script uses an active virtual
environment, otherwise it activates `../.venv` when available or uses
`mattergen-generate` from `PATH`.

---

## 🧩 How to Add a New Guidance Function

The loss registry and general guidance orchestration are in
`mattergen/diffusion/diffusion_loss.py`. Coordination-specific calculations are
kept separately in `mattergen/diffusion/coordination_loss.py`.

### Step 1: Define Your Custom Loss

Create a new loss function that takes predicted structures as a `ChemGraphBatch` object as input and returns a scalar or tensor loss. For example:

```python
def new_loss(x, t, target):
    """
    Example of a new loss function.
    This is just a placeholder and should be replaced with an actual implementation.
    """
    # x : ChemGraph object (ChemGraphBatch usually)
    # t : timestep
    # target : target value
    # Return : torch.tensor with the same size as the batch
    pass
```

### Step 2: Register the Loss in `LOSS_REGISTRY`

Add the name of your new loss function in the `LOSS_REGISTRY`:

```python
LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "volume": volume_loss,
    "environment": environment_loss,
    "energy": energy,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}
```

This allows the key `"new_loss"` to be passed in the CLI `--guidance` argument.

### Step 3: Use Your Loss via CLI

Pass your custom objective directly from the command line:

```bash
--guidance="{'new_loss': target_value}"
```

This value will be routed into your `new_loss` implementation as the input `target` automatically at generation time.

---

## 📃 Output

- Generated crystals are saved in `output_path/`
- Input settings are saved in `input_parameters.txt`
- (Optional) If `print_loss=True`, logs of guidance loss are saved into `diffusion_loss_history.txt`


---

## ⚠️ Notes

- No model retraining is needed to add new guidance functions.
- Environment matching works best when structures are near physical.
- If CUDA memory errors occur, reduce `batch_size` or set `--gpu_memory_gb` manually.

---

## ⚙️ Example Command

These were the best parameters we found for our experiment:

```bash
mattergen-generate results/test_env \
    --pretrained-name=chemical_system \
    --properties_to_condition_on="{'chemical_system':'Cu-P'}" \
    --guidance="{'environment': {'mode': 'huber', 'Cu-P': [4, 2.5], 'Cu-Cu': [0, 2.9]}}" \
    --diffusion_guidance_factor=2.0 \
    --diffusion_loss_weight="[0.01, 0.01, True]" \
    --self_rec_steps=3 --back_step=2 --algo=1
```

---

## 🚀 Contributions

Guidance adapted by Auguste de Lambilly in collaboration with:

- Vladimir Baturin
- Jean-Claude Crivello
- Florence d'Alché-Buc
- Guillaume Lambard
- Nataliya Sokolovska 

For more info on the original project, see MatterGen: [https://github.com/microsoft/mattergen](https://github.com/microsoft/mattergen)
