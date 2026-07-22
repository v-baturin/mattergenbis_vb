# 🧪 scout-matter

This README explains how to use **scout-matter**, our modified version of Microsoft's MatterGen diffusion model, extended with custom **guidance functions** to bias crystal generation. These include **energy minimization**, **environment targeting**, **volume control**, and others. This functionality is entirely **training-free**: no retraining is required when adding new guidance objectives.

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

```bash
--guidance="{'mean_coordination': {
  'mode': 'huber',
  'alpha': 3.0,
  'Cu-P': [4, 2.6],
  'Cu-Cu': [0, 2.9],
  'Cu-S': 1,
  'H-[Pd,Ni,Pt]': 2,
  '[Fe,Nd]-B': 6
}}"
```

- `mode`: can be `l1`, `l2`, or `huber`
- `alpha`: sigmoid steepness in inverse angstroms; optional, with a default of `2.0`
- `A-B`: `[target_CN, cutoff_radius]`
- `A-B`: `int`; in this case the cutoff radius used is the sum of the covalent radii
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

### ⚛️ Energy Objective

```bash
--guidance="{'energy': None}"
```

- Any value (including `None`) is accepted.
- Internally uses the **MatterSim** model to estimate energy.
- Target is not fixed; the gradient guides toward lower energy regions.

### 🏢 Volume Objective

```bash
--guidance="{'volume': 80.0}"
```

- Tries to enforce a specific cell volume in Å³.

### 📊 Combine Multiple Objectives

```bash
--guidance="{'energy': None, 'mean_coordination': {'mode': 'l1', 'Li-O': [4, 2.5]}, 'volume': 75.0}"
```

## 🔁 Multiple Guided Runs

Use the repository-root `multiple_runs.sh` to repeat guided generation. This is the
only runner implementation in the repository. It handles independent runs,
multiple batches per run, OOM recovery, result aggregation, and both environment
and other registered guidance functions.

The environment convenience options are inserted into:

```bash
--guidance="{'environment': {'mode': LOSS_MODE, 'alpha': ALPHA, ENVIRONMENT}}"
```

`--environment` is the comma-separated content of the `environment` dictionary. It may contain one pair target, for example `"'Si-O':[6, 2.5]"`, or several pair targets, for example `"'Cu-P':[4, 2.5], 'Cu-Cu':[0, 2.9]"`.
For a group environment loss, use either `A-[B1,B2,...]` or `[A1,A2,...]-B`.
For example, `"'B-[Fe,Nd]':6"` targets `CN(B-[Fe,Nd]) = 6`, while
`"'[Fe,Nd]-B':6"` targets the mean `CN([Fe,Nd]-B) = 6` over all Fe and Nd centers.
Groups on both sides, such as `[Fe,Nd]-[B,C]`, are not supported.

For another registered loss, provide `--guidance-type` and a complete inner
dictionary through `--guidance-params`, for example:

```bash
./multiple_runs.sh \
    --guidance-type target_coordination_share \
    --guidance-params "{'Co-O':3}"
```

`--guidance-params` cannot be combined with `--environment`, `--loss-mode`, or
`--alpha`. For top-level multiobjective guidance such as `{'environment': {...},
'energy': None}`, call `mattergen-generate` directly.

```bash
./multiple_runs.sh --help
```

Named options are recommended. The original positional interface and the former
OOM-runner short flags are still accepted by this same file for compatibility.
The main options are:

- `--batch-size`: starting batch size; default `20`
- `--num-batches`: batches generated inside each run; default `1`
- `--runs`: number of independent run directories; default `50`
- `--system`: chemical system, for example `Si-O` or `Li-Co-O`
- `--environment`: one or more coordination targets
- `--guidance-type` and `--guidance-params`: generic guidance configuration
- `--forward-weight`, `--backward-weight`, and `--normalize`: diffusion loss settings
- `--self-rec-steps`, `--back-step`, and `--algorithm`: guidance algorithm settings
- `--loss-mode`: `l1`, `l2`, or `huber`; default `l1`
- `--alpha`: sigmoid steepness; default `2.0`
- `--gpu`: GPU index, or `None`; default `None`
- `--gpu-memory-gb` and `--diffusion-guidance-factor`: generation settings
- `--oom-retries`, `--oom-backoff-percent`, `--min-batch-size`, and
  `--oom-wait-seconds`: OOM recovery settings
- `--extra-arg`: append one argument to `mattergen-generate`; repeat as needed
- `--log-file` and `--base-dir`: log and result collection locations
- `--dry-run`: validate and print commands without launching generation

On an OOM failure, the script retries the same run after changing the batch size
to `ceil(current_batch * backoff_percent / 100)`. It stops when retries are
exhausted or the next batch would be smaller than `--min-batch-size`. A non-OOM
failure stops immediately. Every attempt has its own log file.

Example with a default cutoff, meaning six `Si-O` neighbors using the built-in cutoff:

```bash
./multiple_runs.sh \
    --batch-size 20 --runs 50 --system Si-O \
    --environment "'Si-O':6" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 0
```

Example with an explicit cutoff, meaning six `Si-O` neighbors with `r_cut=2.5`:

```bash
./multiple_runs.sh \
    --batch-size 20 --runs 50 --system Si-O \
    --environment "'Si-O':[6, 2.5]" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 0
```

Example with multiple environment targets in the same run:

```bash
./multiple_runs.sh \
    --batch-size 20 --runs 50 --system Cu-P \
    --environment "'Cu-P':[4, 2.5], 'Cu-Cu':[0, 2.9]" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 0
```

Example with a group environment target, meaning `CN(B-[Fe,Nd]) = 6` using the default group cutoff:

```bash
./multiple_runs.sh \
    --batch-size 20 --runs 50 --system Fe-Nd-B \
    --environment "'B-[Fe,Nd]':6" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 0
```

Example with a group environment target and an explicit cutoff:

```bash
./multiple_runs.sh \
    --batch-size 20 --runs 50 --system Fe-Nd-B \
    --environment "'B-[Fe,Nd]':[6, 2.8]" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 0
```

Example with grouped center species, meaning a mean `CN([Fe,Nd]-B) = 6` over
all Fe and Nd atoms:

```bash
./multiple_runs.sh \
    --batch-size 20 --runs 50 --system Fe-Nd-B \
    --environment "'[Fe,Nd]-B':6" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 0
```

Example for 22 `Ni-Pd-H` structures per batch with `CN([Pd,Ni]-H) = 6` and
an explicit `alpha=3` override:

```bash
./multiple_runs.sh \
    --batch-size 22 --runs 50 --system Ni-Pd-H \
    --environment "'[Pd,Ni]-H':6" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --algorithm 1 --loss-mode huber --gpu 2 --alpha 3
```

Each run writes to `run_N/` under
`<base-dir>/results/<system>/<guidance>/<parameters>/<settings>/`. The settings
directory contains the combined `generated_crystals.extxyz` and `durations.csv`;
each run directory contains per-attempt logs. The script uses an active virtual
environment, otherwise it activates `../.venv` when available or uses
`mattergen-generate` from `PATH`.

---

## 🧩 How to Add a New Guidance Function

To implement a custom guidance objective, follow these steps in the scout-matter codebase. All guidance logic is handled in `mattergen/diffusion/diffusion_loss.py`. You need to add your loss inside this file.

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
