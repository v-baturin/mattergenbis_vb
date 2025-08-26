# 🧪 MatterGen Guided Crystal Generation

This README explains how to use the modified **MatterGen** diffusion model from Microsoft, extended with custom **guidance functions** to bias crystal generation. These include **energy minimization**, **environment targeting**, **volume control**, and others. This functionality is entirely **training-free**: no retraining is required when adding new guidance objectives.

---

## 📅 Quick Start

Example to generate structures with environment-based guidance:

```bash
mattergen-generate "results/Li-Co-O_guided_env" \
    --pretrained-name=chemical_system \
    --batch_size=50 \
    --properties_to_condition_on="{'chemical_system':'Li-Co-O'}" \
    --record_trajectories=False \
    --diffusion_guidance_factor=2.0 \
    --guidance="{'environment': {'mode': 'huber', 'Co-O': [6, 2.6]}}" \
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
| `pretrained_name`                                                    | `str`                  | Name of pretrained model from HuggingFace, check README.md to see all the available model (e.g., `chemical_system` to fix the system)               |
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

### 🔮 Environment Objective

```bash
--guidance="{'environment': {
  'mode': 'huber',
  'Cu-P': [4, 2.6],
  'Cu-Cu': [0, 2.9],
  'Cu-S': 1
}}"
```

- `mode`: can be `l1`, `l2`, or `huber`
- `A-B`: `[target_coordination, cutoff_radius]` 
- `A-B`: `int` in this case the cutoff radius used is the sum of the covalence radius
- Multiple atom-pair environments may be defined.

### ⚛️ Energy Objective

```bash
--guidance="{'energy': None}"
```

- Any value (including `None`) is accepted.
- Internally uses the **Mattersim** model to estimate energy.
- Target is not fixed; the gradient guides toward lower energy regions.

### 🏢 Volume Objective

```bash
--guidance="{'volume': 80.0}"
```

- Tries to enforce a specific cell volume in Å³.

### 📊 Combine Multiple Objectives

```bash
--guidance="{'energy': None, 'environment': {'mode': 'l1', 'Li-O': [4, 2.5]}, 'volume': 75.0}"
```

---

## 🧩 How to Add a New Guidance Function

To implement a custom guidance objective, follow these steps in the MatterGen codebase. All guidance logic is handled in `mattergen/diffusion/diffusion_loss.py`. You need to add your loss inside this file.

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

### Step 2: Register the Loss the `LOSS_REGISTRY`

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

```bash
mattergen-generate results/test_env \
    --pretrained-name=chemical_system \
    --properties_to_condition_on="{'chemical_system':'Cu-P'}" \
    --guidance="{'environment': {'mode': 'huber', 'Cu-P': [4, 2.5], 'Cu-Cu': [0, 2.9]}}" \
    --diffusion_guidance_factor=2.0 \
    --diffusion_loss_weight="[0.01, 0.01, True]" \
    --self_rec_steps=3 --back_step=2 --algo=True
```

---

## 🚀 Contributions

Guidance adapted by Auguste de Lambilly in collaboration with:

- Vladimir Baturin
- Jean-Claude Crivello
- Florence d'Alché-Buc
- Guillaume Lambard
- Nataliya Sokolovska 

For more info, see MatterGen: [https://github.com/microsoft/mattergen](https://github.com/microsoft/mattergen)

