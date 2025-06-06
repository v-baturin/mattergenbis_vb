# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Generic, TypeVar

import torch
import json

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import Loss
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.diffusion.timestep_samplers import TimestepSampler, UniformTimestepSampler
from mattergen.common.data.chemgraph import ChemGraph, ChemGraphBatch

T = TypeVar("T", bound=BatchedData)
BatchTransform = Callable[[T], T]


def identity(x: T) -> T:
    return x

# Example: 1 atom, cell is identity
cell = torch.eye(3).unsqueeze(0)  # shape [1, 3, 3]
atomic_numbers = torch.tensor([1], dtype=torch.long)  # e.g., Hydrogen
pos = torch.zeros((1, 3))  # position at origin

g = ChemGraph(
    atomic_numbers=atomic_numbers,
    pos=pos,
    cell=cell,
    pbc=torch.tensor([1, 1, 1], dtype=torch.bool)  # periodic in all directions
)


class DiffusionModule(torch.nn.Module, Generic[T]):
    """Denoising diffusion model for a multi-part state
    diffusion_loss_fn: Loss function that is used for the universal diffusion guidance
    diffusion_loss_weight: Weight for the diffusion loss (theorelically should be 1.0)  
    """

    def __init__(
        self,
        model: ScoreModel[T],
        corruption: MultiCorruption[T],
        loss_fn: Loss,
        pre_corruption_fn: BatchTransform | None = None,
        timestep_sampler: TimestepSampler | None = None,
        diffusion_loss_fn: Callable[[T, torch.Tensor], torch.Tensor] | None = None,
        diffusion_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.corruption = corruption
        self.loss_fn = loss_fn
        self.pre_corruption_fn = pre_corruption_fn or identity
        self.model_targets = {k: ModelTarget(v) for k, v in loss_fn.model_targets.items()}

        self.timestep_sampler = timestep_sampler or UniformTimestepSampler(
            min_t=1e-5,
            max_t=corruption.T,
        )
        self.diffusion_loss_fn = diffusion_loss_fn  
        self.diffusion_loss_weight = diffusion_loss_weight 
        self.diffusion_loss_history = [] # To keep track of diffusion loss values
        self.print_loss_history = False  # Flag to control printing of loss history

        # Check corruption for nn.Modules and register them here.
        self._register_corruption_modules()

    def _register_corruption_modules(self):
        """
        Register corruptions that are instances of `torch.nn.Module`s for proper device, parameter,
        etc handling.
        """
        assert isinstance(self.corruption, MultiCorruption)
        for idx, (key, _corruption) in enumerate(self.corruption._corruptions.items()):
            if isinstance(_corruption, torch.nn.Module):
                self.register_module(f"MultiCorruption:{idx}:{key}", _corruption)

    def calc_loss(
        self, batch: T, node_is_unmasked: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate loss and metrics given a batch of clean data which may include
        context/conditioning fields. Add noise, predict score using score model, then calculate
        loss.

        Args:
            batch: batch of training data
            node_is_unmasked: mask that has a value 1 for nodes that are included in the loss, and
                a value of 0 for nodes that should be ignored. If None, all nodes are included.

        Returns:
            loss: the loss for the batch
            metrics: a dictionary of metrics for the batch
        """
        batch = self.pre_corruption_fn(batch)

        noisy_batch, t = self._corrupt_batch(batch)

        score_model_output = self.model(noisy_batch, t)
        loss, metrics = self.loss_fn(
            multi_corruption=self.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
            node_is_unmasked=node_is_unmasked,
        )
        assert loss.numel() == 1

        return loss, metrics

    def _corrupt_batch(
        self,
        batch: T,
    ) -> tuple[T, torch.Tensor]:
        """
        Corrupt a batch of data for use in a training step:
        - sample a different timestep for each sample in the batch
        - add noise according to the corruption process

        Args:
            batch: Batch of clean states

        Returns:
            noisy_batch: batch of noisy samples
            t: the timestep used for each sample in the batch

        """
        # Sample timesteps
        t = self.sample_timesteps(batch)

        # Add noise to data
        noisy_batch = self.corruption.sample_marginal(batch, t)

        return noisy_batch, t

    def score_fn(self, x: T, t: torch.Tensor) -> T:
        """Calculate the score of a batch of data at a given timestep

        Args:
            x: batch of data
            t: timestep

        Returns:
            score: score of the batch of data at the given timestep
        """
        model_out: T = self.model(x, t)
        fns = {k: convert_model_out_to_score for k in self.corruption.sdes.keys()}

        scores = apply(
            fns=fns,
            model_out=model_out,
            broadcast=dict(t=t, batch=x),
            sde=self.corruption.sdes,
            model_target=self.model_targets,
            batch_idx=self.corruption._get_batch_indices(x),
        )

        # --- NEW: Diffusion loss gradient modification ---
        if self.diffusion_loss_fn is not None and (t<self.corruption.T*0.9).all():
                        # Set requires_grad=True for all relevant fields at once
            replace_kwargs = ["pos", "cell"]

            # Estimate x_0_hat for pos and cell using the Ancestral Sampling Formula
            x0_hat = {}
            for field in replace_kwargs:
                # Get SDE for the relevant field 
                sde = getattr(self.corruption.sdes,field)  
                # Get alpha_t and sigma_t for the current t
                alpha_t, sigma_t = sde.mean_coeff_and_std(x=getattr(x, field), t=t, batch_idx=self.corruption._get_batch_indices(x)[field], batch=x)
                x0_hat[field] = (getattr(x, field) + sigma_t**2 * scores[field]) / alpha_t

            # Create a new ChemGraphBatch estimating x0 with requires_grad=True for pos and cell    
            x_for_grad = ChemGraphBatch(
                atomic_numbers=x.atomic_numbers,
                pos=x0_hat["pos"].clone().detach().requires_grad_(True),
                cell=x0_hat["cell"].clone().detach().requires_grad_(True),
                batch=x.batch,
            )

            #print("cell:",x_for_grad.cell.requires_grad, "\n","pos:",x_for_grad.pos.requires_grad)

            grad_dict = {}
            with torch.autograd.set_grad_enabled(True):
                diffusion_loss = self.diffusion_loss_fn(x_for_grad, t)
            
            if self.print_loss_history:
                self.diffusion_loss_history.append(diffusion_loss.cpu().tolist())
                
            for field in replace_kwargs:
                grad = torch.autograd.grad(
                    diffusion_loss, getattr(x_for_grad, field),
                    grad_outputs=torch.ones_like(diffusion_loss),
                    create_graph=True,
                    allow_unused=True
                )[0]
                if grad is None:
                    grad = torch.zeros_like(getattr(x_for_grad, field))
                #print(f"Gradient for {field}:", grad)
                grad_dict[field] = grad
            # Ensure that the gradients are detached from the computation graph
            #print("Gradients for pos and cell:", grad_dict)
            #print("------------------------------\n",
            #       "Diffusion loss gradient:", grad_dict, "\n",
            #       "Diffusion loss function:", self.diffusion_loss_fn(x_for_grad,None), "\n",
            #      self.diffusion_loss_fn,"\n------------------------------", "\n",
            #      "scores:", scores, "\n",)
            for k in grad_dict:
                if k in scores:
                    #print(f"Before update - scores[{k}]:", scores[k])
                    #print(f"Grad for {k}:", grad_dict[k])
                    scores[k] = scores[k] - self.diffusion_loss_weight * grad_dict[k]
        # --- END NEW ---

        return model_out.replace(**scores)

    def sample_timesteps(self, batch: T) -> torch.Tensor:
        """Sample the timesteps, which will be used to determine how much noise
        to add to data.

        Args:
           batch: batch of data to be corrupted

        Returns: sampled timesteps
        """
        return self.timestep_sampler(
            batch_size=batch.get_batch_size(),
            device=self._get_device(batch),
        )

    def _get_device(self, batch: T) -> torch.device:
        return next(batch[k].device for k in self.corruption.sdes.keys())

    def set_diffusion_loss(
        self,
        diffusion_loss_fn: Callable[[T, torch.Tensor], torch.Tensor],
        diffusion_loss_weight: float = 1.0,
    ):
        """Set or update the diffusion loss function and its weight after the module has been initialized."""
        self.diffusion_loss_fn = diffusion_loss_fn
        self.diffusion_loss_weight = diffusion_loss_weight

    def save_diffusion_loss_history(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.diffusion_loss_history, f)