import functools as ft

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from og.dyn_types import BNFloat, BObs, NFloat
from og.grad_utils import compute_norm, compute_norm_and_clip
from og.jax_types import BBool, BFloat, FloatDict, FloatScalar
from og.rng import PRNGKey
from og.train_state import TrainState

from fge.core.utils.jax_util import get_leading_dim_fast


def update_mse(
    pred: TrainState[NFloat],
    bn_tgt: BNFloat,
    b_obs: BObs,
    name: str,
    sum_dim: bool = False,
    clip_grad: float | None = None,
) -> tuple[TrainState[BFloat], FloatDict]:
    """
    Updates the model parameters using Mean Squared Error (MSE) loss.

    Args:
        pred (TrainState[NFloat]): The current model state containing parameters and optimizer state.
        bn_tgt (BNFloat): The target values (batch of target outputs) for the MSE computation.
        b_obs (BObs): The batch of observations (inputs) to the model.
        name (str): A name identifier for logging purposes.
        sum_dim (bool, optional): If True, sums the MSE across dimensions; otherwise, computes the mean. Defaults to False.
        clip_grad (float | None, optional): Gradient clipping threshold. If None, no clipping is applied. Defaults to None.

    Returns:
        tuple[TrainState[BFloat], FloatDict]:
            - TrainState[BFloat]: The updated model state after applying gradients.
            - FloatDict: A dictionary containing information about the loss and gradient norms.

    Steps:
        1. Computes the MSE loss between the model predictions and the target values.
        2. Optionally sums or averages the loss across dimensions based on `sum_dim`.
        3. Computes gradients of the loss with respect to the model parameters.
        4. Optionally clips the gradients if `clip_grad` is specified.
        5. Updates the model parameters using the computed gradients.
        6. Returns the updated model state and a dictionary of loss and gradient information.
    """

    def get_mse_loss(params):
        """
        Computes the MSE loss between model predictions and target values.
        """
        # Apply the model to the batch of observations using the given parameters.
        bn_Vl = jax.vmap(ft.partial(pred.apply_with, params=params))(b_obs)
        # Ensure the shapes of predictions and targets match.
        assert bn_Vl.shape[0] == bn_tgt.shape[0] == b
        assert bn_Vl.shape == bn_tgt.shape
        ndim = bn_Vl.ndim

        # Compute the element-wise squared error.
        bn_mse = (bn_Vl - bn_tgt) ** 2

        # Aggregate the MSE across dimensions based on `sum_dim`.
        if ndim == 1:
            b_mse = bn_mse
        else:
            if sum_dim:
                b_mse = jnp.sum(bn_mse, axis=1)
            else:
                b_mse = jnp.mean(bn_mse, axis=1)

        # Compute the mean loss over the batch.
        loss_Vl = jnp.mean(b_mse)

        # Store the loss in the info dictionary for logging.
        info = {f"{name}/Loss": loss_Vl}
        return loss_Vl, info

    # Get the batch size from the target shape.
    b = bn_tgt.shape[0]
    # Compute gradients of the loss with respect to the model parameters.
    grads, info = jax.grad(get_mse_loss, has_aux=True)(pred.params)

    # Compute gradient norms and optionally clip them.
    if clip_grad is None:
        info[f"{name}/grad"] = compute_norm(grads)
    else:
        grads, info[f"{name}/grad"] = compute_norm_and_clip(grads, clip_grad)

    # Apply the computed gradients to update the model parameters.
    pred_new = pred.apply_gradients(grads=grads)
    return pred_new, info


def update_bce(
    logits_pred: TrainState[FloatScalar],
    b_obs: BObs,
    b_ispos: BBool,
    name: str,
    smooth: float | None = None,
    clip_grad: float | None = None,
) -> tuple[TrainState[FloatScalar], FloatDict]:
    """Update using binary cross entropy loss."""

    def get_bce_loss(params):
        apply_fn = ft.partial(logits_pred.apply_with, params=params)
        b_logits = jax.vmap(apply_fn)(b_obs)
        assert b_logits.shape[0] == b_ispos.shape[0] == b
        assert b_logits.shape == b_ispos.shape

        b_loss = optax.sigmoid_binary_cross_entropy(b_logits, b_label)
        loss = jnp.mean(b_loss)
        info = {f"{name}/Loss": loss}
        return loss, info

    if smooth is None:
        b_label = b_ispos
    else:
        # Apply label smoothing. pos: 1-smooth, neg: smooth.
        b_label = b_ispos * (1 - smooth) + (1 - b_ispos) * smooth

    # b = b_obs.shape[0]
    b = get_leading_dim_fast(b_obs)
    grads, info = jax.grad(get_bce_loss, has_aux=True)(logits_pred.params)

    grad_label = f"{name}/grad"
    if clip_grad is None:
        info[grad_label] = compute_norm(grads)
    else:
        grads, info[grad_label] = compute_norm_and_clip(grads, clip_grad)

    logits_pred_new = logits_pred.apply_gradients(grads=grads)
    return logits_pred_new, info


def update_policy_ppo(
    key_entropy: PRNGKey,
    policy: TrainState[tfd.Independent],
    b_obs,
    b_control,
    b_logprob_old,
    b_Al,
    clip_ratio: float,
    ent_cf: float,
    clip_grad: float | None = None,
):
    """
    Updates the policy using the Proximal Policy Optimization (PPO) algorithm.

    Args:
        key_entropy (PRNGKey): Random key for entropy calculations.
        policy (TrainState[tfd.Independent]): The current policy's train state.
        b_obs: Batch of observations.
        b_control: Batch of control actions.
        b_logprob_old: Batch of log probabilities of actions under the old policy.
        b_Al: Batch of advantages.
        clip_ratio (float): Clipping ratio for PPO.
        ent_cf (float): Entropy coefficient for regularization.
        clip_grad (float | None, optional): Gradient clipping threshold. If None, no clipping is applied.

    Returns:
        tuple:
            - TrainState[tfd.Independent]: The updated policy train state.
            - FloatDict: A dictionary containing information about the policy update process.
    """

    def get_pol_loss(pol_params):
        """
        Computes the policy loss and associated metrics.

        Args:
            pol_params: Parameters of the policy.

        Returns:
            tuple:
                - float: The computed policy loss.
                - dict: A dictionary containing metrics such as entropy, KL divergence, and clipping fraction.
        """
        pol_apply = ft.partial(policy.apply_with, params=pol_params)

        def get_logprob_entropy(graph, control, key_):
            """
            Computes the log probability and entropy for a given observation and control action.

            Args:
                graph: Observation input.
                control: Control action.
                key_: Random key for entropy calculation.

            Returns:
                tuple:
                    - float: Log probability of the control action.
                    - float: Entropy of the policy distribution.
            """
            dist = pol_apply(graph)
            if isinstance(dist, tfd.Categorical):
                entropy = dist.entropy()
            else:
                entropy = dist.entropy(seed=key_)

            return dist.log_prob(control), entropy

        # Compute log probabilities and entropy for the batch.
        b_logprobs, b_entropy = jax.vmap(get_logprob_entropy)(
            b_obs, b_control, key_entropy
        )
        b_logratios = b_logprobs - b_logprob_old
        b_is_ratio = jnp.exp(b_logratios)

        # Compute the clipped and unclipped policy gradient losses.
        pg_loss_orig = b_Al * b_is_ratio
        pg_loss_clip = b_Al * jnp.clip(b_is_ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss_pg = jnp.maximum(pg_loss_orig, pg_loss_clip).mean()
        pol_clipfrac = jnp.mean(pg_loss_clip > pg_loss_orig)

        # Compute additional metrics.
        logratios_max = b_logratios.max()
        logratios_q90 = jnp.quantile(b_logratios, 0.9)

        mean_entropy = b_entropy.mean()
        loss_entropy = -mean_entropy

        # Compute KL between old and new policy. Adjust lr if needed.
        #   KL( pi_old || pi_new )
        b_logprob_new = b_logprobs
        kl_old_new_ = jnp.mean(b_logprob_old - b_logprob_new)

        # Combine losses with entropy regularization.
        pol_loss = loss_pg + ent_cf * loss_entropy
        info = {
            "Pol/Loss": pol_loss,
            "Pol/entropy": mean_entropy,
            "Pol/clipfrac": pol_clipfrac,
            "Pol/kl_old_new": kl_old_new_,
            "Pol/logratio_max": logratios_max,
            "Pol/logratio_q90": logratios_q90,
        }
        return pol_loss, info

    # Compute gradients of the policy loss with respect to policy parameters.
    grads, pol_info = jax.grad(get_pol_loss, has_aux=True)(policy.params)

    # Compute gradient norms and optionally clip them.
    if clip_grad is None:
        pol_info["Pol/grad"] = compute_norm(grads)
    else:
        grads, pol_info["Pol/grad"] = compute_norm_and_clip(grads, clip_grad)

    # Apply the computed gradients to update the policy parameters.
    policy = policy.apply_gradients(grads=grads)
    return policy, pol_info

def update_policy_ppo_lag(
        key_entropy: PRNGKey,
        policy: TrainState[tfd.Independent],
        b_obs,
        b_control,
        b_logprob_old,
        b_Al,
        b_AlC,
        lam: float,  # Lagrange multiplier
        clip_ratio: float,
        ent_cf: float,
        clip_grad: float | None = None,
):
    """
    PPO Lagrangian.

    Args:
        key_entropy (PRNGKey): Random key for entropy calculations.
        policy (TrainState[tfd.Independent]): The current policy's train state.
        b_obs: Batch of observations.
        b_control: Batch of control actions.
        b_logprob_old: Batch of log probabilities of actions under the old policy.
        b_Al: Batch of advantages.
        clip_ratio (float): Clipping ratio for PPO.
        ent_cf (float): Entropy coefficient for regularization.
        clip_grad (float | None, optional): Gradient clipping threshold. If None, no clipping is applied.

    Returns:
        tuple:
            - TrainState[tfd.Independent]: The updated policy train state.
            - FloatDict: A dictionary containing information about the policy update process.
    """

    def get_pol_loss(pol_params):
        """
        Computes the policy loss and associated metrics.

        Args:
            pol_params: Parameters of the policy.

        Returns:
            tuple:
                - float: The computed policy loss.
                - dict: A dictionary containing metrics such as entropy, KL divergence, and clipping fraction.
        """
        pol_apply = ft.partial(policy.apply_with, params=pol_params)

        def get_logprob_entropy(graph, control, key_):
            """
            Computes the log probability and entropy for a given observation and control action.

            Args:
                graph: Observation input.
                control: Control action.
                key_: Random key for entropy calculation.

            Returns:
                tuple:
                    - float: Log probability of the control action.
                    - float: Entropy of the policy distribution.
            """
            dist = pol_apply(graph)
            if isinstance(dist, tfd.Categorical):
                entropy = dist.entropy()
            else:
                entropy = dist.entropy(seed=key_)

            return dist.log_prob(control), entropy

        # Compute log probabilities and entropy for the batch.
        b_logprobs, b_entropy = jax.vmap(get_logprob_entropy)(
            b_obs, b_control, key_entropy
        )
        b_logratios = b_logprobs - b_logprob_old
        b_is_ratio = jnp.exp(b_logratios)

        # Compute the clipped and unclipped policy gradient losses.
        pg_loss_orig = b_Al * b_is_ratio
        pg_loss_clip = b_Al * jnp.clip(b_is_ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss_pg = jnp.maximum(pg_loss_orig, pg_loss_clip).mean()
        pol_clipfrac = jnp.mean(pg_loss_clip > pg_loss_orig)

        # Compute the cost advantage policy gradient terms
        pg_loss_orig_C = b_AlC * b_is_ratio
        pg_loss_clip_C = b_AlC * jnp.clip(b_is_ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss_pg_C = jnp.maximum(pg_loss_orig_C, pg_loss_clip_C).mean()

        # Compute additional metrics.
        logratios_max = b_logratios.max()
        logratios_q90 = jnp.quantile(b_logratios, 0.9)

        mean_entropy = b_entropy.mean()
        loss_entropy = -mean_entropy

        # Compute KL between old and new policy. Adjust lr if needed.
        #   KL( pi_old || pi_new )
        b_logprob_new = b_logprobs
        kl_old_new_ = jnp.mean(b_logprob_old - b_logprob_new)

        # Combine losses with entropy regularization and Lagrangian cost term.
        pol_loss = loss_pg + ent_cf * loss_entropy + loss_pg_C / lam
        # # TODO
        # pol_loss = loss_pg + ent_cf * loss_entropy

        info = {
            "Pol/Loss": pol_loss,
            "Pol/entropy": mean_entropy,
            "Pol/clipfrac": pol_clipfrac,
            "Pol/kl_old_new": kl_old_new_,
            "Pol/logratio_max": logratios_max,
            "Pol/logratio_q90": logratios_q90,
        }
        return pol_loss, info

    # Compute gradients of the policy loss with respect to policy parameters.
    grads, pol_info = jax.grad(get_pol_loss, has_aux=True)(policy.params)

    # Compute gradient norms and optionally clip them.
    if clip_grad is None:
        pol_info["Pol/grad"] = compute_norm(grads)
    else:
        grads, pol_info["Pol/grad"] = compute_norm_and_clip(grads, clip_grad)

    # Apply the computed gradients to update the policy parameters.
    policy = policy.apply_gradients(grads=grads)
    return policy, pol_info
