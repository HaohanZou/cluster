# pylint: disable=pointless-string-statement,broad-exception-caught
# pyright: reportUnboundVariable=false, reportOptionalMemberAccess=false

import itertools
import time
from copy import deepcopy
from itertools import compress

from scipy import optimize
import sympy as sym
from scipy.special import logsumexp
from scipy.optimize import direct, Bounds, shgo
import multiprocessing as mp
import contextlib
import os
import sys
import math
from torch import nn
import sympytorch
import torch.optim.lr_scheduler as lr_scheduler
import csv
import timeit

import numpy as np
import scipy
import torch
from dso.memory import Batch, make_queue
from dso.program import Program, from_tokens
from dso.task import make_task
from dso.train_stats import StatsLogger
from dso.utils import empirical_entropy, get_duration, log_and_print, weighted_quantile
from dso.variance import quantile_variance
from nesymres.architectures.data_utils import (
    eq_remove_constants,
    eq_sympy_prefix_to_token_library,
)
from torch.multiprocessing import (  # pylint: disable=unused-import  # noqa: F401
    Pool,
    cpu_count,
    get_logger,
)
from torch.nn.utils.rnn import pad_sequence

# from datasets import generate_train_and_val_functions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = get_logger()


def process_raw_batch(raw_batch, controller):
    eqs = []
    eqs_valid = []
    for eq in raw_batch[1]:
        try:
            eq = torch.Tensor(controller.task.library.actionize(eq_sympy_prefix_to_token_library(eq)))
            eqs.append(eq)
            eqs_valid.append(True)
        except Exception:  # pylint: disable=broad-exception-caught
            eqs_valid.append(False)
    if np.array(eqs_valid).sum() == 0:
        return torch.Tensor([]), torch.Tensor([])
    eqs = pad_sequence(eqs, padding_value=controller.tgt_padding_token).T
    data = raw_batch[0][eqs_valid, :, :]
    return data, eqs


def train_controller(
    dltrain,
    dlval,
    dltest,
    controller,
    pool,
    gp_controller,
    output_file,
    pre_train,
    training_epochs,
    batch_outer_datasets,
    batch_inner_equations,
    load_pre_trained_path,
    config,
    controller_saved_path,
    n_epochs=None,
    n_samples=2000000,
    batch_size=1000,
    complexity="token",
    const_optimizer="scipy",
    const_params=None,
    alpha=0.5,
    epsilon=0.05,
    n_cores_batch=1,
    verbose=True,
    save_summary=False,
    save_all_epoch=False,
    baseline="R_e",
    b_jumpstart=False,
    early_stopping=True,
    hof=100,
    eval_all=False,
    save_pareto_front=True,
    debug=0,
    use_memory=False,
    memory_capacity=1e3,
    warm_start=None,
    memory_threshold=None,
    save_positional_entropy=False,
    save_top_samples_per_batch=0,
    save_cache=False,
    save_cache_r_min=0.9,
    save_freq=None,
    save_token_count=False,
    learning_rate=0.001,
    gradient_clip=1,
    patience=float("inf"),
):
    """
    Executes the main training loop.

    Parameters
    ----------
    controller : dso.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    gp_controller : dso.gp.gp_controller.GPController or None
        GP controller object used to generate Programs.

    output_file : str or None
        Path to save results each step.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity function name, used computing Pareto front.

    const_optimizer : str or None, optional
        Name of constant optimizer.

    const_params : dict, optional
        Dict of constant optimizer kwargs.

    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

    verbose : bool, optional
        Whether to print progress.

    save_summary : bool, optional
        Whether to write TensorFlow summaries.

    save_all_epoch : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    save_pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    use_memory : bool, optional
        If True, use memory queue for reward quantile estimation.

    memory_capacity : int
        Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional
        Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional
        Whether to store X% top-performer samples in every batch.

    save_cache : bool
        Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None
        Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

    save_token_count : bool
        Whether to save token counts each batch.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by reward).
    """
    best_val_loss = float("inf")
    waiting = 0
    best_model_state_dict = deepcopy(controller.state_dict())  # pyright: ignore

    try:
        run_gp_meld = gp_controller is not None

        # Config assertions and warnings
        assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

        # Create the priority queue
        k = controller.pqt_k
        if controller.pqt and k is not None and k > 0:
            priority_queue = make_queue(priority=True, capacity=k)
        else:
            priority_queue = None

        # Create the memory queue
        if use_memory:
            raise NotImplementedError
            # assert epsilon is not None and epsilon < 1.0, "Memory queue is only used with risk-seeking."
            # memory_queue = make_queue(controller=controller, priority=False, capacity=int(memory_capacity))

            # # Warm start the queue
            # warm_start = warm_start if warm_start is not None else batch_size
            # actions, obs, priors = controller.sample(warm_start)
            # # Overfitting - try to avoid
            # programs = [from_tokens(a) for a in actions]
            # r = np.array([p.r for p in programs])
            # l = np.array([len(p.traversal) for p in programs])
            # on_policy = np.array([p.originally_on_policy for p in programs])

            # X_train = torch.from_numpy(Program.task.X_train).to(torch.float32).cpu()
            # y_train = torch.from_numpy(Program.task.y_train).to(torch.float32).cpu()
            # sampled_data = torch.cat((X_train, y_train.view(-1, 1)), axis=1)
            # data_to_encode = sampled_data.tile(warm_start, 1, 1)

            # sampled_batch = Batch(
            #     actions=actions,
            #     obs=obs,
            #     priors=priors,
            #     lengths=l,
            #     rewards=r,
            #     on_policy=on_policy,
            #     data_to_encode=data_to_encode.detach().to_numpy(),
            # )
            # memory_queue.push_batch(sampled_batch, programs)
        else:
            memory_queue = None

        # For stochastic Tasks, store each reward computation for each unique traversal
        if Program.task.stochastic:
            r_history = {}  # Dict from Program str to list of rewards
            # It's not really clear whether Programs with const should enter the hof for stochastic Tasks
            assert Program.library.const_token is None, "Constant tokens not yet supported with stochastic Tasks."
            assert not save_pareto_front, "Pareto front not supported with stochastic Tasks."
        else:
            r_history = None

        # Main training loop
        p_final = None  # pylint: disable=unused-variable  # noqa: F841
        r_best = -np.inf
        prev_r_best = None  # pylint: disable=unused-variable  # noqa: F841
        ewma = None if b_jumpstart else 0.0  # EWMA portion of baseline
        n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
        nevals = 0  # Total number of sampled expressions (from RL or GP)
        positional_entropy = np.zeros(shape=(n_epochs, controller.max_length), dtype=np.float32)

        top_samples_per_batch = list()

        start_time = time.time()  # pylint: disable=unused-variable  # noqa: F841

        optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)

        for epoch in range(training_epochs):
            it = 0
            # i = 0
            # for raw_batch in dltrain:
            #     # print(raw_batch[0].shape[0])
            #     print(raw_batch[1][0])
            #     i += raw_batch[0].shape[0]
            # print('ALL', i)
            # continue
            try:
                for raw_batch in dltrain:
                    if raw_batch[0].nelement() == 0:
                        log_and_print("WARNING no data in batch skipping")
                        continue
                    # Set of str representations for all Programs ever seen
                    # s_history = set(r_history.keys()
                    #                 if Program.task.stochastic else Program.cache.keys())
                    t0 = time.time()
                    # data, eqs = process_raw_batch(raw_batch, controller)
                    # if data.nelement() == 0:
                    #     log_and_print('WARNING no data in batch filtered eqs skipping')
                    #     continue
                    # # sampled_data_raw = data.permute(0, 2, 1)
                    # batch_sos_tokens = torch.ones((1, data.shape[0]), dtype=torch.long).to(
                    #     DEVICE) * controller.sos_token
                    # token_eqs = eqs[:, :controller.max_length].to(
                    #     torch.long).to(DEVICE)
                    # tgt = torch.cat((batch_sos_tokens, token_eqs.T), 0)
                    # tgt = tgt.T.tile(batch_inner_equations,1).T

                    # sampled_data_cpu = data.cpu().numpy()
                    # sampled_data = data.cpu() # Could change this not to CPU - however can't change too many things at once
                    sampled_data_cpu = raw_batch[0].cpu().numpy()
                    sampled_data = raw_batch[
                        0
                    ].cpu()  # Could change this not to CPU - however can't change too many things at once
                    tgt = None
                    # tokens = raw_batch[1].to(DEVICE) + 1
                    # token_inputs = tokens.tile(batch_inner_equations, 1)
                    data_to_encode = sampled_data.permute(0, 2, 1).tile(batch_inner_equations, 1, 1)
                    actions, obs, priors = controller.sample(
                        data_to_encode.shape[0], data_to_encode.detach().to(DEVICE)
                    )
                    if actions.size == 0:
                        log_and_print("WARNING : actions shape {}".format(actions.shape))
                        continue

                    programs = []
                    for i in range(sampled_data_cpu.shape[0]):
                        data_xy = sampled_data_cpu[i].T
                        x = data_xy[:, :-1]
                        if x.shape[1] != (controller.encoder_input_dim - 1):
                            x = np.pad(data_xy[:, :-1], ((0, 0), (0, controller.encoder_input_dim - 1 - x.shape[1])))
                        data = (x, data_xy[:, -1])
                        programs.extend(
                            [
                                program_from_tokens_with_custom_data(
                                    a, deepcopy(config["task"]), data
                                )  # pyright: ignore
                                for a in actions[(i * batch_inner_equations) : (i + 1) * batch_inner_equations, :]
                            ]
                        )
                    nevals += data_to_encode.shape[0]

                    # Run GP seeded with the current batch, returning elite samples
                    if run_gp_meld:
                        deap_programs, deap_actions, deap_obs, deap_priors = gp_controller(actions)
                        nevals += gp_controller.nevals

                        # Combine RNN and deap programs, actions, obs, and priors
                        programs = programs + deap_programs
                        actions = np.append(actions, deap_actions, axis=0)
                        obs = np.append(obs, deap_obs, axis=0)
                        priors = np.append(priors, deap_priors, axis=0)

                    # Compute rewards in parallel
                    if pool is not None:
                        # Filter programs that need reward computing
                        programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
                        pool_p_dict = {p.str: p for p in pool.map(work, programs_to_optimize)}
                        programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
                        # Make sure to update cache with new programs
                        # Program.cache.update(pool_p_dict)

                    # Compute rewards (or retrieve cached rewards)
                    r = np.array([p.r for p in programs])
                    r_train = r
                    if r.size == 0:
                        log_and_print("WARNING : rewards are empty {}".format(actions.shape))
                        continue

                    # Back up programs to save them properly later
                    # controller_programs = programs.copy() if save_token_count else None

                    # Need for Vanilla Policy Gradient (epsilon = null)
                    p_train = programs

                    l = np.array([len(p.traversal) for p in programs])  # noqa: E741
                    # Str representations of Programs
                    s = [p.str for p in programs]
                    on_policy = np.array([p.originally_on_policy for p in programs])
                    invalid = np.array([p.invalid for p in programs], dtype=bool)
                    invalid_percent = (np.sum(invalid) / invalid.shape[0]) * 100

                    if save_positional_entropy:
                        positional_entropy[epoch] = np.apply_along_axis(empirical_entropy, 0, actions)

                    if save_top_samples_per_batch > 0:
                        # sort in descending order: larger rewards -> better solutions
                        sorted_idx = np.argsort(r)[::-1]
                        one_perc = int(len(programs) * float(save_top_samples_per_batch))
                        for idx in sorted_idx[:one_perc]:
                            top_samples_per_batch.append([epoch, r[idx], repr(programs[idx])])

                    if eval_all:
                        success = [p.evaluate.get("success") for p in programs]
                        # Check for success before risk-seeking, but don't break until after
                        if any(success):
                            p_final = programs[success.index(True)]  # pylint: disable=unused-variable  # noqa: F841

                    # Update reward history
                    if r_history is not None:
                        for p in programs:
                            key = p.str
                            if key in r_history:
                                r_history[key].append(p.r)
                            else:
                                r_history[key] = [p.r]

                    # Store in variables the values for the whole batch (those variables will be modified below)
                    r_full = r  # pylint: disable=unused-variable  # noqa: F841
                    l_full = l  # pylint: disable=unused-variable  # noqa: F841
                    s_full = s  # pylint: disable=unused-variable  # noqa: F841
                    actions_full = actions  # pylint: disable=unused-variable  # noqa: F841
                    invalid_full = invalid  # pylint: disable=unused-variable  # noqa: F841
                    r_max = np.max(r)
                    r_best = max(r_max, r_best)

                    """
                    Apply risk-seeking policy gradient: compute the empirical quantile of
                    rewards and filter out programs with lesser reward.
                    """
                    if epsilon is not None and epsilon < 1.0:  # Empirical quantile
                        # Compute reward quantile estimate
                        if use_memory:  # Memory-augmented quantile
                            # Get subset of Programs not in buffer
                            unique_programs = [
                                p for p in programs if p.str not in memory_queue.unique_items  # pyright: ignore
                            ]
                            N = len(unique_programs)

                            # Get rewards
                            memory_r = memory_queue.get_rewards()  # pyright: ignore
                            sample_r = [p.r for p in unique_programs]
                            combined_r = np.concatenate([memory_r, sample_r])

                            # Compute quantile weights
                            memory_w = memory_queue.compute_probs()  # pyright: ignore
                            if N == 0:
                                print("WARNING: Found no unique samples in batch!")
                                combined_w = memory_w / memory_w.sum()  # Renormalize
                            else:
                                sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                                combined_w = np.concatenate([memory_w, sample_w])

                            # Quantile variance/bias estimates
                            if memory_threshold is not None:
                                print("Memory weight:", memory_w.sum())
                                if memory_w.sum() > memory_threshold:
                                    quantile_variance(memory_queue, controller, batch_size, epsilon, epoch)

                            # Compute the weighted quantile
                            quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - epsilon)

                        else:  # Empirical quantile
                            # quantile = np.quantile(
                            #     r, 1 - epsilon, interpolation="higher")
                            quantiles = [
                                np.quantile(  # pyright: ignore
                                    r[batch_inner_equations * i : batch_inner_equations * (i + 1)],
                                    1 - epsilon,
                                    interpolation="higher",
                                )
                                for i in range(int(np.floor(r.shape[0] / batch_inner_equations)))
                            ]
                            quantile = np.mean(quantiles)

                        r_raw_sum = np.sum(r)
                        r_raw_mean = r.sum() / (r != 0).sum()

                        # These guys can contain the GP solutions if we run GP
                        """
                            Here we get the returned as well as stored programs and properties.

                            If we are returning the GP programs to the controller, p and r will be exactly the same
                            as p_train and r_train. Otherwise, p and r will still contain the GP programs so they
                            can still fall into the hall of fame. p_train and r_train will be different and no longer
                            contain the GP program items.
                        """

                        # keep = r >= quantile
                        keep = np.concatenate(
                            [
                                r[batch_inner_equations * i : batch_inner_equations * (i + 1)] >= quantiles[i]
                                for i in range(int(np.floor(r.shape[0] / batch_inner_equations)))
                            ]
                        )
                        l = l[keep]  # noqa: E741
                        s = list(compress(s, keep))  # pyright: ignore
                        invalid = invalid[keep]

                        # Option: don't keep the GP programs for return to controller
                        if run_gp_meld and not gp_controller.return_gp_obs:
                            """
                            If we are not returning the GP components to the controller, we will remove them from
                            r_train and p_train by augmenting 'keep'. We just chop off the GP elements which are indexed
                            from batch_size to the end of the list.
                            """
                            _r = r[keep]
                            _p = list(compress(programs, keep))
                            keep[batch_size:] = False
                            r_train = r[keep]
                            p_train = list(compress(programs, keep))  # pyright: ignore

                            """
                                These contain all the programs and rewards regardless of whether they are returned to the controller.
                                This way, they can still be stored in the hall of fame.
                            """
                            r = _r
                            programs = _p
                        else:
                            """
                            Since we are returning the GP programs to the contorller, p and r are the same as p_train and r_train.
                            """
                            r_train = r = r[keep]
                            p_train = programs = list(compress(programs, keep))  # pyright: ignore

                        """
                            get the action, observation, priors and on_policy status of all programs returned to the controller.
                        """
                        actions = actions[keep, :]
                        obs = obs[keep, :, :]
                        priors = priors[keep, :, :]
                        on_policy = on_policy[keep]
                    else:
                        keep = None

                    r_quantile_sum = np.sum(r)
                    # Clip bounds of rewards to prevent NaNs in gradient descent
                    r = np.clip(r, -1e6, 1e6)
                    r_train = np.clip(r_train, -1e6, 1e6)

                    # Compute baseline
                    # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
                    if baseline == "ewma_R":
                        ewma = np.mean(r_train) if ewma is None else alpha * np.mean(r_train) + (1 - alpha) * ewma
                        b_train = ewma
                    elif baseline == "R_e":  # Default
                        ewma = -1
                        b_train = quantile  # pyright: ignore
                    elif baseline == "ewma_R_e":
                        ewma = (
                            np.min(r_train) if ewma is None else alpha * quantile + (1 - alpha) * ewma
                        )  # pyright: ignore
                        b_train = ewma
                    elif baseline == "combined":
                        ewma = (
                            np.mean(r_train) - quantile  # pyright: ignore
                            if ewma is None
                            else alpha * (np.mean(r_train) - quantile) + (1 - alpha) * ewma  # pyright: ignore
                        )
                        b_train = quantile + ewma  # pyright: ignore

                    # Compute sequence lengths
                    lengths = np.array([min(len(p.traversal), controller.max_length) for p in p_train], dtype=np.int32)

                    if data_to_encode is not None:
                        if run_gp_meld:
                            data_to_encode_train = torch.cat(
                                [
                                    data_to_encode.detach(),
                                    data_to_encode.detach()[-1, :, :].tile(config["gp_meld"]["train_n"], 1, 1),
                                ]
                            )
                            if keep is not None:
                                data_to_encode_train = data_to_encode_train[keep, :, :].cpu().numpy()
                            else:
                                data_to_encode_train = data_to_encode_train[:, :, :].cpu().numpy()
                        elif keep is not None:
                            data_to_encode_train = data_to_encode[keep, :, :].detach().cpu().numpy()
                        else:
                            data_to_encode_train = data_to_encode[:, :, :].detach().cpu().numpy()
                    else:
                        data_to_encode_train = np.array([])

                    if tgt is not None:
                        if run_gp_meld:
                            tgt_to_encode_train = torch.cat(
                                [tgt.T.detach(), tgt.T.detach()[-1, :].tile(config["gp_meld"]["train_n"], 1)]
                            ).T
                            if keep is not None:
                                tgt_to_encode_train = tgt_to_encode_train[:, keep].cpu().numpy()
                            else:
                                tgt_to_encode_train = tgt_to_encode_train[:, :].cpu().numpy()
                        elif keep is not None:
                            tgt_to_encode_train = tgt[:, keep].detach().cpu().numpy()
                        else:
                            tgt_to_encode_train = tgt[:, :].detach().cpu().numpy()
                    else:
                        tgt_to_encode_train = np.array([])

                    # Create the Batch
                    sampled_batch = Batch(
                        actions=actions,
                        obs=obs,
                        priors=priors,
                        lengths=lengths,
                        rewards=r_train,
                        on_policy=on_policy,
                        data_to_encode=data_to_encode_train,
                        tgt=tgt_to_encode_train,
                    )

                    # Update and sample from the priority queue
                    if priority_queue is not None:
                        priority_queue.push_best(sampled_batch, programs)
                        pqt_batch = priority_queue.sample_batch(controller.pqt_batch_size)
                    else:
                        pqt_batch = None

                    # Train the controller
                    # if pre_train:
                    #     data_to_encode_all = sampled_data.detach()
                    # else:
                    #     data_to_encode_all = data_to_encode
                    try:
                        controller.train()
                        optimizer.zero_grad()
                        loss, mle_loss = controller.train_loss(b_train, sampled_batch, pqt_batch)  # pyright: ignore
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip)  # pyright: ignore
                        optimizer.step()
                        train_loss = loss.item()
                        # Look at val loss
                        controller.eval()
                        val_loss = compute_val_loss(
                            dlval,
                            batch_inner_equations,
                            controller,
                            run_gp_meld,
                            config,
                            gp_controller,
                            pool,
                            eval_all,
                            epsilon,
                            baseline,
                            alpha,
                        )
                        if epsilon is not None:
                            log_and_print(
                                f"[epoch={epoch+1:04d}|iter={it+1:04d}] train_loss={train_loss:.5f} \t| "
                                f"eqs_invalid %={invalid_percent:.2f} \t|  quantile={quantile:.5f} \t| "
                                f"r_quantile_sum={r_quantile_sum:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t| "
                                f"r_raw_val_sum={1/val_loss:.5f} \t| r_raw_mean={r_raw_mean:.5f} \t| "
                                f"s/it={time.time() - t0:.5f}"
                            )
                        else:
                            if mle_loss is not None:
                                log_and_print(
                                    f"[epoch={epoch+1:04d}|iter={it+1:04d}] train_loss={train_loss:.5f} \t| "
                                    f"mle_loss={mle_loss:.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| "
                                    f"r_raw_sum={np.sum(r):.5f} \t| r_raw_val_sum={1/val_loss:.5f} \t| "
                                    f"r_raw_mean={np.mean(r):.5f} \t| s/it={time.time() - t0:.5f}"
                                )
                            else:
                                log_and_print(
                                    f"[epoch={epoch+1:04d}|iter={it+1:04d}] train_loss={train_loss:.5f} \t| "
                                    f"eqs_invalid %={invalid_percent:.2f} \t| r_raw_sum={np.sum(r):.5f} \t| "
                                    f"r_raw_val_sum={1/val_loss:.5f} \t| r_raw_mean={np.mean(r):.5f} \t| "
                                    f"s/it={time.time() - t0:.5f}"
                                )
                        it += 1
                        torch.save(
                            best_model_state_dict, controller_saved_path.replace("controller", "controller_training")
                        )

                        # Early stopping procedure
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            log_and_print(f"[New best val loss model - saving]: reward_sum={1/val_loss:.5f}")
                            best_model_state_dict = deepcopy(controller.state_dict())  # pyright: ignore
                            torch.save(best_model_state_dict, controller_saved_path)
                            waiting = 0
                        elif waiting > patience:
                            break
                        else:
                            waiting += 1
                    except Exception as e:
                        log_and_print("[ERROR] Error {}".format(str(e)))
                        controller_saved_path = controller_saved_path.replace("controller", "controller_errored")
                        continue
            except AssertionError as e:
                log_and_print("[ERROR] Error {}".format(str(e)))
                controller_saved_path = controller_saved_path.replace("controller", "controller_errored")
                continue
    except KeyboardInterrupt:
        log_and_print("Interrupted training")
        torch.save(controller.state_dict(), controller_saved_path)
    return controller


def optomize_at_test(
    controller,
    pool,
    gp_controller,
    output_file,
    pre_train,
    config,
    controller_saved_path,
    n_epochs=None,
    n_samples=2000000,
    batch_size=1000,
    complexity="token",
    const_optimizer="scipy",
    const_params=None,
    alpha=0.5,
    epsilon=0.05,
    n_cores_batch=1,
    verbose=True,
    save_summary=False,
    save_all_epoch=False,
    baseline="R_e",
    b_jumpstart=False,
    early_stopping=True,
    hof=100,
    eval_all=False,
    save_pareto_front=True,
    debug=0,
    use_memory=False,
    memory_capacity=1e3,
    warm_start=None,
    memory_threshold=None,
    save_positional_entropy=False,
    save_top_samples_per_batch=0,
    save_cache=False,
    save_cache_r_min=0.9,
    save_freq=None,
    save_token_count=False,
    gradient_clip=1,
    save_true_log_likelihood=False,
    true_action=None,
):
    """
    Executes the main training loop.

    Parameters
    ----------
    controller : dso.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    gp_controller : dso.gp.gp_controller.GPController or None
        GP controller object used to generate Programs.

    output_file : str or None
        Path to save results each step.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity function name, used computing Pareto front.

    const_optimizer : str or None, optional
        Name of constant optimizer.

    const_params : dict, optional
        Dict of constant optimizer kwargs.

    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

    verbose : bool, optional
        Whether to print progress.

    save_summary : bool, optional
        Whether to write TensorFlow summaries.

    save_all_epoch : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    save_pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    use_memory : bool, optional
        If True, use memory queue for reward quantile estimation.

    memory_capacity : int
        Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional
        Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional
        Whether to store X% top-performer samples in every batch.

    save_cache : bool
        Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None
        Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

    save_token_count : bool
        Whether to save token counts each batch.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by reward).
    """
    controller_saved_path = controller_saved_path.replace("controller", "controller_test")
    quantile_over_fit_times_limit = float("inf")  # This could help it ?!
    quantile_over_fit_times = 0
    run_gp_meld = gp_controller is not None
    controller.rl_weight = 1.0
    eval_all = True

    total_unique_set = set()

    # Config assertions and warnings
    assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

    # Create the priority queue
    k = controller.pqt_k
    if controller.pqt and k is not None and k > 0:
        priority_queue = make_queue(priority=True, capacity=k)
    else:
        priority_queue = None

    # Create the memory queue
    if use_memory:
        assert epsilon is not None and epsilon < 1.0, "Memory queue is only used with risk-seeking."
        memory_queue = make_queue(controller=controller, priority=False, capacity=int(memory_capacity))

        # Warm start the queue
        warm_start = warm_start if warm_start is not None else batch_size
        actions, obs, priors = controller.sample(warm_start)
        programs = [from_tokens(a) for a in actions]
        r = np.array([p.r for p in programs])
        l = np.array([len(p.traversal) for p in programs])  # noqa: E741
        on_policy = np.array([p.originally_on_policy for p in programs])

        X_train = torch.from_numpy(Program.task.X_train).to(torch.float32).cpu()  # pyright: ignore
        y_train = torch.from_numpy(Program.task.y_train).to(torch.float32).cpu()  # pyright: ignore
        sampled_data = torch.cat((X_train, y_train.view(-1, 1)), axis=1)  # pyright: ignore
        data_to_encode = sampled_data.tile(warm_start, 1, 1)

        sampled_batch = Batch(  # pyright: ignore
            actions=actions,
            obs=obs,
            priors=priors,
            lengths=l,
            rewards=r,
            on_policy=on_policy,
            data_to_encode=data_to_encode.detach().to_numpy(),
        )
        memory_queue.push_batch(sampled_batch, programs)
    else:
        memory_queue = None

    # For stochastic Tasks, store each reward computation for each unique traversal
    if Program.task.stochastic:  # pyright: ignore
        r_history = {}  # Dict from Program str to list of rewards
        # It's not really clear whether Programs with const should enter the hof for stochastic Tasks
        assert (
            Program.library.const_token is None  # pyright: ignore
        ), "Constant tokens not yet supported with stochastic Tasks."
        assert not save_pareto_front, "Pareto front not supported with stochastic Tasks."
    else:
        r_history = None

    # Main training loop
    p_final = None
    r_best = -np.inf
    prev_r_best = None
    ewma = None if b_jumpstart else 0.0  # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
    nevals = 0  # Total number of sampled expressions (from RL or GP)
    positional_entropy = np.zeros(shape=(n_epochs, controller.max_length), dtype=np.float32)

    top_samples_per_batch = list()

    logger = StatsLogger(
        output_file,
        save_summary,
        save_all_epoch,
        hof,
        save_pareto_front,
        save_positional_entropy,
        save_top_samples_per_batch,
        save_cache,
        save_cache_r_min,
        save_freq,  # pyright: ignore
        save_token_count,
    )

    start_time = time.time()
    if verbose:
        print("-- RUNNING EPOCHS START -------------")

    optimizer = torch.optim.Adam(controller.parameters(), lr=controller.learning_rate)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    result = {"best": [], "quantile": [], "baseline": []}

    if pre_train:
        # X_train = torch.from_numpy(Program.task.X_train).to(torch.float32).to(DEVICE)
        # y_train = torch.from_numpy(Program.task.y_train).to(torch.float32).to(DEVICE)
        # sampled_data = torch.cat((X_train, y_train.view(-1, 1)), axis=1)  # pyright: ignore
        # data_to_encode = sampled_data.tile(batch_size, 1, 1)
        data_to_encode = prepare_encoder_input()
        data_to_encode = data_to_encode.long().to(DEVICE)
        data_to_encode = data_to_encode.tile(batch_size, 1, 1)
        data_to_encode = data_to_encode.squeeze(dim = 1)
        print(data_to_encode.shape)
    else:
        data_to_encode = None
    gp_collection = []
    for epoch in range(n_epochs):
        t0 = time.time()

        # Set of str representations for all Programs ever seen
        s_history = set(r_history.keys() if Program.task.stochastic else Program.cache.keys())

        # Sample batch of Programs from the Controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)
        actions, obs, priors = controller.sample(batch_size, data_to_encode)
        programs = [from_tokens(a) for a in actions]
        nevals += batch_size

        ## GP elite samples will be added to policy gradient

        # Run GP seeded with the current batch, returning elite samples
        Program.task.X_train, Program.task.y_train= torch.from_numpy(Program.task.X_train).to("cuda"), torch.from_numpy(Program.task.y_train).to("cuda")
        start = timeit.default_timer() 
        
        if run_gp_meld:
            deap_programs, deap_actions, deap_obs, deap_priors = gp_controller(actions)
            nevals += gp_controller.nevals

            # Combine RNN and deap programs, actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)
        end = timeit.default_timer()

        ## Aggregate deap result
        
        if len(gp_collection) == 0:
            gp_collection = deap_programs
            gp_collection_actions = deap_actions
            gp_collection_obs = deap_obs
            gp_collection_priors = deap_priors
        else:
            gp_collection = gp_collection + deap_programs
            gp_collection_actions = np.append(gp_collection_actions, deap_actions, axis = 0)
            gp_collection_obs = np.append(gp_collection_obs, deap_obs, axis = 0)
            gp_collection_priors = np.append(gp_collection_priors, deap_priors, axis = 0)
        
        print(end - start)
        pool = None
        Program.task.X_train, Program.task.y_train = Program.task.X_train.cpu().detach().numpy(), Program.task.y_train.cpu().detach().numpy()
        # Compute rewards in parallel
        if pool is not None:
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)
        
        temp = Program.task.X_train
        
        if (epoch > (-1)):
            if True:
            ## pgd collection for each program
                print("\n================PGD Collection Starts=========================\n")
                pgd_initial = torch.rand(Program.task.X_train.shape[0] // 10, Program.task.X_train.shape[1])
                with mp.Pool(processes= 2, initializer=init_worker) as pool:
                    pgd_output = pool.starmap(pgd_check, [(i.sympy_expr[0], pgd_initial) for i in programs if (not i.invalid)])

                pgd_example = []
                for j in pgd_output:
                    if len(j[1]) > 0:
                        if len(pgd_example) > 0:
                            pgd_example = np.concatenate((pgd_example, j[1]), axis = 0)
                        else:
                            pgd_example = j[1]

                ratio = min((epoch) * 0.005 + 0.1, 0.3)
                if len(pgd_example) > Program.task.X_train.shape[0] * ratio:
                    random_ind = np.random.randint(0, len(pgd_example), int(np.floor(Program.task.X_train.shape[0] * ratio)))
                    print(random_ind.shape)
                    pgd_example = pgd_example[random_ind]
                if len(pgd_example) > 0:
                    Program.task.X_train = np.vstack((Program.task.X_train, pgd_example))
                    Program.task.y_train = np.zeros(Program.task.X_train.shape[0])
                print(f"\n==============Collect {len(pgd_example)} Examples ===================\n")
        
        
        # Compute rewards (or retrieve cached rewards)
        Program.task.X_train, Program.task.y_train= torch.from_numpy(Program.task.X_train).to("cuda"), torch.from_numpy(Program.task.y_train).to("cuda")
        # r = np.array([p.r for p in programs])
        start = timeit.default_timer()
        r = np.array([p.update_r() for p in programs])
        r_train = r
        end = timeit.default_timer()
        print(end - start)
        
        Program.task.X_train = temp
        Program.task.y_train = np.zeros(Program.task.X_train.shape[0])
        # Program.task.X_train, Program.task.y_train = Program.task.X_train.cpu().detach().numpy(), Program.task.y_train.cpu().detach().numpy()
        
        # Back up programs to save them properly later
        controller_programs = programs.copy() if save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train = programs

        l = np.array([len(p.traversal) for p in programs])  # noqa: E741
        s = [p.str for p in programs]  # Str representations of Programs
        on_policy = np.array([p.originally_on_policy for p in programs])
        invalid = np.array([p.invalid for p in programs], dtype=bool)
        invalid_percent = (np.sum(invalid) / invalid.shape[0]) * 100

        if save_positional_entropy:
            positional_entropy[epoch] = np.apply_along_axis(empirical_entropy, 0, actions)

        if save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            one_perc = int(len(programs) * float(save_top_samples_per_batch))
            for idx in sorted_idx[:one_perc]:
                top_samples_per_batch.append([epoch, r[idx], repr(programs[idx])])
                
        '''         
        if eval_all:
            success = [p.evaluate.get("success") for p in programs]  # pyright: ignore
            # for idx, suc in enumerate(success):
            #     if suc:
            #         p = programs[idx]
            #         traverasl = p.traversal.copy()
            #         key = ','.join([str(i) for i in traverasl])
            #         total_unique_set.add(key)

            # Check for success before risk-seeking, but don't break until after
            if any(success):
                p_final = programs[success.index(True)]
        '''
        # Update reward history
        if r_history is not None:
            for p in programs:
                key = p.str
                if key in r_history:
                    r_history[key].append(p.r)
                else:
                    r_history[key] = [p.r]

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)
        r_best = max(r_max, r_best)
        r_raw_sum = np.sum(r)
        r_raw_mean = r.sum() / (r != 0).sum()

        epsilon = max(0.05, 0.2 - int(epoch // 5) * 0.025)
        epsilon_r = 0.05

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        print("Epsilon",epsilon)
        if epsilon is not None and epsilon < 1.0:
            # Compute reward quantile estimate
            if use_memory:  # Memory-augmented quantile
                # Get subset of Programs not in buffer
                unique_programs = [p for p in programs if p.str not in memory_queue.unique_items]
                N = len(unique_programs)

                # Get rewards
                memory_r = memory_queue.get_rewards()
                sample_r = [p.r for p in unique_programs]
                combined_r = np.concatenate([memory_r, sample_r])

                # Compute quantile weights
                memory_w = memory_queue.compute_probs()
                if N == 0:
                    print("WARNING: Found no unique samples in batch!")
                    combined_w = memory_w / memory_w.sum()  # Renormalize
                else:
                    sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                    combined_w = np.concatenate([memory_w, sample_w])

                # Quantile variance/bias estimates
                if memory_threshold is not None:
                    print("Memory weight:", memory_w.sum())
                    if memory_w.sum() > memory_threshold:
                        quantile_variance(memory_queue, controller, batch_size, epsilon, epoch)

                # Compute the weighted quantile
                quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - epsilon)

            else:  # Empirical quantile
                quantile = np.quantile(r, 1 - epsilon, interpolation="higher")  # pyright: ignore
                quantile_ver = np.quantile(r, 1 - epsilon_r , interpolation="higher")

            # These guys can contain the GP solutions if we run GP
            """
                Here we get the returned as well as stored programs and properties.

                If we are returning the GP programs to the controller, p and r will be exactly the same
                as p_train and r_train. Otherwise, p and r will still contain the GP programs so they
                can still fall into the hall of fame. p_train and r_train will be different and no longer
                contain the GP program items.
            """

            keep = r >= quantile
            keep_c = r >= max(quantile, -101.99)
            keep_ver = r >= quantile_ver
            l = l[keep]  # noqa: E741
            s = list(compress(s, keep))  # pyright: ignore
            invalid = invalid[keep]

            # Option: don't keep the GP programs for return to controller
            # gp_controller.return_gp_obs = False
            if run_gp_meld and not gp_controller.return_gp_obs:
                """
                If we are not returning the GP components to the controller, we will remove them from
                r_train and p_train by augmenting 'keep'. We just chop off the GP elements which are indexed
                from batch_size to the end of the list.
                """
                _r = r[keep]
                _p = list(compress(programs, keep))  # pyright: ignore
                print(batch_size)
                print(keep.shape)
                print(r.shape)
                print(np.sum(keep))
                keep[batch_size:] = False
                print(np.sum(keep))
                r_train = r[keep]
                p_train = list(compress(programs, keep))  # pyright: ignore
                print(r_train.shape)

                """
                    These contain all the programs and rewards regardless of whether they are returned to the controller.
                    This way, they can still be stored in the hall of fame.
                """
                r = _r
                programs = _p
                print("Running GP")
            else:
                print("Right Direction")
                """
                Since we are returning the GP programs to the contorller, p and r are the same as p_train and r_train.
                """
                print(batch_size)
                print(keep.shape)
                print(r.shape)
                r_train_ver = r[keep_ver]
                r_train = r = r[keep]
                # p_train_c = list(compress(programs, keep_c))
                p_train_ver = list(compress(programs, keep_ver))
                p_train = programs = list(compress(programs, keep))  # pyright: ignore
                print(r_train.shape)
                # print(len(p_train_c))
            
            
            """
                get the action, observation, priors and on_policy status of all programs returned to the controller.
            """
            actions = actions[keep, :]
            obs = obs[keep, :, :]
            priors = priors[keep, :, :]
            on_policy = on_policy[keep]
        else:
            keep = None
        
        ## counter_example feedback
        
        counter_example = np.array([])
        count = 0
        result['best'].append(r_max)
        result['quantile'].append(quantile)


        print(f"Number of Programs for Minimization: {len(p_train_ver)}")
        with mp.Pool(processes= mp.cpu_count(), initializer=init_worker) as pool:
            counter_example_splits = pool.map(check_options, [i.sympy_expr[0] for i in p_train_ver])
            # counter_example_splits = pool.starmap(pgd_check, [(i.sympy_expr[0], Program.task.X_test) for i in p_train])

        pool = None
        print(len(counter_example_splits))
        for j in counter_example_splits:
            if len(j[1]) > 0:
                if len(counter_example) > 0:
                    # print(j[1].shape)
                    count += j[1].shape[0]
                    # print(count)
                    counter_example = np.concatenate((counter_example, j[1]), axis = 0)
                else:
                    counter_example = j[1]


        if len(counter_example) > 80000:
            random_ind = np.random.randint(0, len(counter_example), 80000)
            counter_example = counter_example[random_ind]
        
        if len(counter_example) > 0:
            
            Program.task.X_test = np.vstack((Program.task.X_test, counter_example))

            if  len(Program.task.X_test) > 160000:
                random_ind = np.random.randint(0, Program.task.X_test.shape[0], 160000)
                Program.task.X_test = Program.task.X_test[random_ind]

        Program.task.y_test = np.zeros(Program.task.X_test.shape[0])

        '''
        if epoch % 5 == 0:
            Program.task.X_train = Program.task.X_test
            Program.task.y_train = Program.task.y_test
        '''
        if epoch % 1 == 0:

            if epoch == 0:
                Program.task.X_train = Program.task.X_test
                Program.task.y_train = Program.task.y_test
            
            else:
                portion = (min((epoch - (-1)) * 0.004, 0.12))
                
                add_size = int(portion * Program.task.X_test.shape[0])
                add_part_ind = np.random.randint(0, Program.task.X_test.shape[0], add_size)
                add_part = Program.task.X_test[add_part_ind]
                
                keep_size = int((1 - portion) * Program.task.X_test.shape[0])
                keep_part_ind = np.random.randint(0, Program.task.X_train.shape[0], keep_size)
                keep_part = Program.task.X_train[keep_part_ind]
                
                Program.task.X_train = np.vstack((keep_part, add_part))
                Program.task.y_train = np.zeros(Program.task.X_train.shape[0])

        print(Program.task.X_train.shape)
        print(Program.task.y_train.shape)

        print(Program.task.X_test.shape)
        print(Program.task.y_test.shape)

        
        
        
        print(f"\n===================Successfully Added {counter_example.shape} Counter Examples=======================\n")

        del counter_example

        if eval_all:
            success = [p.evaluate.get("success") for p in programs]  # pyright: ignore
            if any(success):
                minimizer_check = [check_options(programs[i].sympy_expr[0] * 1e5)[0] if success[i] else False for i in range(len(success))]
                overall_check = [final_check(programs[i].sympy_expr[0]) if minimizer_check[i] else False for i in range(len(success))]
                success = overall_check
                if any(success) > 0:
                    p_final = programs[success.index(True)]
                    print(f"Number of valid equations: {np.sum(success)}. Program should stopped")
                    print(p_final.sympy_expr[0])
                    print([programs[i].sympy_expr[0] for i in range(len(success)) if success[i]])

                    log_and_print(f"Number of valid equations: {np.sum(success)}. Program should stopped")
                    log_and_print(p_final.sympy_expr[0])
                    log_and_print([programs[i].sympy_expr[0] for i in range(len(success)) if success[i]])
                else:
                    print(f"Number of potential valid equations: {np.sum(success)}")
            else:
                print(f"\n============None of the candidates fulfill the conditions. Training CONTINUES!=================\n")
                log_and_print(f"\n============None of the candidates fulfill the conditions. Training CONTINUES!=================\n")
        
        
        
        r_quantile_mean = np.mean(r)
        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)
        r_train = np.clip(r_train, -1e6, 1e6)

        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if baseline == "ewma_R":
            ewma = np.mean(r_train) if ewma is None else alpha * np.mean(r_train) + (1 - alpha) * ewma
            b_train = ewma
        elif baseline == "R_e":  # Default
            ewma = -1
            b_train = quantile
        elif baseline == "ewma_R_e":
            ewma = np.min(r_train) if ewma is None else alpha * quantile + (1 - alpha) * ewma
            b_train = ewma
        elif baseline == "combined":
            ewma = (
                np.mean(r_train) - quantile
                if ewma is None
                else alpha * (np.mean(r_train) - quantile) + (1 - alpha) * ewma
            )
            b_train = quantile + ewma
        result['baseline'].append(b_train)
        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), controller.max_length) for p in p_train], dtype=np.int32)
        # run_gp_meld = False
        if data_to_encode is not None:
            if run_gp_meld:
                data_to_encode_train = torch.cat(
                    [
                        data_to_encode.detach(),
                        data_to_encode.detach()[-1, :].tile(config["gp_meld"]["train_n"], 1),
                    ]
                )
                if keep is not None:
                    data_to_encode_train = data_to_encode_train[keep, :].cpu().numpy()
                else:
                    data_to_encode_train = data_to_encode_train[:, :].cpu().numpy()
            elif keep is not None:
                data_to_encode_train = data_to_encode[keep, :].detach().cpu().numpy()
            else:
                data_to_encode_train = data_to_encode[:, :].detach().cpu().numpy()
            print(data_to_encode_train.shape)
        else:
            data_to_encode_train = np.array([])
        tgt_train = np.array([])
        # run_gp_meld = True
        # Create the Batch
        sampled_batch = Batch(
            actions=actions,
            obs=obs,
            priors=priors,
            lengths=lengths,
            rewards=r_train,
            on_policy=on_policy,
            data_to_encode=data_to_encode_train,
            tgt=tgt_train,
        )

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.push_best(sampled_batch, programs)
            pqt_batch = priority_queue.sample_batch(controller.pqt_batch_size)
        else:
            pqt_batch = None

        if save_true_log_likelihood and true_action is not None:
            if pre_train:
                nll = controller.compute_neg_log_likelihood(sampled_data.tile(1, 1, 1).detach(), true_action).item()
            else:
                nll = controller.compute_neg_log_likelihood(None, true_action, sampled_batch).item()

        # Train the controller
        controller.train()
        optimizer.zero_grad()
        loss, summaries = controller.train_loss(b_train, sampled_batch, pqt_batch, test=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            controller.parameters(), gradient_clip)
        optimizer.step()
        # scheduler.step()
        # print(
        #     f'[epoch={epoch+1:04d}] train_loss={loss:.5f}')

        # wall time calculation for the epoch
        epoch_walltime = time.time() - start_time

        torch.save(controller.state_dict(), controller_saved_path)

        print("\n==============Start Supervised Learning==============\n")

        Program.task.X_train, Program.task.y_train= torch.from_numpy(Program.task.X_train).to("cuda"), torch.from_numpy(Program.task.y_train).to("cuda")
        ## Supervised Learning from elite programs  

        ## freeze encoder parameter
        # for param in controller.model.dym_embedded_encoder.parameters():
            # param.requires_grad = False

        # for param in controller.model.dym_reduce_dim.parameters():
            # param.requires_grad = False

        ## Sort gp programs by reward and select the top 50 programs
        
        gp_collection_reward = np.array([gp_p.r for gp_p in gp_collection])
        gp_ratio = config["gp_meld"]["train_n"] / len(gp_collection)
        gp_quantile = np.quantile(gp_collection_reward, 1 - gp_ratio, interpolation="higher")  # pyright: ignore
        gp_keep = gp_collection_reward >= gp_quantile

        gp_collection = list(compress(gp_collection, gp_keep))
        print(np.mean(np.array([gp_p.r for gp_p in gp_collection])))
        gp_collection_actions = gp_collection_actions[gp_keep]
        gp_collection_obs = gp_collection_obs[gp_keep]
        gp_collection_priors = gp_collection_priors[gp_keep]
        gp_collection_reward = gp_collection_reward[gp_keep]
        
        ## optimize decoder structure
        for super in range(7):
            
            
            deap_lengths = np.array([min(len(p.traversal), controller.max_length) for p in gp_collection], dtype=np.int32)
            deap_r = np.array([p.r for p in gp_collection])
            deap_on_policy = np.array([p.originally_on_policy for p in gp_collection])
            deap_data_to_encode_train = data_to_encode.detach()[-1, :].tile(len(gp_collection), 1).cpu().numpy()
            deap_tgt_train = np.array([])
            
            '''
            deap_lengths = np.array([min(len(p.traversal), controller.max_length) for p in deap_programs], dtype=np.int32)
            deap_r = np.array([p.r for p in deap_programs])
            deap_on_policy = np.array([p.originally_on_policy for p in deap_programs])
            deap_data_to_encode_train = data_to_encode.detach()[-1, :].tile(config["gp_meld"]["train_n"], 1).cpu().numpy()
            deap_tgt_train = np.array([])
            '''

            # Create the Batch
            sampled_batch = Batch(
                actions=gp_collection_actions,
                obs=gp_collection_obs,
                priors=gp_collection_priors,
                lengths=deap_lengths,
                rewards=deap_r,
                on_policy=deap_on_policy,
                data_to_encode=deap_data_to_encode_train,
                tgt=deap_tgt_train,
            )

            controller.rl_weight = 0.15

            # Train the controller
            controller.train()
            optimizer.zero_grad()
            deap_b_train = quantile_ver
            loss_s, summaries_s = controller.train_loss(deap_b_train, sampled_batch, pqt_batch, test=True)
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(
                controller.parameters(), gradient_clip)
            optimizer.step()

            controller.rl_weight = 1

        ## Re-open encoder parameter
        # for param in controller.model.dym_embedded_encoder.parameters():
            # param.requires_grad = True

        # for param in controller.model.dym_reduce_dim.parameters():
            # param.requires_grad = True
            
        Program.task.X_train, Program.task.y_train= Program.task.X_train.cpu().detach().numpy(), Program.task.y_train.cpu().detach().numpy()
        print("\n==============End Supervised Learning==============\n")

        # Collect sub-batch statistics and write output
        logger.save_stats(
            r_full,
            l_full,
            actions_full,
            s_full,
            invalid_full,
            r,
            l,
            actions,
            s,
            invalid,
            r_best,
            r_max,
            ewma,
            summaries,
            epoch,
            s_history,
            b_train,
            epoch_walltime,
            controller_programs,
        )

        # Update the memory queue
        if memory_queue is not None:
            memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        new_r_best = False

        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = programs[np.argmax(r)]
            r2, acc_iid, acc_ood = compute_metrics(p_r_best)

        prev_r_best = r_best
        nmse_test = p_r_best.evaluate["nmse_test"]  # pyright: ignore
        if epsilon is not None:
            if save_true_log_likelihood:
                log_and_print(
                    f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| train_loss={loss.item():.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| quantile={quantile:.5f} \t| r_quantile_mean={r_quantile_mean:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t| nmse_test={nmse_test} \t| nll={nll} \t| true_equation_set_count={len(total_unique_set)} \t| s/it={time.time() - t0:.5f}"
                )
            else:
                log_and_print(
                    f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| train_loss={loss.item():.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| quantile={quantile:.5f} \t| r_quantile_mean={r_quantile_mean:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t| nmse_test={nmse_test} \t| true_equation_set_count={len(total_unique_set)} \t| s/it={time.time() - t0:.5f}"
                )
            if quantile > 0.9:
                quantile_over_fit_times += 1
                if quantile_over_fit_times > quantile_over_fit_times_limit:
                    log_and_print("Converged overfitting detected breaking out")
                    break
            else:
                quantile_over_fit_times = 0
        else:
            log_and_print(
                f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| train_loss={loss.item():.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t|  nmse_test={nmse_test} \t| s/it={time.time() - t0:.5f}"
            )

        # Print new best expression
        if verbose and new_r_best:
            log_and_print(
                "[{}] Training epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )
            log_and_print("\n\t** New best")
            p_r_best.print_stats()

        '''
        # Stop if early stopping criteria is met
        if eval_all and any(success):
            log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            break
        if early_stopping and p_r_best.evaluate.get("success"):  # pyright: ignore
            log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            break
        '''
        
        if True:
            if eval_all and any(success):
                log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
                
                with open('test_stats/runtime_result_5.csv', 'w', newline='') as csvfile:
                    # Create a DictWriter object, with fieldnames as the keys of the dictionary
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys())
    
                    # Write the header (the keys of the dictionary)
                    writer.writeheader()
    
                    # Use zip to combine the lists into rows and write them as rows in the CSV
                    for row in zip(*result.values()):
                        writer.writerow(dict(zip(result.keys(), row)))
                
                break
                
            # if early_stopping and p_r_best.evaluate.get("success"):  # pyright: ignore
                # print("Only Risk Term is Satisfied. But Counter Examples may be Found!")
                # log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
                # break
        

        if verbose and (epoch + 1) % 10 == 0:
            log_and_print(
                "[{}] Training epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )

        if debug >= 2:
            log_and_print("\nParameter means after epoch {} of {}:".format(epoch + 1, n_epochs))
            # print_var_means()

        if verbose and (epoch + 1) == n_epochs:
            log_and_print(
                "[{}] Ending training after epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )

        if nevals > n_samples:
            break

    if verbose:
        log_and_print("-- RUNNING EPOCHS END ---------------\n")
        log_and_print("-- EVALUATION START ----------------")
        # print("\n[{}] Evaluating the hall of fame...\n".format(get_duration(start_time)))

    controller.prior.report_constraint_counts()

    # Save all results available only after all epochs are finished. Also return metrics to be added to the summary file
    results_add = logger.save_results(positional_entropy, top_samples_per_batch, r_history, pool, epoch, nevals)

    # Print the priority queue at the end of training
    if verbose and priority_queue is not None:
        for i, item in enumerate(priority_queue.iter_in_order()):
            log_and_print("\nPriority queue entry {}:".format(i))
            p = Program.cache[item[0]]
            p.print_stats()

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final if p_final is not None else p_r_best
    result = {
        "r": p.r,
    }
    result.update(p.evaluate)  # pyright: ignore
    result.update({"expression": repr(p.sympy_expr), "traversal": repr(p), "program": p})
    result.update(results_add)  # pyright: ignore

    if verbose:
        log_and_print("-- EVALUATION END ------------------")
    return result


def program_from_tokens_with_custom_data(tokens, config_task, data, on_policy=True):
    # data : (X, y)
    # X : array-like, shape = [n_samples, n_features]
    # y : shape = [n_samples,]
    config_task["dataset"] = data
    task = make_task(**config_task)
    p = Program(tokens, on_policy=on_policy, custom_task=task)
    return p


def work(p):
    """Compute reward and return it with optimized constants"""
    r = p.r  # pylint: disable=unused-variable  # noqa: F841
    return p


def compute_metrics(p):
    y_hat_test = p.execute(p.task.X_test)
    y_test = p.task.y_test
    acc_iid = acc_tau(y_test, y_hat_test)
    y_test, y_hat_test = remove_nans(y_test, y_hat_test)
    try:
        if y_test.size > 1:
            r2 = scipy.stats.pearsonr(np.nan_to_num(y_test), np.nan_to_num(y_hat_test))[0]
        else:
            r2 = 0
    except Exception as e:
        logger.exception("Error in pearson calculation {}".format(e))
        r2 = 0
    y_hat_test = p.execute(p.task.X_test_ood)
    y_test = p.task.y_test_ood
    acc_ood = acc_tau(y_test, y_hat_test)
    return r2, acc_iid, acc_ood


def compute_metrics_gp(ind, gp_main, dataset):
    f = gp_main.toolbox.compile(expr=ind)
    y_hat_test = f(*dataset.X_test.T)
    y_test = dataset.y_test
    acc_iid = acc_tau(y_test, y_hat_test)
    y_test, y_hat_test = remove_nans(y_test, y_hat_test)
    if y_test.size > 1:
        r2 = scipy.stats.pearsonr(y_test, y_hat_test)[0]
    else:
        r2 = 0

    y_hat_test = f(*dataset.X_test_ood.T)
    y_test = dataset.y_test_ood
    acc_ood = acc_tau(y_test, y_hat_test)
    return r2, acc_iid, acc_ood


def acc_tau(y, y_hat, tau=0.05):
    error = np.abs(((y_hat - y) / y))
    error = np.sort(error)[: -int(error.size * 0.05)]
    return (error <= tau).mean()


def make_X(spec, n_input_var, dataset_size_multiplier):
    """Creates X values based on specification"""

    features = []
    for i in range(1, n_input_var + 1):
        # Hierarchy: "all" --> "x{}".format(i)
        input_var = "x{}".format(i)
        if "all" in spec:
            input_var = "all"
        elif input_var not in spec:
            input_var = "x1"

        if "U" in spec[input_var]:
            low, high, n = spec[input_var]["U"]
            n = int(n * dataset_size_multiplier)
            feature = np.random.uniform(low=low, high=high, size=n)
        elif "E" in spec[input_var]:
            start, stop, step = spec[input_var]["E"]
            if step > stop - start:
                n = step
            else:
                n = int((stop - start) / step) + 1
            n = int(n * dataset_size_multiplier)
            feature = np.linspace(start=start, stop=stop, num=n, endpoint=True)
        else:
            raise ValueError("Did not recognize specification for {}: {}.".format(input_var, spec[input_var]))
        features.append(feature)

    # Do multivariable combinations
    if "E" in spec[input_var] and n_input_var > 1:
        X = np.array(list(itertools.product(*features)))
    else:
        X = np.column_stack(features)

    return X


def remove_nans(y_test, y_hat_test):
    y_test_r = []
    y_hat_test_r = []
    for i in zip(y_test, y_hat_test):
        if not np.isnan(i[0]) and not np.isnan(i[1]):
            y_test_r.append(i[0])
            y_hat_test_r.append(i[1])
    y_test_r = np.array(y_test_r)
    y_hat_test_r = np.array(y_hat_test_r)
    return y_test_r, y_hat_test_r


def compute_val_loss(
    dlval,
    batch_inner_equations,
    controller,
    run_gp_meld,
    config,
    gp_controller,
    pool,
    eval_all,
    epsilon,
    baseline,
    alpha,
):
    # Put into val mode & correct train mode above
    it = 0
    val_loss = np.inf  # pylint: disable=unused-variable  # noqa: F841
    r_total = 0
    for raw_batch in dlval:
        if raw_batch[0].nelement() == 0:
            log_and_print("WARNING no data in batch skipping")
            continue
        # Set of str representations for all Programs ever seen
        # s_history = set(r_history.keys()
        #                 if Program.task.stochastic else Program.cache.keys())
        t0 = time.time()  # pylint: disable=unused-variable  # noqa: F841
        data, eqs = process_raw_batch(raw_batch, controller)
        if data.nelement() == 0:
            log_and_print("WARNING no data in batch filtered eqs skipping")
            continue
        # sampled_data_raw = data.permute(0, 2, 1)
        batch_sos_tokens = torch.ones((1, data.shape[0]), dtype=torch.long).to(DEVICE) * controller.sos_token
        token_eqs = eqs[:, : controller.max_length].to(torch.long).to(DEVICE)
        tgt = torch.cat((batch_sos_tokens, token_eqs.T), 0)
        tgt = tgt.T.tile(batch_inner_equations, 1).T

        sampled_data_cpu = data.cpu().numpy()
        sampled_data = data.cpu()  # Could change this not to CPU - however can't change too many things at once
        # tokens = raw_batch[1].to(DEVICE) + 1
        # token_inputs = tokens.tile(batch_inner_equations, 1)
        data_to_encode = sampled_data.permute(0, 2, 1).tile(batch_inner_equations, 1, 1)
        actions, obs, priors = controller.sample(data_to_encode.shape[0], data_to_encode.detach().to(DEVICE))
        if actions.size == 0:
            log_and_print("WARNING : actions shape {}".format(actions.shape))
            continue
        programs = []
        for i in range(sampled_data_cpu.shape[0]):
            data_xy = sampled_data_cpu[i].T
            x = data_xy[:, :-1]
            if x.shape[1] != (controller.encoder_input_dim - 1):
                x = np.pad(data_xy[:, :-1], ((0, 0), (0, controller.encoder_input_dim - 1 - x.shape[1])))
            data = (x, data_xy[:, -1])
            programs.extend(
                [
                    program_from_tokens_with_custom_data(a, deepcopy(config["task"]), data)
                    for a in actions[(i * batch_inner_equations) : (i + 1) * batch_inner_equations, :]
                ]
            )

        # Run GP seeded with the current batch, returning elite samples
        if run_gp_meld:
            deap_programs, deap_actions, deap_obs, deap_priors = gp_controller(actions)

            # Combine RNN and deap programs, actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)

        # Compute rewards in parallel
        if pool is not None:
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            # Make sure to update cache with new programs
            # Program.cache.update(pool_p_dict)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])
        r_train = r  # pylint: disable=unused-variable  # noqa: F841
        if r.size == 0:
            log_and_print("WARNING : rewards are empty {}".format(actions.shape))
            continue

        # Back up programs to save them properly later
        # controller_programs = programs.copy() if save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train = programs  # pylint: disable=unused-variable  # noqa: F841

        l = np.array([len(p.traversal) for p in programs])  # noqa: E741
        # Str representations of Programs
        s = [p.str for p in programs]
        # pylint: disable-next=unused-variable
        on_policy = np.array([p.originally_on_policy for p in programs])  # noqa: F841
        invalid = np.array([p.invalid for p in programs], dtype=bool)
        invalid_percent = (np.sum(invalid) / invalid.shape[0]) * 100  # pylint: disable=unused-variable  # noqa: F841

        if eval_all:
            success = [p.evaluate.get("success") for p in programs]
            # Check for success before risk-seeking, but don't break until after
            if any(success):
                p_final = programs[success.index(True)]  # pylint: disable=unused-variable  # noqa: F841

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l  # pylint: disable=unused-variable  # noqa: F841
        s_full = s  # pylint: disable=unused-variable  # noqa: F841
        actions_full = actions  # pylint: disable=unused-variable  # noqa: F841
        invalid_full = invalid  # pylint: disable=unused-variable  # noqa: F841
        r_max = np.max(r)  # pylint: disable=unused-variable  # noqa: F841

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        r_total = np.mean(r_full)
        it += 1
        break
    r_avg = r_total / it
    return 1 / r_avg
    # if epsilon is not None and epsilon < 1.0:  # Empirical quantile
    #     quantile = np.quantile(
    #         r, 1 - epsilon, interpolation="higher")

    #     r_raw_sum = np.sum(r)
    #     r_raw_mean = r.sum() / (r != 0).sum()

    #     # These guys can contain the GP solutions if we run GP
    #     '''
    #         Here we get the returned as well as stored programs and properties.

    #         If we are returning the GP programs to the controller, p and r will be exactly the same
    #         as p_train and r_train. Otherwise, p and r will still contain the GP programs so they
    #         can still fall into the hall of fame. p_train and r_train will be different and no longer
    #         contain the GP program items.
    #     '''

    #     keep = r >= quantile
    #     l = l[keep]
    #     s = list(compress(s, keep))
    #     invalid = invalid[keep]

    #     # Option: don't keep the GP programs for return to controller
    #     if run_gp_meld and not gp_controller.return_gp_obs:
    #         '''
    #             If we are not returning the GP components to the controller, we will remove them from
    #             r_train and p_train by augmenting 'keep'. We just chop off the GP elements which are indexed
    #             from batch_size to the end of the list.
    #         '''
    #         _r = r[keep]
    #         _p = list(compress(programs, keep))
    #         keep[batch_size:] = False
    #         r_train = r[keep]
    #         p_train = list(compress(programs, keep))

    #         '''
    #             These contain all the programs and rewards regardless of whether they are returned to the controller.
    #             This way, they can still be stored in the hall of fame.
    #         '''
    #         r = _r
    #         programs = _p
    #     else:
    #         '''
    #             Since we are returning the GP programs to the contorller, p and r are the same as p_train and r_train.
    #         '''
    #         r_train = r = r[keep]
    #         p_train = programs = list(compress(programs, keep))

    #     '''
    #         get the action, observation, priors and on_policy status of all programs returned to the controller.
    #     '''
    #     actions = actions[keep, :]
    #     obs = obs[keep, :, :]
    #     priors = priors[keep, :, :]
    #     on_policy = on_policy[keep]
    # else:
    #     keep = None

    # r_quantile_sum = np.sum(r)
    # # Clip bounds of rewards to prevent NaNs in gradient descent
    # r = np.clip(r, -1e6, 1e6)
    # r_train = np.clip(r_train, -1e6, 1e6)

    # # Compute baseline
    # # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
    # if baseline == "ewma_R":
    #     ewma = np.mean(r_train) if ewma is None else alpha * \
    #         np.mean(r_train) + \
    #         (1 - alpha) * ewma
    #     b_train = ewma
    # elif baseline == "R_e":  # Default
    #     ewma = -1
    #     b_train = quantile
    # elif baseline == "ewma_R_e":
    #     ewma = np.min(r_train) if ewma is None else alpha * \
    #         quantile + (1 - alpha) * ewma
    #     b_train = ewma
    # elif baseline == "combined":
    #     ewma = np.mean(r_train) - quantile if ewma is None else alpha * \
    #         (np.mean(r_train) - quantile) + \
    #         (1 - alpha) * ewma
    #     b_train = quantile + ewma

    # # Compute sequence lengths
    # lengths = np.array([min(len(p.traversal), controller.max_length)
    #                     for p in p_train], dtype=np.int32)

    # if data_to_encode is not None:
    #     if run_gp_meld:
    #         data_to_encode_train = torch.cat([data_to_encode.detach(), data_to_encode.detach()[-1, :, :].tile(config['gp_meld']['train_n'], 1, 1)])
    #         if keep is not None:
    #             data_to_encode_train = data_to_encode_train[keep, :, :].cpu().numpy()
    #         else:
    #             data_to_encode_train = data_to_encode_train[:, :, :].cpu().numpy()
    #     elif keep is not None:
    #         data_to_encode_train = data_to_encode[keep, :, :].detach(
    #         ).cpu().numpy()
    #     else:
    #         data_to_encode_train = data_to_encode[:, :, :].detach(
    #         ).cpu().numpy()
    # else:
    #     data_to_encode_train = np.array([])

    # # Create the Batch
    # sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
    #                         lengths=lengths, rewards=r_train, on_policy=on_policy, data_to_encode=data_to_encode_train)

    # # Update and sample from the priority queue
    # pqt_batch = None

    # # Train the controller
    # # if pre_train:
    # #     data_to_encode_all = sampled_data.detach()
    # # else:
    # #     data_to_encode_all = data_to_encode
    # try:
    #     loss, summaries = controller.train_loss(b_train, sampled_batch, pqt_batch)
    #     val_loss = loss.item()
    #     return val_loss
    # except Exception as e:
    #     log_and_print('[ERROR] Error {}'.format(str(e)))
    #     continue


def gp_at_test(
    controller,
    pool,
    gp_controller,
    output_file,
    pre_train,
    config,
    test_task,
    controller_saved_path,
    n_epochs=None,
    n_samples=2000000,
    batch_size=1000,
    complexity="token",
    const_optimizer="scipy",
    const_params=None,
    alpha=0.5,
    epsilon=0.05,
    n_cores_batch=1,
    verbose=True,
    save_summary=False,
    save_all_epoch=False,
    baseline="R_e",
    b_jumpstart=False,
    early_stopping=True,
    hof=100,
    eval_all=False,
    save_pareto_front=True,
    debug=0,
    use_memory=False,
    memory_capacity=1e3,
    warm_start=None,
    memory_threshold=None,
    save_positional_entropy=False,
    save_top_samples_per_batch=0,
    save_cache=False,
    true_action=None,
    save_cache_r_min=0.9,
    save_freq=None,
    save_token_count=False,
    learning_rate=0.001,
    gradient_clip=1,
    save_true_log_likelihood=False,
):
    """
    Executes the main training loop.

    Parameters
    ----------
    controller : dso.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    gp_controller : dso.gp.gp_controller.GPController or None
        GP controller object used to generate Programs.

    output_file : str or None
        Path to save results each step.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity function name, used computing Pareto front.

    const_optimizer : str or None, optional
        Name of constant optimizer.

    const_params : dict, optional
        Dict of constant optimizer kwargs.

    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

    verbose : bool, optional
        Whether to print progress.

    save_summary : bool, optional
        Whether to write TensorFlow summaries.

    save_all_epoch : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    save_pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    use_memory : bool, optional
        If True, use memory queue for reward quantile estimation.

    memory_capacity : int
        Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional
        Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional
        Whether to store X% top-performer samples in every batch.

    save_cache : bool
        Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None
        Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

    save_token_count : bool
        Whether to save token counts each batch.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by reward).
    """
    # import random
    # import warnings
    # from dso.program import from_tokens
    # from utils.train import compute_metrics_gp, compute_metrics

    t0 = time.time()
    ind_best, logbook, hof = controller.train()
    print(f"time taken: {time.time()-t0}")
    traversal = [token.name for token in ind_best]
    traversal = [i if i[:3] != "ARG" else f"x{int(i[3])+1}" for i in traversal]
    a = test_task.library.actionize(traversal)
    p_final = from_tokens(a)

    nevals = 0
    for epoch_d in logbook.chapters["fitness"]:
        nevals += epoch_d["nevals"]
        epoch = epoch_d["gen"]
        best_nmse = epoch_d["min"]
        r_best = 1 / (1 + best_nmse)
        logger.info(f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| r_best={r_best:.5f} \t|  nmse_test={best_nmse}")

    # r2, acc_iid, acc_ood = compute_metrics_gp(ind_best, gp_main, dataset)

    hof_ps = []
    for ind in hof:
        traversal = [token.name for token in ind]
        traversal = [i if i[:3] != "ARG" else f"x{int(i[3])+1}" for i in traversal]
        a = test_task.library.actionize(traversal)
        hof_ps.append(from_tokens(a))
    log_and_print(" Pareto Front ")
    for i, p in enumerate(hof_ps):
        log_and_print(f"    {i}: S=000 R={p.r:.6f} C={p.complexity:.2f} <-- {p.sympy_expr}")
    if verbose:
        log_and_print("-- RUNNING EPOCHS END ---------------\n")
        log_and_print("-- EVALUATION START ----------------")
        # print("\n[{}] Evaluating the hall of fame...\n".format(get_duration(start_time)))

    # Save all results available only after all epochs are finished. Also return metrics to be added to the summary file
    # results_add = logger.save_results(None, None, None, pool, epoch, nevals)

    # Print the priority queue at the end of training

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final
    result = {
        "r": p.r,
    }
    result.update(p.evaluate)  # pyright: ignore
    result.update({"expression": repr(p.sympy_expr), "traversal": repr(p), "program": p})
    # result.update(results_add)

    if verbose:
        log_and_print("-- EVALUATION END ------------------")
    return result


def sample_nesymres_at_test(
    controller,
    pool,
    gp_controller,
    output_file,
    pre_train,
    config,
    test_task,
    controller_saved_path,
    n_epochs=None,
    n_samples=2000000,
    batch_size=1000,
    complexity="token",
    const_optimizer="scipy",
    const_params=None,
    alpha=0.5,
    epsilon=0.05,
    n_cores_batch=1,
    verbose=True,
    save_summary=False,
    save_all_epoch=False,
    baseline="R_e",
    b_jumpstart=False,
    early_stopping=True,
    hof=100,
    eval_all=False,
    save_pareto_front=True,
    debug=0,
    use_memory=False,
    memory_capacity=1e3,
    warm_start=None,
    memory_threshold=None,
    save_positional_entropy=False,
    save_top_samples_per_batch=0,
    save_cache=False,
    save_cache_r_min=0.9,
    save_freq=None,
    save_token_count=False,
    learning_rate=0.001,
    gradient_clip=1,
    save_true_log_likelihood=False,
    true_action=None,
):
    """
    Executes the main training loop.

    Parameters
    ----------
    controller : dso.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    gp_controller : dso.gp.gp_controller.GPController or None
        GP controller object used to generate Programs.

    output_file : str or None
        Path to save results each step.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity function name, used computing Pareto front.

    const_optimizer : str or None, optional
        Name of constant optimizer.

    const_params : dict, optional
        Dict of constant optimizer kwargs.

    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

    verbose : bool, optional
        Whether to print progress.

    save_summary : bool, optional
        Whether to write TensorFlow summaries.

    save_all_epoch : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    save_pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    use_memory : bool, optional
        If True, use memory queue for reward quantile estimation.

    memory_capacity : int
        Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional
        Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional
        Whether to store X% top-performer samples in every batch.

    save_cache : bool
        Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None
        Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

    save_token_count : bool
        Whether to save token counts each batch.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by reward).
    """
    # import random
    import warnings

    # Main training loop
    p_final = None
    r_best = -np.inf
    prev_r_best = None
    nevals = 0  # Total number of sampled expressions
    # eval_all = True
    total_unique_set = set()

    logger = StatsLogger(  # pylint: disable=unused-variable  # noqa: F841
        output_file,
        save_summary,
        save_all_epoch,
        hof,
        save_pareto_front,
        save_positional_entropy,
        save_top_samples_per_batch,
        save_cache,
        save_cache_r_min,
        save_freq,  # pyright: ignore
        save_token_count,
    )

    start_time = time.time()
    if verbose:
        print("-- RUNNING EPOCHS START -------------")

    # beamsize = 2**random.randint(4,10)
    beamsize = 2**8
    epoch = 0
    seen_same_result_now_limit = 1
    seen_same_result_now = 0
    while nevals < n_samples:
        t0 = time.time()
        # beamsize += 1
        # beamsize = 2**beamsize_exp
        # beamsize = 2**random.randint(4,10)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            samples = controller.sample(test_task.X_train, test_task.y_train, beamsize=beamsize)  # samples=2000)
            if save_true_log_likelihood:
                nll = controller.compute_neg_log_likelihood(test_task.X_train, test_task.y_train, true_action).item()
            nevals += len(samples)
        # print(f'SAMPLING: {len(samples) /(time.time() - t0)}')
        programs = []
        actions = []
        for sample_prefix in samples:
            try:
                eq = eq_remove_constants(eq_sympy_prefix_to_token_library(sample_prefix))
                a = test_task.library.actionize(eq)
                programs.append(from_tokens(a))
                actions.append(a)
            except Exception:
                # print(e)
                continue
        # Compute rewards in parallel
        if pool is not None:
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])
        r_train = r  # pylint: disable=unused-variable  # noqa: F841

        # Back up programs to save them properly later
        # pylint: disable-next=unused-variable
        controller_programs = programs.copy() if save_token_count else None  # noqa: F841

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train = programs  # pylint: disable=unused-variable  # noqa: F841

        l = np.array([len(p.traversal) for p in programs])  # noqa: E741
        s = [p.str for p in programs]  # Str representations of Programs
        # pylint: disable-next=unused-variable
        on_policy = np.array([p.originally_on_policy for p in programs])  # noqa: F841
        invalid = np.array([p.invalid for p in programs], dtype=bool)
        invalid_percent = (np.sum(invalid) / invalid.shape[0]) * 100

        if eval_all:
            success = [p.evaluate.get("success") for p in programs]

            # for idx, suc in enumerate(success):
            #     if suc:
            #         p = programs[idx]
            #         traverasl = p.traversal.copy()
            #         key = ','.join([str(i) for i in traverasl])
            #         total_unique_set.add(key)

            # Check for success before risk-seeking, but don't break until after
            if any(success):
                p_final = programs[success.index(True)]

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r  # pylint: disable=unused-variable  # noqa: F841
        l_full = l  # pylint: disable=unused-variable  # noqa: F841
        s_full = s  # pylint: disable=unused-variable  # noqa: F841
        actions_full = np.array(actions)  # pylint: disable=unused-variable  # noqa: F841
        invalid_full = invalid  # pylint: disable=unused-variable  # noqa: F841
        r_max = np.max(r)
        r_best = max(r_max, r_best)
        r_raw_sum = np.sum(r)
        r_raw_mean = r.sum() / (r != 0).sum()
        ewma = 0  # pylint: disable=unused-variable  # noqa: F841
        summaries = None  # pylint: disable=unused-variable  # noqa: F841
        s_history = None  # pylint: disable=unused-variable  # noqa: F841
        b_train = None  # pylint: disable=unused-variable  # noqa: F841

        epoch_walltime = time.time() - start_time  # pylint: disable=unused-variable  # noqa: F841

        # Collect sub-batch statistics and write output
        # logger.save_stats(r_full, l_full, actions_full, s_full, invalid_full, r,
        #                   l, actions, s, invalid, r_best, r_max, ewma, summaries, epoch,
        #                   s_history, b_train, epoch_walltime, controller_programs)

        # Update new best expression
        new_r_best = False

        if r_max == r_best:
            seen_same_result_now += 1

        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = programs[np.argmax(r)]
            r2, acc_iid, acc_ood = compute_metrics(p_r_best)

        prev_r_best = r_best
        nmse_test = p_r_best.evaluate["nmse_test"]
        if epsilon is not None:
            if save_true_log_likelihood:
                log_and_print(
                    f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t| nmse_test={nmse_test} \t| nll={nll} \t| s/it={time.time() - t0:.5f}"
                )
            else:
                log_and_print(
                    f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t| nmse_test={nmse_test} \t| true_equation_set_count={len(total_unique_set)} \t| s/it={time.time() - t0:.5f}"
                )
        else:
            log_and_print(
                f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t|  nmse_test={nmse_test} \t| s/it={time.time() - t0:.5f}"
            )

        # Print new best expression
        if verbose and new_r_best:
            log_and_print(
                "[{}] Training epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )
            log_and_print("\n\t** New best")
            p_r_best.print_stats()

        # Stop if early stopping criteria is met
        if eval_all and any(success):
            log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            break
        if early_stopping and p_r_best.evaluate.get("success"):
            log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            break

        if verbose and (epoch + 1) % 10 == 0:
            log_and_print(
                "[{}] Training epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )

        if debug >= 2:
            log_and_print("\nParameter means after epoch {} of {}:".format(epoch + 1, n_epochs))
            # print_var_means()

        if verbose and (epoch + 1) == n_epochs:
            log_and_print(
                "[{}] Ending training after epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )

        if nevals > n_samples:
            break

        if seen_same_result_now > seen_same_result_now_limit:
            log_and_print(f"Seen same result now {seen_same_result_now} times, breaking out")
            break

        epoch += 1

    if verbose:
        log_and_print("-- RUNNING EPOCHS END ---------------\n")
        log_and_print("-- EVALUATION START ----------------")
        # print("\n[{}] Evaluating the hall of fame...\n".format(get_duration(start_time)))

    # Save all results available only after all epochs are finished. Also return metrics to be added to the summary file
    # results_add = logger.save_results(None, None, None, pool, epoch, nevals)

    # Print the priority queue at the end of training

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final if p_final is not None else p_r_best
    result = {
        "r": p.r,
    }
    result.update(p.evaluate)
    result.update({"expression": repr(p.sympy_expr), "traversal": repr(p), "program": p})
    # result.update(results_add)

    if verbose:
        log_and_print("-- EVALUATION END ------------------")
    return result


## helper function for root finding

def derivative_calculate(v, x1, x2, x3, x4, x5, x6, x7, x8):

    
    # x1_deri = -x1 + 0.5 * x2 - 0.1 * x5 ** 2
    # x2_deri = -0.5 * x1 - x2
    # x3_deri = -x3 + 0.5 * x4 - 0.1 * x1 ** 2
    # x4_deri = -0.5 * x3 - x4
    # x5_deri = -x5 + 0.5 * x6
    # x6_deri = -0.5 * x5 - x6 + 0.1 * x2 ** 2

    # x7_deri = -x7 + 0.5 * x8
    # x8_deri = -0.5 * x7 - x8
    

    # x1_deri = x2
    # x2_deri = - 5 * sym.sin(x1) - 0.1 * x2
    
    # x1_deri = x2
    # x2_deri = -x1 - (1 - x1 ** 2) * x2

    # x1_deri = x2
    # x2_deri = - 1 * sym.sin(x1)*sym.cos(x1) - x2 - 1 * sym.sin(x3)*sym.cos(x3)
    # x3_deri = x2 - x3

    # x1_deri = x2
    # x2_deri = - 1 * x1 - x2 - 1 * x3
    # x3_deri = x2 - x3

    # x1_deri = -x1 + 0.5 * x2 - 0.1 * x3 ** 2
    # x2_deri = -0.5 * x1 - x2
    # x3_deri = -x3 + 0.5 * x4 - 0.1 * x1 ** 2
    # x4_deri = -0.5 * x3 - x4

    
    # x1_deri = x2
    # x2_deri = - 1 * sym.sin(x1)*sym.cos(x1) - x2 - 1 * sym.sin(x3)*sym.cos(x3)
    # x3_deri = x2 - x3

    # x1_deri = sym.sin(x2)
    # x2_deri = -sym.cos(x2) / (1 - x1)
    # x2_deri = -x2 - 1 * (sym.sin(x2) / (x2)) *x1

    # x1_deri = -x1 + 0.5 * x2 - 0.1 * x9 ** 2
    # x2_deri = -0.5 * x1 - x2
    # x3_deri = -x3 + 0.5 * x4 - 0.1 * x1 ** 2
    # x4_deri = -0.5 * x3 - x4
    # x5_deri = -x5 + 0.5 * x6 + 0.1 * x7 ** 2
    # x6_deri = -0.5 * x5 - x6
    # x7_deri = -x7 + 0.5 * x8
    # x8_deri = -0.5 * x7 - x8
    # x9_deri = -x9 + 0.5 * x10
    # x10_deri = -0.5 * x9 - x10 + 0.1 * x2 ** 2

    # x1_deri = x4 - (x4 + x5 + x6) / 3
    # x2_deri = x5 - (x4 + x5 + x6) / 3
    # x3_deri = x6 - (x4 + x5 + x6) / 3

    # x4_deri = -2 * x4 - sym.sin(x1 - x2) - sym.sin(x1 - x3)
    # x5_deri = -2 * x5 - sym.sin(x2 - x1) - sym.sin(x2 - x3)
    # x6_deri = -2 * x6 - sym.sin(x3 - x1) - sym.sin(x3 - x2)

    x1_deri = -x1 + 0.5 * x2 - 0.1 * x5 ** 2
    x2_deri = -0.5 * x1 - x2
    x3_deri = -x3 + 0.5 * x4 - 0.1 * x1 ** 2
    x4_deri = -0.5 * x3 - x4
    x5_deri = -x5 + 0.5 * x6 + 0.1 * x7 ** 2
    x6_deri = -0.5 * x5 - x6
    x7_deri = -x7 + 0.5 * x8
    x8_deri = -0.5 * x7 - x8 - 0.1 * x4 ** 2

    # x1_deri = -x1 + 0.5 * x2 - 0.1 * x7 ** 2
    # x2_deri = -0.5 * x1 - x2
    # x3_deri = -x3 + 0.5 * x4 + 0.1 * x5 ** 2
    # x4_deri = -0.5 * x3 - x4
    # x5_deri = -x5 + 0.5 * x6 
    # x6_deri = -0.5 * x5 - x6
    # x7_deri = -x7 + 0.5 * x8
    # x8_deri = -0.5 * x7 - x8 + 0.1 * x2 ** 2

    return ((-1) * (v.diff(x1) * x1_deri + v.diff(x2) * x2_deri + v.diff(x3) * x3_deri + v.diff(x4) * x4_deri+ v.diff(x5) * x5_deri + v.diff(x6) * x6_deri+ v.diff(x7) * x7_deri + v.diff(x8) * x8_deri))#  # # ))#  # + v.diff(x9) * x9_deri+ v.diff(x10) * x10_deri


def find_root(func):
    # root = optimize.fsolve(func, guess)
    # outer_root = False

    bounds = Bounds([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    result = shgo(func, bounds, n = 1024, iters = 3, sampling_method = "simplicial")
    root = result.x


    if abs(root[0]) > 1.5:
        outer_root = True
        outer_root = True
    
    # elif abs(root[2]) > 1:
        # outer_root = True
    
    # elif abs(root[3]) > 1:
        # outer_root = True
    '''
    elif abs(root[4]) > 1:
        outer_root = True
    elif abs(root[5]) > 1:
        outer_root = True
    '''
        
    # res = func(root)
    # res_success = (np.abs(res[0])<=0.001 and ((np.linalg.norm(root)>0.001) and (not outer_root)))
    # if np.linalg.norm(guess)==0:
        # res_success = ((np.abs(res[0])<=0.001 and (np.linalg.norm(root)<0.001)) or (np.abs(res[0])>0.0001))
    res_success = True
    return root, res_success

def counter_exp_finder_deri(root1, func1, root2, func2, num=700):
    counter_example = []
    pd_counter_example = []
    # distance = np.linspace(np.array([-0.5,-0.5, -0.5]),np.array([0.5,0.5, 0.5]),num)
    distance = np.random.uniform(0, 0.25, (num,8)) # 0.3
    # for i in distance:
        # for j in range(len(i)):
            # i[j] += np.random.randn()

    distance = np.concatenate((distance,np.random.randn(distance.shape[0],distance.shape[1])*0.01),axis=0)
    for j in range(num*2):
        root1_minus = root1 - distance[j]
        root1_minus = np.clip(root1_minus, -1.0, 1.0)
        root1_plus = root1 + distance[j]
        root1_plus = np.clip(root1_plus,  -1.0, 1.0)
        root2_minus = root2 - distance[j]
        root2_minus = np.clip(root2_minus,-1.0, 1.0)
        root2_plus = root2 + distance[j]
        root2_plus = np.clip(root2_plus, -1.0, 1.0)

        value1 = func1(root1_minus)
        value2 = func1(root1_plus)
        value3 = func2(root2_minus)
        value4 = func2(root2_plus)
        
        if value1 < 0:
            '''
            if np.sum([i == 0 for i in root1_minus]):
                if value1[0] < 0:
                    pd_counter_example.append((root1_minus).copy())
                else:
                    pd_counter_example.append((root1_minus).copy())
            '''
            pd_counter_example.append((root1_minus).copy())

        if value2 < 0:
            '''
            if np.sum([i == 0 for i in root1_plus]):
                if value1[0] < 0:
                    pd_counter_example.append((root1_plus).copy())
                else:
                    pd_counter_example.append((root1_plus).copy())
            '''
            pd_counter_example.append((root1_plus).copy())
        
        if value3 < - 0:
            '''
            if np.sum([i == 0 for i in root2_minus]):
                if value3 < - 0.0001:
                    counter_example.append((root2_minus).copy())
            else:
                if value3[0] < - 0.0001:
                    counter_example.append((root2_minus).copy())
            '''
            counter_example.append((root2_minus).copy())
        if value4 < - 0:
            '''
            if np.sum([i == 0 for i in root2_plus]):
                if value4[0] < - 0.0001:
                    counter_example.append((root2_plus).copy())
            else:
                if value4[0] < - 0.0001:
                    counter_example.append((root2_plus).copy())
            '''
            counter_example.append((root2_plus).copy())
    del distance

    if func1(root1) < 0:
        pd_counter_example.append(root1.copy())
    if func2(root2) < 0:
        counter_example.append((root2.copy()))
    return counter_example, pd_counter_example

def check_options(sympy_expr):
    
    x1 = sym.symbols('x1')
    x2 = sym.symbols('x2')

    x3, x4, x5, x6 = sym.symbols('x3, x4, x5, x6')

    x7, x8, x9, x10 = sym.symbols("x7, x8, x9, x10")

    '''
    if len(list(sympy_expr.free_symbols)) < 6:
        return False, np.array([])
    '''

    origin = float(sympy_expr.evalf(subs = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0, "x6": 0, "x7": 0, "x8": 0}))
    sympy_expr = sympy_expr - origin
    # sympy_expr = sympy_expr.subs(x3, 9.81)
    numpy_expr = sym.lambdify((x1,x2,x3,x4,x5,x6,x7,x8), sympy_expr, "numpy")
    v_dot = derivative_calculate(sympy_expr,x1,x2,x3,x4,x5,x6,x7,x8)
    numpy_v_dot = sym.lambdify((x1,x2,x3,x4,x5,x6,x7,x8), v_dot, "numpy")
        
    function1 = lambda x: numpy_expr(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
    function2 = lambda x: numpy_v_dot(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])


    root_1, root_1_sucs = find_root(function1)
    root_2, root_2_sucs = find_root(function2)

    counter_example, pd = counter_exp_finder_deri(root_1, function1, root_2, function2)
    counter_exp = counter_example
    counter_exp = np.array(counter_exp)

    
    if ((len(counter_exp) + len(pd)) == 0): 
        valid = True
        return valid, counter_exp
    else:
        valid = False
        return valid, counter_exp
    

def final_check(sympy_expr):
    x1 = sym.symbols('x1')
    x2 = sym.symbols('x2')

    x3, x4, x5, x6 = sym.symbols('x3, x4, x5, x6')

    x7, x8, x9, x10 = sym.symbols("x7, x8, x9, x10")

    origin = float(sympy_expr.evalf(subs = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0, "x6": 0, "x7": 0, "x8": 0}))
    sympy_expr = sympy_expr - origin
    # sympy_expr = sympy_expr.subs(x3, 9.81)
    numpy_expr = sym.lambdify((x1,x2,x3,x4,x5,x6,x7,x8), sympy_expr, "numpy")
    v_dot = derivative_calculate(sympy_expr,x1,x2,x3,x4,x5,x6,x7,x8)
    numpy_v_dot = sym.lambdify((x1,x2,x3,x4,x5,x6,x7,x8), v_dot, "numpy")

    
    # whole state random check
    global function_check_1
    global function_check_2

    def function_check_1(x):
        return numpy_expr(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
    
    def function_check_2(x):
        return numpy_v_dot(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
    
    # pgd check
    pgd_check_result = True
    for i in range(5):
        valid, counter_exp = pgd_check(sympy_expr, torch.ones((1000000, 8)))
        if not valid:
            pgd_check_result = False
            break
    
    if not pgd_check_result:
        return False

    num_points = 10 ** 6
    # pool_1 = None
    for i in range(10):
        # data = np.random.uniform(-1.0, 1.0, (1 * 10 ** 7, 6))

        boundary_points = []

        # Iterate over each dimension to create points near the boundaries
        for j in range(8):
            # Points near the -1 boundary for the i-th dimension
            point_set_low = np.random.uniform(-1.0, 1.0, (num_points, 8))
            point_set_low[:, j] = -1 + np.abs(0.001 * np.random.rand(num_points))
    
            # Points near the 1 boundary for the i-th dimension
            point_set_high = np.random.uniform(-1.0, 1.0, (num_points, 8))
            point_set_high[:, j] = 1 -  np.abs(0.001 * np.random.rand(num_points))
    

            point_uniform = np.random.uniform(-1.0, 1.0, (num_points, 8))
            # Add these points to the boundary_points list
            boundary_points.append(point_set_low)
            boundary_points.append(point_set_high)
            boundary_points.append(point_uniform)

        # Combine all boundary points into one array
        boundary_points = np.vstack(boundary_points)

        check_1 = function_check_1(boundary_points)
        check_2 = function_check_2(boundary_points)

        if any(check_1 < 0) or any(check_2 < 0):
            return False
        
    return True

def pgd_attack(
    x0, f, eps, steps=10, lower_boundary=None, upper_boundary=None, direction="minimize"
):
    """
    Use adversarial attack (PGD) to find violating points.
    Args:
      x0: initialization points, in [batch, state_dim].
      f: function f(x) to find the worst case x to maximize.
      eps: perturbation added to x0.
      steps: number of pgd steps.
      lower_boundary: absolute lower bounds of x.
      upper_boundary: absolute upper bounds of x.
    """
    # Set all parameters without gradient, this can speedup things significantly
    grad_status = {}
    try:
        for p in f.parameters():
            grad_status[p] = p.requires_grad
            p.requires_grad_(False)
    except:
        pass

    step_size = eps / steps * 2
    noise = torch.randn_like(x0) * step_size
    if lower_boundary is not None:
        lower_boundary = torch.max(lower_boundary, x0 - eps)
    else:
        lower_boundary = x0 - eps
    if upper_boundary is not None:
        upper_boundary = torch.min(upper_boundary, x0 + eps)
    else:
        upper_boundary = x0 + eps
    x = x0.detach().clone().requires_grad_()
    # Save the best x and best loss.
    best_x = torch.clone(x).detach().requires_grad_(False)
    fill_value = float("-inf") if direction == "maximize" else float("inf")
    best_loss = torch.full(
        size=(x.size(0),),
        requires_grad=False,
        fill_value=fill_value,
        device=x.device,
        dtype=x.dtype,
    )
    for i in range(steps):
        output = f(x1 = x[:,[0]], x2 = x[:,[1]], x3 = x[:,[2]], x4 = x[:,[3]], x5 = x[:,[4]], x6 = x[:,[5]], x7 = x[:,[6]], x8 = x[:,[7]]).squeeze(1).squeeze(1)
        # output = torch.clamp(f(x).squeeze(1), max=0)
        output.mean().backward()
        if direction == "maximize":
            improved_mask = output >= best_loss
        else:
            improved_mask = output <= best_loss
        best_x[improved_mask] = x[improved_mask]
        best_loss[improved_mask] = output[improved_mask]
        # print(f'step = {i}', output.view(-1).detach())
        # print(x.detach(), best_x)
        noise = torch.randn_like(x0) * step_size / (i + 1)
        if direction == "maximize":
            x = (
                (
                    torch.clamp(
                        x + torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )
        else:
            x = (
                (
                    torch.clamp(
                        x - torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )

    # restore the gradient requirement for model parameters
    try:
        for p in f.parameters():
            p.requires_grad_(grad_status[p])
    except:
        pass
    return best_x

def pgd_check(sympy, X):

    if len(list(sympy.free_symbols)) < 8:
        return False, np.array([])

    x1, x2, x3, x4, x5, x6, x7, x8 = sym.symbols('x1, x2, x3, x4, x5, x6, x7, x8')

    sympy = derivative_calculate(sympy, x1, x2, x3, x4, x5, x6, x7, x8)

    if len(list(sympy.free_symbols)) < 1:
        return False, np.array([])
    
    sympy_torch = sympytorch.SymPyModule(expressions=[sympy]).to("cuda")
    data = torch.rand((X.shape[0], X.shape[1])).to("cuda")
    result = pgd_attack(data, sympy_torch, 0.8, steps=30, lower_boundary=torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).to("cuda"), upper_boundary=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to("cuda"), direction="minimize")
    result = torch.clamp(result, -1.0, 1.0)
    evaluation = sympy_torch(x1 = result[:,[0]], x2 = result[:,[1]], x3 = result[:,[2]], x4 = result[:,[3]], x5 = result[:,[4]], x6 = result[:,[5]], x7 = result[:,[6]], x8 = result[:,[7]]).squeeze(1).squeeze(1)
    result = result[evaluation < - 1e-15]

    return (len(result) == 0), result.cpu().detach().numpy()
    
def init_worker():
    # Redirect stdout and stderr to suppress print statements in workers
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def prepare_encoder_input():

    global vocab

    vocab = {'start': 0, 'add': 1, 'mul': 2, 'pow': 3, 'sin': 4, 'cos': 5, '+': 6, '-': 7, 
             '1':8 ,'2':9, '3':10, '4':11, '5':12, '6':13, '7':14, '8':15, '9': 16, '0':17, 
             'E+0': 18, 'E-1': 19, 'E-2':20, 'E-3':21, "x1":22, 'x2':23, "x3": 24, "x4": 25,
              "x5": 26, "x6": 27, "x7": 28, "x8": 29, "end": 30}

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sym.symbols("x1, x2, x3, x4, x5, x6, x7, x8, x9, x10")

    # f = [-x1 + 0.5 * x2 - 0.1 * x3 ** 2,
        # -0.5 * x1 - x2,
        # -x3 + 0.5 * x4 - 0.1 * x1 ** 2,
        # -0.5 * x3 - x4]


    # f = [
        # -x1 + 0.5 * x2 - 0.1 * x5 ** 2,
        # -0.5 * x1 - x2,
        # -x3 + 0.5 * x4 - 0.1 * x1 ** 2,
        # -0.5 * x3 - x4,
        # -x5 + 0.5 * x6,
        # -0.5 * x5 - x6 + 0.1 * x2 ** 2
        # ]
    
    # f = [-x1 + 0.5 * x2 - 0.1 * x9 ** 2,
        # -0.5 * x1 - x2,
        # -x3 + 0.5 * x4 - 0.1 * x1 ** 2,
        # -0.5 * x3 - x4,
        # -x5 + 0.5 * x6 + 0.1 * x7 ** 2,
        # -0.5 * x5 - x6,
        # -x7 + 0.5 * x8,
        # -0.5 * x7 - x8,
        # -x9 + 0.5 * x10,
        # -0.5 * x9 - x10 + 0.1 * x2 ** 2]

    # f = [x2,
        # -x1 - (1 - x1 ** 2) * x2]

    f = [
        -x1 + 0.5 * x2 - 0.1 * x5 ** 2,
        -0.5 * x1 - x2,
        -x3 + 0.5 * x4 - 0.1 * x1 ** 2,
        -0.5 * x3 - x4,
        -x5 + 0.5 * x6 + 0.1 * x7 ** 2,
        -0.5 * x5 - x6,
        -x7 + 0.5 * x8,
        -0.5 * x7 - x8 - 0.1 * x4 ** 2
    ]

    # f = [x2,
        # - 5 * sym.sin(x1) - 0.1 * x2]

    # f = [-x1 + 0.5 * x2 - 0.1 * x7 ** 2,
         # -0.5 * x1 - x2,
         # -x3 + 0.5 * x4 + 0.1 * x5 ** 2,
         # -0.5 * x3 - x4,
         # -x5 + 0.5 * x6,
         # -0.5 * x5 - x6,
         # -x7 + 0.5 * x8,
         # -0.5 * x7 - x8 + 0.1 * x2 ** 2
        #  ]

    # f = [x2,
        # - 1 * sym.sin(x1)*sym.cos(x1) - x2 - 1 * sym.sin(x3)*sym.cos(x3),
        # x2 - x3]


    polish_expr = [to_polish_with_encoding(i) for i in f]

    input_ids = [[vocab[token] for token in i] for i in polish_expr] 

    # concant dynamics together
    result = []

    for i in input_ids:
        i.insert(0, vocab['start'])
        i.append(vocab['end'])
        result.extend(i)

    return torch.tensor(result)


# Function to encode an integer in I10
def encode_integer_i10(number):
    sign = '+' if number >= 0 else '-'
    digits = [str(int(digit)) for digit in str(abs(number))]
    return [sign] + (digits)

# Function to encode a real number in E100
def encode_real_f10_e100(number):
    sign = '+' if number >= 0 else '-'
    number = abs(number)
    
    if number == 0:
        return [sign, 0, 'E0']
    
    exponent = math.floor(math.log10(number))
    mantissa = number / (10 ** exponent)
    
    # Round mantissa to 4 significant digits
    mantissa = round(mantissa * 1000)
    
    # Adjust exponent if mantissa rounding affects it
    if mantissa >= 10000:
        mantissa = mantissa // 10
        exponent += 1

    # Clamp the exponent to the range [-100, 100]
    exponent = max(min(exponent, 100), -100)
    
    return [sign] + [str(int(digit)) for digit in str(mantissa)] + [f'E{exponent:+d}']

# Function to convert a sympy expression to Polish notation with I10 and E100 encoding
def to_polish_with_encoding(expr):
    if isinstance(expr, sym.Add):
        return ['add'] + sum([to_polish_with_encoding(arg) for arg in expr.args], [])
    elif isinstance(expr, sym.Mul):
        return ['mul'] + sum([to_polish_with_encoding(arg) for arg in expr.args], [])
    elif isinstance(expr, sym.Pow):
        return ['pow'] + to_polish_with_encoding(expr.args[0]) + to_polish_with_encoding(expr.args[1])
    elif isinstance(expr, sym.sin):
        return ['sin'] + to_polish_with_encoding(expr.args[0])
    elif isinstance(expr, sym.cos):
        return ['cos'] + to_polish_with_encoding(expr.args[0])
    elif isinstance(expr, sym.Symbol):
        return [str(expr)]
    elif isinstance(expr, sym.Integer):
        return encode_integer_i10(int(expr))
    elif isinstance(expr, sym.Float):
        # if (expr == 5) or (expr == -5):
            # return ['+' if expr >= 0 else '-'] + ['gravity']
        # else:
        return encode_real_f10_e100(float(expr))
    elif isinstance(expr, sym.Expr) and expr.is_negative:
        return ['sub', '0'] + to_polish_with_encoding(-expr)
    return [str(expr)]

    