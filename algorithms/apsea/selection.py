"""APSEA 算法的三种环境选择机制模块 (对应 EnvironmentalSelection_*.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array, Population
from algorithms.apsea.fitness import cal_fitness


def truncation(pop_obj: Array, k: int) -> list[int]:
    """与 PlatEMO 的 Truncation 函数完全一致的迭代剔除策略。

    Args:
        pop_obj: 目标函数值矩阵 (N, M)。
        k: 需要剔除的解数量。

    Returns:
        被剔除的解在 pop_obj 中的本地下标列表。
    """
    n = len(pop_obj)
    distances = np.linalg.norm(pop_obj[:, None, :] - pop_obj[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    deleted_mask = np.zeros(n, dtype=bool)

    while deleted_mask.sum() < k:
        remain = np.flatnonzero(~deleted_mask)
        sub_dist = distances[remain][:, remain]
        sub_dist_sorted = np.sort(sub_dist, axis=1)

        # 按照按行字典序最小找到最拥挤解（最先删除）
        remove_idx_in_remain = min(range(len(remain)), key=lambda i: tuple(sub_dist_sorted[i]))
        deleted_mask[remain[remove_idx_in_remain]] = True

    return np.flatnonzero(deleted_mask).tolist()


def take_pop(pop: Population, idx: Array) -> Population:
    """按下标提取种群。"""
    return Population(
        x=pop.x[idx],
        f=pop.f[idx],
        cv=pop.cv[idx],
        g=pop.g[idx] if pop.g is not None else None,
        h=pop.h[idx] if pop.h is not None else None,
    )


def environmental_selection_cdp(pop: Population, target_n: int) -> tuple[Population, Array]:
    """对应 EnvironmentalSelection_CDP.m：基于 CDP 约束支配的选择。"""
    fit = cal_fitness(pop.f, cv=pop.cv)

    next_mask = fit < 1.0
    if next_mask.sum() < target_n:
        rank = np.argsort(fit)
        next_mask[rank[:target_n]] = True
    elif next_mask.sum() > target_n:
        chosen_indices = np.flatnonzero(next_mask)
        num_to_del = len(chosen_indices) - target_n
        del_local_indices = truncation(pop.f[chosen_indices], num_to_del)
        next_mask[chosen_indices[del_local_indices]] = False

    chosen_final = np.flatnonzero(next_mask)
    selected_pop = take_pop(pop, chosen_final)
    selected_fit = fit[chosen_final]
    return selected_pop, selected_fit


def environmental_selection_no_constrained(pop: Population, target_n: int) -> tuple[Population, Array]:
    """对应 EnvironmentalSelection_noConstrained.m：完全忽视约束的选择。"""
    fit = cal_fitness(pop.f, None)

    next_mask = fit < 1.0
    if next_mask.sum() < target_n:
        rank = np.argsort(fit)
        next_mask[rank[:target_n]] = True
    elif next_mask.sum() > target_n:
        chosen_indices = np.flatnonzero(next_mask)
        num_to_del = len(chosen_indices) - target_n
        del_local_indices = truncation(pop.f[chosen_indices], num_to_del)
        next_mask[chosen_indices[del_local_indices]] = False

    chosen_final = np.flatnonzero(next_mask)
    selected_pop = take_pop(pop, chosen_final)
    selected_fit = fit[chosen_final]
    return selected_pop, selected_fit


def environmental_selection_epsilon(
    pop: Population, target_n: int, var_epsilon: float
) -> tuple[Population, Array]:
    """对应 EnvironmentalSelection_Epsilon.m：基于 epsilon 界限的双分层选择。"""
    is_feasible = pop.cv <= var_epsilon
    f_idx = np.flatnonzero(is_feasible)
    if_idx = np.flatnonzero(~is_feasible)

    f_pop = take_pop(pop, f_idx) if len(f_idx) > 0 else None
    if_pop = take_pop(pop, if_idx) if len(if_idx) > 0 else None

    return_pops: list[Population] = []
    return_fits: list[Array] = []

    if f_pop is None or len(f_pop.x) == 0:
        assert if_pop is not None
        if_fit = cal_fitness(if_pop.f, cv=if_pop.cv)
        next2 = if_fit < 1.0
        if next2.sum() <= target_n:
            rank = np.argsort(if_fit)
            next2[rank[:target_n]] = True
        elif next2.sum() > target_n:
            chosen = np.flatnonzero(next2)
            del_local = truncation(if_pop.f[chosen], len(chosen) - target_n)
            next2[chosen[del_local]] = False

        chosen_final = np.flatnonzero(next2)
        out_pop = take_pop(if_pop, chosen_final)
        out_fit = if_fit[chosen_final]
        rank = np.argsort(out_fit)
        return take_pop(out_pop, rank), out_fit[rank]

    elif len(f_pop.x) <= target_n:
        # 将 [f_pop.f, f_pop.cv] 作为多目标评价
        f_objs_with_cv = np.column_stack([f_pop.f, f_pop.cv])
        f_fit = cal_fitness(f_objs_with_cv, None)

        next1 = f_fit < 1.0
        rank1 = np.argsort(f_fit)
        next1[rank1[: len(f_pop.x)]] = True

        chosen_f = np.flatnonzero(next1)
        res_f_pop = take_pop(f_pop, chosen_f)
        res_f_fit = f_fit[chosen_f]
        rank_f = np.argsort(res_f_fit)
        res_f_pop = take_pop(res_f_pop, rank_f)
        res_f_fit = res_f_fit[rank_f]

        return_pops.append(res_f_pop)
        return_fits.append(res_f_fit)

        need_if = target_n - len(res_f_pop.x)
        if need_if > 0 and if_pop is not None and len(if_pop.x) > 0:
            if_fit = cal_fitness(if_pop.f, cv=if_pop.cv)
            next2 = if_fit < 1.0
            if next2.sum() <= need_if:
                rank2 = np.argsort(if_fit)
                next2[rank2[:need_if]] = True
            elif next2.sum() > need_if:
                chosen_if = np.flatnonzero(next2)
                del_local = truncation(if_pop.f[chosen_if], len(chosen_if) - need_if)
                next2[chosen_if[del_local]] = False

            chosen_if_final = np.flatnonzero(next2)
            res_if_pop = take_pop(if_pop, chosen_if_final)
            res_if_fit = if_fit[chosen_if_final] + (res_f_fit.max() if len(res_f_fit) else 0.0)
            rank_if = np.argsort(res_if_fit)
            res_if_pop = take_pop(res_if_pop, rank_if)
            res_if_fit = res_if_fit[rank_if]

            return_pops.append(res_if_pop)
            return_fits.append(res_if_fit)

    else:  # len(f_pop.x) > target_n
        f_objs_with_cv = np.column_stack([f_pop.f, f_pop.cv])
        f_fit = cal_fitness(f_objs_with_cv, None)
        next1 = f_fit < 1.0
        if next1.sum() <= target_n:
            rank1 = np.argsort(f_fit)
            next1[rank1[:target_n]] = True
        elif next1.sum() > target_n:
            chosen = np.flatnonzero(next1)
            del_local = truncation(f_pop.f[chosen], len(chosen) - target_n)
            next1[chosen[del_local]] = False

        chosen_final = np.flatnonzero(next1)
        res_f_pop = take_pop(f_pop, chosen_final)
        res_f_fit = f_fit[chosen_final]
        rank_f = np.argsort(res_f_fit)
        return_pops.append(take_pop(res_f_pop, rank_f))
        return_fits.append(res_f_fit[rank_f])

    # 归并种群与适应度
    merged_x = np.concatenate([p.x for p in return_pops], axis=0)
    merged_f = np.concatenate([p.f for p in return_pops], axis=0)
    merged_cv = np.concatenate([p.cv for p in return_pops], axis=0)
    merged_g = np.concatenate([p.g for p in return_pops if p.g is not None], axis=0) if return_pops[0].g is not None else None
    merged_h = np.concatenate([p.h for p in return_pops if p.h is not None], axis=0) if return_pops[0].h is not None else None

    final_pop = Population(x=merged_x, f=merged_f, cv=merged_cv, g=merged_g, h=merged_h)
    final_fit = np.concatenate(return_fits, axis=0)
    return final_pop, final_fit
