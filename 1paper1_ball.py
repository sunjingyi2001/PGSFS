# -*- coding: utf-8 -*-
"""《模板GPT版(2)》的粒球与数据编码阶段（固定参数实验版）。

重要：只接受拟合/训练行；外层测试数据不得输入。用户原始数据集无表头、
最后一列为单标签；分类属性的位置必须通过 categorical_indices 或先前的
"前 n_categorical 列"约定显式提供。保留 0 基原始属性索引。

用户确认：条件属性沿用原程序的缺失值补全（名义属性众数、实值属性均值并保留两位小数）；
决策/标签列保持原样，NaN 或 * 均表示无标记样本。对条件属性的预处理属于实验数据准备，不写入理论定义；拟合行以外数据
不得用于补全统计量。仅允许标签（*或空值）保持缺失。
"""
from __future__ import annotations
import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from joblib import dump
from scipy.linalg import solve

SQRT2 = float(np.sqrt(2.0))


def _key(v):
    return (type(v).__name__, repr(v))


def _missing(v):
    return pd.isna(v) or (isinstance(v, str) and v.strip() in ('?', ''))


def _impute_fit_conditions(condition_df, categorical_indices, ordered_maps):
    """遵循原程序：每个属性基于拟合行补全一次，不修改标签列。

    名义/有序类别取众数，并列众数取首次出现的值；数值取非缺失均值，
    round(..., 2) 后写入缺失位置。全缺失属性和非有限数值明确报错。
    注意：该步骤是论文理论模型以外的数据预处理，不调整算法核心定义。
    """
    completed = condition_df.copy().astype(object)
    categorical_indices = set(categorical_indices)
    ordered_indices = set(ordered_maps)
    info = []
    for a in range(condition_df.shape[1]):
        vals = completed.iloc[:, a].tolist()
        missing = np.asarray([_missing(v) for v in vals], dtype=bool)
        valid = [v for v, bad in zip(vals, missing) if not bad]
        if not valid:
            raise ValueError(f'条件属性 a_{a+1} 全部缺失，沿用原程序的明确报错规则')
        if a in categorical_indices or a in ordered_indices:
            # 使用键而非类别值本身，以兼容可能出现的混合类别类型。
            counts, representatives = {}, {}
            for value in valid:
                key = _key(value)
                counts[key] = counts.get(key, 0) + 1
                representatives.setdefault(key, value)
            most_frequent_key = max(counts, key=counts.get)
            fill_value = representatives[most_frequent_key]
            if missing.any():
                completed.iloc[np.flatnonzero(missing), a] = fill_value
            feature_type = 'categorical' if a in categorical_indices else 'ordered'
        else:
            numeric_valid = np.asarray(pd.to_numeric(pd.Series(valid), errors='raise'), dtype=float)
            if not np.isfinite(numeric_valid).all():
                raise ValueError(f'数值属性 a_{a+1} 的非缺失位置存在 inf 或非有限数值')
            fill_value = round(float(numeric_valid.mean()), 2)
            if missing.any():
                completed.iloc[np.flatnonzero(missing), a] = fill_value
            # 确保该列仍为数值类型，且非缺失值不额外四舍五入。
            completed.iloc[:, a] = np.asarray(
                pd.to_numeric(completed.iloc[:, a], errors='raise'), dtype=float)
            feature_type = 'numerical'
        info.append(dict(original_feature_index=a, feature_type=feature_type,
                         missing_count=int(missing.sum()), fill_value=fill_value))
    return completed, info


def _group_distances(points, center, groups):
    """(3.1)：对每个原始属性组取 L2，再按组数平均。"""
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    out = np.zeros(points.shape[0], dtype=float)
    for group in groups:
        cols = group['processed_indices']
        out += np.linalg.norm(points[:, cols] - center[cols], axis=1)
    return out / len(groups)


def _group_distances_pair(left, right, groups):
    left, right = np.asarray(left), np.asarray(right)
    out = np.zeros(len(right), dtype=float)
    for g in groups:
        c = g['processed_indices']
        out += np.linalg.norm(right[:, c] - left[c], axis=1)
    return out / len(groups)


def _encode_fit(df, categorical_indices, ordered_maps):
    """(2.2)--(2.7)：严格的训练行预处理与整组去重。"""
    n, p = df.shape
    categorical_indices = set(categorical_indices)
    ordered_maps = dict(ordered_maps)
    if any(x < 0 or x >= p for x in categorical_indices | set(ordered_maps)):
        raise ValueError('属性类型索引超出条件属性范围')
    if categorical_indices & set(ordered_maps):
        raise ValueError('有序属性不得同时声明为名义分类属性')
    raw = df.to_numpy(dtype=object)
    fitted = []
    removed_constants = []
    eq_classes = []
    candidates = []
    for a in range(p):
        vals = raw[:, a].tolist()
        if any(_missing(x) for x in vals):
            raise ValueError(f'原始属性 a_{a+1} 在预处理后仍含缺失值；请检查补全步骤')
        if a in categorical_indices:
            if any(isinstance(x,(int,float,np.number)) and not np.isfinite(x) for x in vals):
                raise ValueError(f'名义属性 a_{a+1} 存在非有限数值')
            categories = []
            seen = set()
            for v in vals:
                kv = _key(v)
                if kv not in seen:
                    seen.add(kv)
                    categories.append(v)
            if len(categories) == 1:
                removed_constants.append(a)
                continue
            cat_map = {_key(v): i for i, v in enumerate(categories)}
            ids = [cat_map[_key(v)] for v in vals]
            # P_a 不是字符串标号，而是样本索引的等价划分。
            partition = frozenset(frozenset(i for i, t in enumerate(ids) if t == k)
                                   for k in range(len(categories)))
            block = np.zeros((n, len(categories) + 1), dtype=float)
            block[np.arange(n), ids] = 1 / SQRT2
            fitted.append({'original_feature_index': a, 'feature_type': 'nom',
                           'raw_categories': categories, 'encoded': block,
                           'signature': partition})
        else:
            typ = 'ord' if a in ordered_maps else 'num'
            if typ == 'ord':
                order = ordered_maps[a]
                # 列表意味着等级依次映射 1,2,...；字典允许显式严格递增分数。
                if isinstance(order, dict):
                    ordered_items = list(order.items())
                else:
                    ordered_items = [(label, i+1) for i,label in enumerate(order)]
                if len({_key(label) for label,score in ordered_items}) != len(ordered_items):
                    raise ValueError(f'有序属性 a_{a+1} 的等级表含重复值')
                numeric_scores = np.asarray([score for label,score in ordered_items], dtype=float)
                if not np.isfinite(numeric_scores).all() or not np.all(np.diff(numeric_scores)>0):
                    raise ValueError(f'有序属性 a_{a+1} 的分数必须按声明顺序严格递增')
                lookup = {_key(label): float(score) for label,score in ordered_items}
                try:
                    v = np.asarray([lookup[_key(x)] for x in vals], dtype=float)
                except KeyError as e:
                    raise ValueError(f'有序属性 a_{a+1} 含未声明等级: {e}') from e
            else:
                try:
                    v = np.asarray(pd.to_numeric(pd.Series(vals), errors='raise'), dtype=float)
                except (TypeError, ValueError) as e:
                    raise ValueError(f'数值属性 a_{a+1} 出现无法转换的值') from e
            if not np.isfinite(v).all():
                raise ValueError(f'属性 a_{a+1} 存在非有限值')
            if len(set(v.tolist())) == 1:
                removed_constants.append(a)
                continue
            mu, std = float(v.mean()), float(v.std(ddof=0))
            if not std > 0:
                raise ValueError(f'属性 a_{a+1} 的非恒定标准差发生数值下溢')
            block = ((v - mu) / std)[:, None]
            fitted.append({'original_feature_index': a, 'feature_type': typ,
                           'mean': mu, 'std': std, 'encoded': block,
                           'signature': None, 'ordered_map': ordered_maps.get(a)})

    # 按原始索引顺序筛除相同类型的等价属性：数值标准化列精确相等/相反，
    # 有序属性与数值属性互不合并，名义属性按拟合行等价划分合并。
    for item in fitted:
        matches = False
        for keeper in candidates:
            if item['feature_type'] != keeper['feature_type']:
                continue
            if item['feature_type'] == 'nom':
                matches = item['signature'] == keeper['signature']
            else:
                va, vb = item['encoded'][:, 0], keeper['encoded'][:, 0]
                matches = np.array_equal(va, vb) or np.array_equal(va, -vb)
            if matches:
                for c in eq_classes:
                    if c[0] == keeper['original_feature_index']:
                        c.append(item['original_feature_index'])
                        break
                break
        if not matches:
            candidates.append(item)
            eq_classes.append([item['original_feature_index']])
    if not candidates:
        raise ValueError('去除恒定与等价属性后没有可用条件属性 A=∅')
    parts, groups = [], []
    cursor = 0
    for item in candidates:
        block = item.pop('encoded')
        item.pop('signature', None)
        parts.append(block)
        count = block.shape[1]
        item['processed_indices'] = list(range(cursor, cursor + count))
        item['n_fit_categories'] = (count - 1) if item['feature_type'] == 'nom' else None
        item['group_penalty_weight'] = 1.0  # 本文默认 nu_a=1
        cursor += count
        groups.append(item)
    Z = np.concatenate(parts, axis=1)
    return Z, groups, removed_constants, eq_classes


def _ball_statistics(Z, indices, groups, depth=0):
    indices = np.asarray(indices, dtype=int)
    center = Z[indices].mean(axis=0)
    radius = float(_group_distances(Z[indices], center, groups).mean())
    dispersions = np.asarray([np.mean(np.sum((Z[np.ix_(indices, g['processed_indices'])]
                                      - center[g['processed_indices']]) ** 2, axis=1))
                              for g in groups], dtype=float)
    return dict(indices=indices, size=int(len(indices)), center=center,
                radius=radius, group_dispersion=dispersions, depth=depth)


def _farthest_pair(Z, indices, groups):
    """(3.4) 稳定的最远样本对；同距离按 (i,j) 字典序取第一对。"""
    sorted_idx = np.sort(indices)
    best_pair, best_distance = None, 0.0
    for pos, u in enumerate(sorted_idx[:-1]):
        tail = sorted_idx[pos + 1:]
        dist = _group_distances_pair(Z[u], Z[tail], groups)
        k = int(np.argmax(dist))
        if dist[k] > best_distance:
            best_distance = float(dist[k])
            best_pair = (int(u), int(tail[k]))
    return best_pair, best_distance


def _balls_fit(Z, groups, nmin):
    n = len(Z)
    stack = [(np.arange(n, dtype=int), 0)]
    leaves, history = [], []
    while stack:
        indices, depth = stack.pop()
        parent = _ball_statistics(Z, indices, groups, depth)
        if len(indices) < 2 * nmin:
            leaves.append(parent)
            continue
        pair, max_distance = _farthest_pair(Z, indices, groups)
        if pair is None or max_distance == 0:
            leaves.append(parent)
            continue
        u, v = pair
        others = indices[(indices != u) & (indices != v)]
        du = _group_distances(Z[others], Z[u], groups)
        dv = _group_distances(Z[others], Z[v], groups)
        first = np.r_[u, others[du <= dv]]
        second = np.r_[v, others[du > dv]]
        if len(first) < nmin or len(second) < nmin:
            leaves.append(parent)
            continue
        b1 = _ball_statistics(Z, first, groups, depth + 1)
        b2 = _ball_statistics(Z, second, groups, depth + 1)
        weighted_radius = (b1['size'] * b1['radius'] + b2['size'] * b2['radius']) / len(indices)
        if not weighted_radius < parent['radius']:
            leaves.append(parent)
            continue
        history.append({'seeds': pair, 'depth': depth,
                        'parent_radius': parent['radius'],
                        'weighted_child_radius': weighted_radius})
        # LIFO：先入栈较大种子分支，后入栈较小种子分支。
        stack.append((second, depth + 1))
        stack.append((first, depth + 1))
    leaves.sort(key=lambda b: int(np.min(b['indices'])))
    for j, b in enumerate(leaves):
        b['ball_id'] = j
    coverage = np.sort(np.concatenate([b['indices'] for b in leaves]))
    if not np.array_equal(coverage, np.arange(n)):
        raise RuntimeError('分球未形成无重复的完整划分')
    return leaves, history


def _channel_fit(Z, groups, labeled_indices, beta, kL, kg, kappa_g,
                 kappa_e, bandwidth, nmin):
    """每个通道独立计算分球、B、T、Kβ、R，保证通道球编号不混用。"""
    n = len(Z)
    if len(labeled_indices) == 0:
        raise ValueError('至少需要一个已标记训练行')
    balls, split_history = _balls_fit(Z, groups, nmin=nmin)
    m, N = len(balls), len(labeled_indices)
    centers = np.stack([b['center'] for b in balls])
    radii = np.asarray([b['radius'] for b in balls])
    # source map (3.7): 只读标记行的索引和几何，不读标记值。
    B = np.zeros((m, N), dtype=float)
    e0 = np.zeros(m, dtype=float)
    labeled_lookup = {int(x): i for i, x in enumerate(labeled_indices)}
    for j, ball in enumerate(balls):
        shared = [labeled_lookup[int(i)] for i in ball['indices'] if int(i) in labeled_lookup]
        if shared:
            B[j, shared] = 1 / len(shared)
            e0[j] = len(shared) / (len(shared) + kappa_e)
            continue
        dist = _group_distances(Z[labeled_indices], centers[j], groups)
        zero = np.flatnonzero(dist == 0)
        if len(zero):
            B[j, zero] = 1 / len(zero)
        else:
            neighbors = np.lexsort((labeled_indices, dist))[:min(kL, N)]
            inv = 1 / dist[neighbors]
            B[j, neighbors] = inv / inv.sum()
    if not np.allclose(B.sum(axis=1), 1, atol=1e-12):
        raise RuntimeError('标签来源矩阵 B 非行随机')
    # Overlap (3.9) and mutually symmetrized graph (3.10).
    dcc = np.vstack([_group_distances(centers, c, groups) for c in centers])
    O = np.eye(m, dtype=float)
    for j in range(m):
        for q in range(j+1, m):
            den = radii[j] + radii[q]
            val = max(0, 1-dcc[j,q]/den) if den > 0 else float(dcc[j,q] == 0)
            O[j,q] = O[q,j] = val
    if kg and kappa_g:
        K_neigh = np.zeros((m,m), dtype=bool)
        for j in range(m):
            ids = [q for q in range(m) if q != j]
            ids.sort(key=lambda q: (dcc[j,q],q))
            K_neigh[j, ids[:min(kg,m-1)]] = True
        mutual = K_neigh | K_neigh.T
        O += kappa_g * mutual / (1+dcc**2)
        np.fill_diagonal(O, 1)
    T = O / O.sum(axis=1,keepdims=True)
    # Solve (I-beta T) K = (1-beta) I, no explicit inverse.
    Kbeta = solve(np.eye(m)-beta*T, (1-beta)*np.eye(m), assume_a='gen')
    Kbeta[Kbeta < 0] = np.maximum(Kbeta[Kbeta<0], 0)
    Kbeta /= Kbeta.sum(axis=1, keepdims=True)
    e_star = Kbeta @ e0
    # R (3.5)--(3.6): v_ja=mean ||z_ia-c_ja||², bandwidth per original group.
    upsilon = np.zeros((n,m), dtype=float)
    for k, g in enumerate(groups):
        c = g['processed_indices']
        h = float(bandwidth[g['original_feature_index']])
        dispersion = np.asarray([b['group_dispersion'][k] for b in balls])
        sq = np.sum((Z[:,None,c]-centers[None,:,c])**2, axis=2)
        upsilon -= np.log1p(sq / (dispersion[None,:]+h*h)) / len(groups)
    upsilon -= upsilon.max(axis=1, keepdims=True)
    if np.any(upsilon < np.log(np.nextafter(0., 1.))):
        raise FloatingPointError('R 归一化存在无法表示的正值；按论文要求报告失败')
    R = np.exp(upsilon)
    Omega = R / R.sum(axis=1,keepdims=True)
    Lambda = Omega @ Kbeta @ B
    if not np.allclose(Lambda.sum(axis=1),1,atol=1e-8):
        raise RuntimeError('来源传递矩阵 Lambda 非行随机')
    return dict(groups=groups, balls=balls, split_history=split_history,
                B=B, T=T, Kbeta=Kbeta, e0=e0, e_star=e_star,
                R=R, Omega=Omega, Lambda=Lambda, ball_count=m)


class ComputeMatrix:
    def __init__(self, name_ds, number, decision, n_categorical,
                 *, categorical_indices=None, ordered_maps=None,
                 input_filename=None, fitting_indices=None, run_tag=None,
                 nmin=2, kL=5, kg=5, kappa_g=1.0, kappa_e=1.0,
                 beta=0.8, bandwidth=1.0):
        self.name_ds = name_ds
        self.run_tag = run_tag
        self.output_stem = name_ds if run_tag is None else f'{name_ds}__{run_tag}'
        self.number = int(number)
        self.decision = int(decision)
        self.n_categorical = int(n_categorical)
        self.categorical_indices = (list(range(self.n_categorical)) if categorical_indices is None
                                    else list(categorical_indices))
        if len(self.categorical_indices) != self.n_categorical:
            raise ValueError('分类属性索引数与提供的分类属性数不一致')
        if len(set(self.categorical_indices)) != self.n_categorical:
            raise ValueError('分类属性索引不得重复')
        self.ordered_maps = ordered_maps or {}
        self.input_filename = input_filename or f'{name_ds}(0.25).xlsx'
        self.fitting_indices = fitting_indices
        self.nmin = int(nmin)
        self.kL = int(kL)
        self.kg, self.kappa_g = kg, kappa_g
        self.kappa_e, self.beta, self.bandwidth = kappa_e, beta, bandwidth
        root = Path(__file__).resolve().parent.parent
        self.dataset_dir = root/'Dataset'
        self.ball_dir = root/'GranularBalls'

    def GranularBallAnalysis(self):
        # 除 alpha/delta/lambda 之外，固定参数均按预先约定的实验设置执行。
        if not (self.nmin >= 1 and self.kL >= 1 and self.kg >= 0 and
                self.kappa_g >= 0 and self.kappa_e > 0 and 0 < self.beta < 1):
            raise ValueError('nmin/kL/kg/kappa_g/kappa_e/beta 越界')
        if self.nmin != 2 or self.kL != 5 or self.kg != 5 or self.kappa_g != 1.0 or self.kappa_e != 1.0 or self.beta != 0.8:
            raise ValueError('当前版本仅支持已确认的固定参数：nmin=2, kL=kg=5, kappa_g=kappa_e=1, beta=0.8')
        if not isinstance(self.bandwidth, (int, float, np.number)) or float(self.bandwidth) != 1.0:
            raise ValueError('当前版本固定所有原始属性带宽 h_a=1')
        if isinstance(self.bandwidth, (int, float, np.number)):
            widths = {j: float(self.bandwidth) for j in range(self.decision)}
        else:
            widths = dict(self.bandwidth)
        if any(a not in widths or not np.isfinite(widths[a]) or widths[a] <= 0
               for a in range(self.decision)):
            raise ValueError('每个原始条件属性需配置有限正带宽 h_a')
        raw = pd.read_excel(self.dataset_dir/self.input_filename, header=None)
        if raw.shape[1] != self.decision + 1:
            raise ValueError(f'实际 {raw.shape[1]} 列，期望 {self.decision}+1 列；最后一列必须是标签')
        if len(raw) != self.number:
            raise ValueError(f'实际 {len(raw)} 行，不等于声明样本数 {self.number}')
        # 只允许显式传入训练/拟合行。全表全输入意味着全表为拟合行，不能称为外层十折验证。
        fit_ids = np.arange(len(raw)) if self.fitting_indices is None else np.asarray(self.fitting_indices,dtype=int)
        if len(fit_ids)==0 or len(set(map(int,fit_ids)))!=len(fit_ids) or min(fit_ids)<0 or max(fit_ids)>=len(raw):
            raise ValueError('fitting_indices 必须为无重复且有效的训练行索引')
        fit_df = raw.iloc[fit_ids].reset_index(drop=True)
        # 先补条件属性，绝对不修改最后一列标签，也不使用标签决定填充值。
        condition_completed, imputation_info = _impute_fit_conditions(
            fit_df.iloc[:, :self.decision], self.categorical_indices, self.ordered_maps)
        # 新建 DataFrame 而非写入整数 dtype 的原 DataFrame，避免 pandas 类型不兼容警告。
        completed_full_df = pd.concat(
            [condition_completed.reset_index(drop=True),
             fit_df.iloc[:, self.decision:].reset_index(drop=True)], axis=1)
        completed_full_df.columns = fit_df.columns
        # 保存无表头的补全文件供核对；保留 * 和缺失标签原样。
        completed_dir = Path(__file__).resolve().parent.parent / 'CompletedDataset'
        completed_dir.mkdir(parents=True, exist_ok=True)
        completed_path = completed_dir / f'{self.output_stem}_complete.xlsx'
        completed_full_df.to_excel(completed_path, header=False, index=False)
        if any(item['missing_count'] for item in imputation_info):
            print('条件属性补全总数：', sum(item['missing_count'] for item in imputation_info))
        print('已保存补全数据（标签原样）：', completed_path)
        Z, groups, removed, equivalences = _encode_fit(condition_completed,
                                                       self.categorical_indices,self.ordered_maps)
        y_raw = fit_df.iloc[:,-1].to_numpy(dtype=object)
        labeled = np.asarray([i for i,v in enumerate(y_raw) if not pd.isna(v) and str(v).strip()!='*'],dtype=int)
        if not len(labeled):
            raise ValueError('拟合行必须至少包含一个已标记样本')
        channels = {}
        a_num = [g for g in groups if g['feature_type'] in ('num','ord')]
        a_cat = [g for g in groups if g['feature_type']=='nom']
        channel_groups = {'mix':groups}
        if a_num and a_cat:
            channel_groups.update(num=a_num,cat=a_cat)
        for name, subgroups in channel_groups.items():
            cols = [c for g in subgroups for c in g['processed_indices']]
            subZ = Z[:,cols]
            subgroups_local = []
            pos = 0
            for g in subgroups:
                g2 = dict(g)
                g2['processed_indices'] = list(range(pos,pos+len(g['processed_indices'])))
                pos += len(g['processed_indices'])
                subgroups_local.append(g2)
            # nmin 显式向通道级分球传播。
            channels[name] = _channel_fit(subZ,subgroups_local,labeled,self.beta,self.kL,
                                          self.kg,self.kappa_g,self.kappa_e,widths,self.nmin)
            channels[name]['global_processed_indices'] = cols
            print(f'{name} channel: {len(subgroups)} attrs, {len(channels[name]["balls"])} balls')
        state = dict(version='GB_TEMPLATE_GPT2_V1', dataset_name=self.name_ds,run_tag=self.run_tag,
                     n_samples=len(Z), original_fit_row_indices=fit_ids,
                     n_original_condition_attributes=self.decision,
                     n_categorical_attributes=self.n_categorical,
                     X=Z, feature_mapping=groups,
                     removed_constant_indices=removed,equivalence_classes=equivalences,
                     imputation_info=imputation_info,
                     completed_dataset_path=str(completed_path),
                     preprocessing_rule='categorical_mode_numeric_mean_round_2',
                     experiment_profile='ONLY_ALPHA_DELTA_LAMBDA_GRID',
                     y_raw=y_raw, labeled_indices=labeled,
                     unlabeled_indices=np.setdiff1d(np.arange(len(Z)),labeled),
                     channels=channels,
                     parameters={'nmin':self.nmin,'kL':self.kL,'kg':self.kg,
                                 'kappa_g':self.kappa_g,'kappa_e':self.kappa_e,
                                 'beta':self.beta,'bandwidth':widths},
                     protocol='fitting_rows_only')
        self.ball_dir.mkdir(parents=True,exist_ok=True)
        path=self.ball_dir/f'{self.output_stem}_gb01.joblib'
        dump(state,path,compress=3)
        rows = [{'channel':v,'ball_id':b['ball_id'],'size':b['size'],
                 'radius':b['radius'],'sample_indices':','.join(map(str,b['indices']))}
                for v,ch in channels.items() for b in ch['balls']]
        pd.DataFrame(rows).to_csv(self.ball_dir/f'{self.output_stem}_gb01.csv',index=False,encoding='utf-8-sig')
        print(f'已保存：{path}；保留原始属性 {len(groups)} / {self.decision}')
        return state


# ----------------------- 用户原有数据集入口（保持原列表，不改数值） -----------------------
if __name__=='__main__':

    # Fixed parameters for PGSFS
    K_G = 5
    KAPPA_G = 1.0
    KAPPA_E = 1.0
    H_BANDWIDTH = 1.0
    N_MIN = 2
    K_L = 5
    BETA = 0.8

    # 论文的有序属性必须提供等级映射；下面为空仅适用于没有有序属性的数据集。
    ORDERED_MAPS_BY_DATASET = {}
    # 默认与旧代码兼容：前 n_categorical 个属性为分类属性；其他排列请显式填写。
    CATEGORICAL_INDICES_BY_DATASET = {}

    dic_ds = {


        # 'Musk (Version 1)': ['Musk (Version 1)', 476, 166, 0],
        # 'Breast_Cancer': ['Breast_Cancer', 569, 30, 0],
        # 'Connectionist bench': ['Connectionist bench', 208, 60, 0],
        # 'Semeion handwritten': ['Semeion handwritten', 1593, 256, 256],
        # 'Gastrointestinal lesions': ['Gastrointestinal lesions', 152, 698, 0],
        # 'qsar': ['qsar', 1687, 1024, 1024],
        # 'Parkinsons_Classification_Complete': ['Parkinsons_Classification_Complete', 756, 753, 1],
        # '191_DARWIN': ['191_DARWIN', 174, 450, 0],
        # '180_Toxicity处理后的数据文件': ['180_Toxicity处理后的数据文件', 171, 1203, 163],



    }
    for name_ds,n_samples,n_cond,n_cat in dic_ds.values():
        print('\n运行数据集：',name_ds)
        ds=ComputeMatrix(name_ds,n_samples,n_cond,n_cat,
                         categorical_indices=CATEGORICAL_INDICES_BY_DATASET.get(name_ds),
                         ordered_maps=ORDERED_MAPS_BY_DATASET.get(name_ds),
                         nmin=N_MIN,kL=K_L,beta=BETA,
                         kg=K_G,kappa_g=KAPPA_G,kappa_e=KAPPA_E,
                         bandwidth=H_BANDWIDTH)
        ds.GranularBallAnalysis()
