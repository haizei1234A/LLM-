# -*- coding: utf-8 -*-
"""
TOST（质量非劣/等效）评测 ·
- 分层：支持“原样分层”(passthrough) 或 “简化”(simple=json/general)
- FDR：Benjamini–Hochberg 对“非劣侧”p 值做控制
- 稳健区间：优先 MBB（按 time_idx 排序），可指定/自动块长，并输出块长敏感性
- 后分层：支持 gateway/baseline/pooled，也可从 JSON/CSV 注入目标权重
- 功效与样本量：按“配对均值差”的常规模型估计 post-hoc 功效与 N@80%（近似）
- 硬约束：可加载 json_ok 指标，验证失败率 ≤ 0.5%
- 产出：runs/tost_report.csv | runs/tost_overall.csv | runs/TOST_design_memo.txt
"""

import argparse, pandas as pd, numpy as np, math, os, json, sys, importlib
from typing import Dict, Tuple, List

# ------------- I/O 与工具 -------------
PREFERRED_SCORE_KEYS = ("score", "value", "mean", "overall", "score_power", "accuracy", "acc")

# —— 正态分布：有 SciPy 用 SciPy，没 SciPy 用 statistics.NormalDist（统一走 _norm_ppf/_norm_cdf）——
try:
    _scipy_stats_spec = importlib.util.find_spec("scipy.stats")
except ModuleNotFoundError:
    _scipy_stats_spec = None

try:
    if _scipy_stats_spec is None:
        raise ModuleNotFoundError("scipy.stats not available")
    from scipy.stats import norm as _norm_dist  # type: ignore
except (ImportError, ModuleNotFoundError):
    from statistics import NormalDist
    _norm_dist = NormalDist()

    def _norm_ppf(p: float) -> float:
        return float(_norm_dist.inv_cdf(p))

    def _norm_cdf(x: float) -> float:
        return float(_norm_dist.cdf(x))
else:
    def _norm_ppf(p: float) -> float:
        return float(_norm_dist.ppf(p))

    def _norm_cdf(x: float) -> float:
        return float(_norm_dist.cdf(x))


def _coerce_score_from_nested(obj):
    if isinstance(obj, dict):
        for key in PREFERRED_SCORE_KEYS:
            if key in obj:
                try:
                    return _coerce_single_score(obj[key])
                except ValueError:
                    pass
        for value in obj.values():
            try:
                return _coerce_single_score(value)
            except ValueError:
                continue
        return float("nan")
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            try:
                return _coerce_single_score(value)
            except ValueError:
                continue
        return float("nan")
    return _coerce_single_score(obj)


def _sanitize_structured_string(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    # 宽松处理：允许使用单引号以及 float('nan') 这类 Python 表达
    cleaned = cleaned.replace("float('nan')", "NaN").replace('float("nan")', "NaN")
    cleaned = cleaned.replace("'", '"')
    return cleaned


def _coerce_single_score(raw) -> float:
    if isinstance(raw, (int, float, np.number)):
        return float(raw)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return float("nan")
    if isinstance(raw, str):
        text = raw.strip()
        lower = text.lower()
        if lower in {"nan", "null", "none"}:
            return float("nan")
        try:
            return float(text)
        except ValueError:
            if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
                sanitized = _sanitize_structured_string(text)
                if not sanitized:
                    return float("nan")
                try:
                    parsed = json.loads(sanitized)
                except Exception:
                    parsed = None
                if parsed is not None:
                    return _coerce_score_from_nested(parsed)
    if isinstance(raw, dict) or isinstance(raw, (list, tuple, set)):
        return _coerce_score_from_nested(raw)
    raise ValueError(f"无法解析 score 值：{raw!r}")


def read_scores(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    need = {"id","metric","score","stratum"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p} 缺少列 {need}")
    # 明确将 score 列转为数值；若包含对象/字典会导致后续 float(...) 失败
    score_series = pd.to_numeric(df["score"], errors="coerce")
    if score_series.isna().any():
        score_series = score_series.copy()
        mask = score_series.isna()
        for idx, raw in df.loc[mask, "score"].items():
            try:
                coerced = _coerce_single_score(raw)
            except Exception:
                continue
            score_series.at[idx] = coerced
        mask = score_series.isna()
        if mask.any():
            bad = df.loc[mask, ["id", "metric", "score"]].head(5)
            hint = bad.to_dict(orient="records")
            raise ValueError(f"{p} 的 score 列包含无法解析为数值的取值，例如: {hint}")
    df["score"] = score_series.astype(float)
    if "cluster" not in df.columns: df["cluster"] = "c"
    if "time_idx" not in df.columns: df["time_idx"] = np.arange(len(df))
    return df

def load_target_mix(path: str) -> Dict[str, float]:
    if not path:
        return {}
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {str(k): float(v) for k,v in m.items()}
    # 允许 CSV 两列：stratum,weight
    df = pd.read_csv(path)
    if not {"stratum","weight"}.issubset(df.columns):
        raise ValueError("外部权重文件需包含列：stratum, weight")
    return {str(r["stratum"]): float(r["weight"]) for _,r in df.iterrows()}

def normalize_strata(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "simple":
        df = df.copy()
        df["stratum"] = df["stratum"].apply(lambda s: "json" if str(s).lower()=="json" else "general")
        return df
    # passthrough：保持上游构造的多维分层（如 json|ctxL|enh|compH|...）
    return df

def merge_pair(baseline: str, gateway: str, metric: str, strata_mode: str) -> pd.DataFrame:
    db = read_scores(baseline); dg = read_scores(gateway)
    db = db[db["metric"]==metric].copy()
    dg = dg[dg["metric"]==metric].copy()
    db = normalize_strata(db, strata_mode)
    dg = normalize_strata(dg, strata_mode)
    df = db.merge(dg, on=["id","metric","stratum"], suffixes=("_b","_g"))
    if df.empty:
        bad_b = db.groupby("stratum")["id"].nunique().to_dict()
        bad_g = dg.groupby("stratum")["id"].nunique().to_dict()
        raise ValueError(f"两组没有可配对样本（id/metric/stratum 不匹配）。baseline={bad_b}, gateway={bad_g}")
    df["diff"] = df["score_g"] - df["score_b"]
    # time 轴统一为 gateway
    if "time_idx_g" in df.columns: df["time_idx"] = df["time_idx_g"]
    elif "time_idx_b" in df.columns: df["time_idx"] = df["time_idx_b"]
    else: df["time_idx"] = np.arange(len(df))
    # cluster 优先 gateway
    df["cluster_use"] = df["cluster_g"] if "cluster_g" in df.columns else df["cluster_b"] if "cluster_b" in df.columns else "c"
    return df

# ------------- Bootstrap（含 MBB） -------------
def mbb_sample_mean(x: np.ndarray, block: int, rng: np.random.Generator) -> float:
    n = len(x)
    if n == 0: return np.nan
    b = int(max(1, min(block, n)))
    starts = rng.integers(0, n, size=math.ceil(n/b))
    out = []
    for s in starts:
        seg = x[s:s+b]
        if len(seg) < b:
            seg = np.concatenate([seg, x[:b-len(seg)]])
        out.append(seg)
    take = np.concatenate(out)[:n]
    return float(np.mean(take))

def cluster_bootstrap_mean(x_by_cluster: np.ndarray, rng: np.random.Generator) -> float:
    if len(x_by_cluster) == 0: return float("nan")
    picks = rng.choice(x_by_cluster, size=len(x_by_cluster), replace=True)
    return float(np.mean(picks))

def iid_bootstrap_mean(x: np.ndarray, rng: np.random.Generator) -> float:
    if len(x) == 0: return float("nan")
    picks = rng.choice(x, size=len(x), replace=True)
    return float(np.mean(picks))

def choose_block_length(n: int) -> int:
    # 经验法则：n^(1/3) 四舍五入
    return max(1, int(round(n ** (1/3))))

def boot_stratum(sub: pd.DataFrame, B: int, rng: np.random.Generator, mbb_b: int = None) -> Tuple[np.ndarray, str, int]:
    x = sub.sort_values("time_idx")["diff"].to_numpy()
    n = len(x)
    if n == 0:
        return np.array([]), "none", 0
    if sub["time_idx"].nunique() > 1:
        b = mbb_b if (mbb_b and mbb_b > 0) else choose_block_length(n)
        boot = np.array([mbb_sample_mean(x, b, rng) for _ in range(B)], dtype=float)
        return boot, "mbb", b
    # 聚类自助
    clusters = sub["cluster_use"]
    if clusters.nunique() > 1:
        grp = sub.groupby(clusters)["diff"].mean().to_numpy()
        boot = np.array([cluster_bootstrap_mean(grp, rng) for _ in range(B)], dtype=float)
        return boot, "cluster", 0
    # IID
    boot = np.array([iid_bootstrap_mean(x, rng) for _ in range(B)], dtype=float)
    return boot, "iid", 0

def boot_overall(df: pd.DataFrame, B: int, target_mix: Dict[str,float], rng: np.random.Generator,
                 strata_b_override: Dict[str,int] = None) -> Tuple[Dict[str,np.ndarray], np.ndarray, Dict[str,float], Dict[str,Tuple[str,int]]]:
    strata = sorted(df["stratum"].unique())
    # 权重
    if not target_mix:
        w = pd.Series({s: float((df["stratum"]==s).mean()) for s in strata})
    else:
        w = pd.Series(target_mix).reindex(strata).fillna(0.0)
    if w.sum() == 0:
        w = pd.Series({s: 1.0/len(strata) for s in strata})
    w = w / w.sum()

    boot_by, methods = {}, {}
    for s in strata:
        sub = df[df["stratum"]==s].copy()
        b_override = (strata_b_override or {}).get(s, None)
        boot_s, how, b = boot_stratum(sub, B, rng, mbb_b=b_override)
        boot_by[s] = boot_s
        methods[s] = (how, b)

    overall = []
    for i in range(B):
        acc = 0.0
        for s in strata:
            arr = boot_by[s]
            if arr.size == 0:
                continue
            acc += w[s] * float(arr[i])
        overall.append(acc)
    return boot_by, np.array(overall, dtype=float), w.to_dict(), methods

# ------------- 统计与判定 -------------
def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = np.array([(i+1)*alpha/m for i in range(m)])
    mask = np.zeros(m, dtype=bool)
    k = -1
    for r, idx in enumerate(order):
        if pvals[idx] <= thresh[r]:
            mask[idx] = True
            k = r
    if k < 0:
        return mask
    cutoff = thresh[k]
    return pvals <= cutoff

def summarize_tost(boot: np.ndarray, eps: float) -> Dict[str, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot.size>0 else (float("nan"), float("nan"))
    # TOST：两单侧
    p_noninf = float(np.mean(boot <= -eps)) if boot.size>0 else float("nan")  # H0: μ_g - μ_b ≤ -ε
    p_notsup = float(np.mean(boot >= +eps)) if boot.size>0 else float("nan")  # H0: μ_g - μ_b ≥ +ε
    return dict(ci95=(float(lo), float(hi)), p_noninferior=p_noninf, p_not_superior=p_notsup)

def approx_power_and_n(diff_mean: float, se: float, eps: float, alpha: float = 0.05, power_target: float = 0.8) -> Dict[str, float]:
    """
    近似功效与样本量（配对均值差；正态近似）
    - 非劣检验：H0: Δ ≤ -ε, H1: Δ > -ε
    - 使用当前样本的均值差与 SE 做 post-hoc 功效；反推 N 以达到 80% 功效（用 z 近似）
    """
    if se <= 1e-12:
        return dict(posthoc_power=1.0, n_factor_for_80pwr=1.0)
    z_alpha = _norm_ppf(1.0 - alpha)  # 单侧
    # 设定“备择距离”：Δ_obs + ε（到门槛的距离）
    delta_eff = (diff_mean + eps)
    z = delta_eff / se
    # 关键修复：统一用 _norm_cdf，避免 scipy 依赖与未定义的 st
    posthoc_power = float(_norm_cdf(z - z_alpha))

    # 反推 N 以达 80% 功效：se_new = se * sqrt(n_old / n_new)
    # 目标： (delta_eff / se_new) - z_alpha = z_beta  ==> se_new = delta_eff / (z_alpha + z_beta)
    z_beta = _norm_ppf(power_target)
    if delta_eff <= 1e-12:
        n_factor = float("inf")
    else:
        se_new = delta_eff / (z_alpha + z_beta)
        n_factor = (se / max(se_new,1e-12))**2
    return dict(posthoc_power=max(0.0, min(1.0, posthoc_power)), n_factor_for_80pwr=n_factor)

# ------------- 主程序 -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--epsilon", type=float, default=0.01, help="默认 ε；可被 --epsilon-map 覆盖")
    ap.add_argument("--epsilon-map", default=None, help="可选：JSON 映射 {stratum: ε}")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--equiv", action="store_true", help="两侧均过视为等效，否则只判非劣")
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", choices=["gateway","baseline","pooled"], default="gateway")
    ap.add_argument("--target-mix", default=None, help="可选：外部目标权重 JSON/CSV（列: stratum,weight）")
    ap.add_argument("--strata-mode", choices=["passthrough","simple"], default="passthrough")
    ap.add_argument("--mbb-b", type=int, default=0, help="全局 MBB 块长（0=自动 n^(1/3)）")
    ap.add_argument("--mbb-sensitivity", default="5,10,20", help="报告敏感性 b 候选（逗号分隔），仅对用到 MBB 的分层汇报 CI 宽度")
    ap.add_argument("--min-n", type=int, default=10, help="每个分层的最小样本数（不足仍计算，但在报表标记 low_n）")
    ap.add_argument("--json-metric", default=None, help="baseline_json.csv,gateway_json.csv（硬门：失败率 ≤ 0.5%）")
    ap.add_argument("--design-notes", default="", help="可选：将 ε 来源/监管锚点等说明写入设计备忘录")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    eps_map = {}
    if args.epsilon_map:
        with open(args.epsilon_map, "r", encoding="utf-8") as f:
            eps_map = {str(k): float(v) for k,v in json.load(f).items()}

    # 读取并合并样本
    df = merge_pair(args.baseline, args.gateway, args.metric, strata_mode=args.strata_mode)

    # 目标权重
    target_mix = {}
    if args.target_mix:
        target_mix = load_target_mix(args.target_mix)
    elif args.target == "gateway":
        target_mix = (df.groupby("stratum")["score_g"].size()/len(df)).to_dict()
    elif args.target == "baseline":
        target_mix = (df.groupby("stratum")["score_b"].size()/len(df)).to_dict()
    else:
        t1 = (df.groupby("stratum")["score_g"].size()/len(df))
        t2 = (df.groupby("stratum")["score_b"].size()/len(df))
        target_mix = (0.5*(t1.add(t2, fill_value=0))).fillna(0).to_dict()

    # 分层块长覆盖：可全局，也可按分层覆写（本版仅全局；需要时可扩展）
    strata_b_override = {}

    # Bootstrap
    boot_by, overall_boot, weights, methods = boot_overall(
        df, B=args.B, target_mix=target_mix, rng=rng,
        strata_b_override=strata_b_override if args.mbb_b<=0 else {s:args.mbb_b for s in df["stratum"].unique()}
    )

    # 分层 TOST + FDR
    per_rows, pvals = [], []
    sensitivity_b = [int(x) for x in str(args.mbb_sensitivity).split(",") if str(x).strip().isdigit()]
    for s, boot in boot_by.items():
        eps_s = eps_map.get(s, args.epsilon)
        mean_diff = float(np.mean(boot)) if boot.size>0 else float("nan")
        r = summarize_tost(boot, eps_s)
        pass_noninf = (r["p_noninferior"] < args.alpha)
        pass_equiv  = (pass_noninf and (r["p_not_superior"] < args.alpha)) if args.equiv else pass_noninf

        # 近似 SE 与功效/样本量（用 CI 宽度反推）
        ci_lo, ci_hi = r["ci95"]
        se_approx = float((ci_hi - ci_lo) / (2.0 * 1.96)) if (not math.isnan(ci_lo) and not math.isnan(ci_hi)) else float("nan")
        power_info = approx_power_and_n(mean_diff, se_approx, eps_s, alpha=args.alpha) if not math.isnan(se_approx) else {"posthoc_power": float("nan"), "n_factor_for_80pwr": float("nan")}

        # MBB 敏感性：只在采用 MBB 的分层上报告不同 b 的 CI 宽度（不改变主判定）
        sens = {}
        how, b_used = methods.get(s, ("none", 0))
        if how == "mbb" and boot.size>0 and len(sensitivity_b)>0:
            x = df[df["stratum"]==s].sort_values("time_idx")["diff"].to_numpy()
            for bb in sensitivity_b:
                tmp = np.array([mbb_sample_mean(x, bb, rng) for _ in range(min(2000, args.B))], dtype=float)
                lo2, hi2 = np.percentile(tmp, [2.5, 97.5])
                sens[str(bb)] = round(float(hi2 - lo2), 6)

        n_s = int((df["stratum"]==s).sum())
        per_rows.append(dict(
            metric=args.metric, stratum=s, n=n_s, low_n=(n_s < args.min_n),
            weight=round(weights.get(s,0.0),6),
            epsilon=eps_s,
            mean_diff=round(mean_diff,6),
            ci95=f"[{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]",
            p_noninferior=round(r["p_noninferior"],6),
            p_not_superior=round(r["p_not_superior"],6),
            pass_noninferior=pass_noninf,
            pass_equiv=pass_equiv,
            method=how, mbb_block=b_used,
            ci_width=round(float(r["ci95"][1]-r["ci95"][0]),6),
            se_approx=round(se_approx,6) if not math.isnan(se_approx) else None,
            posthoc_power=round(power_info.get("posthoc_power", float("nan")),6),
            n_factor_for_80pwr=round(power_info.get("n_factor_for_80pwr", float("nan")),3),
            mbb_sensitivity=sens if sens else None
        ))
        pvals.append(r["p_noninferior"])

    per = pd.DataFrame(per_rows)
    if not per.empty:
        per["pass_noninferior_fdr"] = bh_fdr(per["p_noninferior"].to_numpy(), alpha=args.alpha)

    # 总体（后分层）
    overall = summarize_tost(overall_boot, args.epsilon)  # 总体用默认 ε，仅作参考；严格判定仍以分层 + FDR
    overall_mean = float(np.mean(overall_boot))
    ov = pd.DataFrame([dict(
        metric=args.metric, scope="overall_poststrat",
        mean_diff=round(overall_mean,6),
        ci95=f"[{overall['ci95'][0]:.6f}, {overall['ci95'][1]:.6f}]",
        p_noninferior=round(overall["p_noninferior"],6),
        p_not_superior=round(overall["p_not_superior"],6),
        pass_noninferior=(overall["p_noninferior"]<args.alpha),
        pass_equiv=((overall["p_noninferior"]<args.alpha) and (overall["p_not_superior"]<args.alpha)) if args.equiv else (overall["p_noninferior"]<args.alpha)
    )])

    # 硬约束：JSON 失败率 ≤ 0.5%
    hard_guard = {}
    if args.json_metric:
        try:
            b_csv, g_csv = [p.strip() for p in args.json_metric.split(",")]
            jb = read_scores(b_csv); jg = read_scores(g_csv)
            jb = jb[jb["metric"]=="json_ok"]; jg = jg[jg["metric"]=="json_ok"]
            fail_b = 1.0 - float(jb["score"].mean()) if not jb.empty else float("nan")
            fail_g = 1.0 - float(jg["score"].mean()) if not jg.empty else float("nan")
            hard_guard = {
                "json_fail_baseline": round(fail_b,6),
                "json_fail_gateway":  round(fail_g,6),
                "json_guard_pass": (fail_g<=0.005) if not math.isnan(fail_g) else False
            }
        except Exception as e:
            hard_guard = {"json_error": str(e)}

    # FDR 门并入总体结论（2.2）
    fdr_gate_pass = True if per.empty else bool(per["pass_noninferior_fdr"].all())
    ov.loc[0,"fdr_gate_pass"] = fdr_gate_pass
    if hard_guard:
        ov.loc[0,"json_guard_pass"] = hard_guard.get("json_guard_pass", False)

    ov.loc[0,"pass_noninferior"] = bool(ov.loc[0,"pass_noninferior"] and fdr_gate_pass and (hard_guard.get("json_guard_pass", True)))
    ov.loc[0,"pass_equiv"]       = bool(ov.loc[0,"pass_equiv"]       and fdr_gate_pass and (hard_guard.get("json_guard_pass", True)))

    # ---------- 输出 ----------
    os.makedirs("runs", exist_ok=True)
    per_path = os.path.join("runs","tost_report.csv")
    ov_path  = os.path.join("runs","tost_overall.csv")
    memo_path= os.path.join("runs","TOST_design_memo.txt")
    per.to_csv(per_path, index=False, encoding="utf-8")
    ov.to_csv(ov_path, index=False, encoding="utf-8")

    memo = {
        "metric": args.metric,
        "epsilon_default": args.epsilon,
        "epsilon_map": eps_map or "(default for all strata)",
        "alpha": args.alpha,
        "equivalence_mode": bool(args.equiv),
        "bootstrap_B": int(args.B),
        "seed": int(args.seed),
        "strata_mode": args.strata_mode,
        "mbb_block_global": int(args.mbb_b),
        "mbb_sensitivity_candidates": sensitivity_b,
        "per_stratum_method_block": {s: {"method": m[0], "mbb_block": m[1]} for s,m in methods.items()},
        "weight_target": args.target,
        "poststrat_weights": {k: round(v,6) for k,v in weights.items()},
        "hard_guard_json": hard_guard or "(not provided)",
        "min_n": int(args.min_n),
        "design_notes": args.design_notes or "",
        "conclusion": {
            "fdr_gate_pass": bool(fdr_gate_pass),
            "json_guard_pass": bool(hard_guard.get("json_guard_pass", True)) if hard_guard else True,
            "overall_pass_noninferior": bool(ov.loc[0,"pass_noninferior"]),
            "overall_pass_equiv": bool(ov.loc[0,"pass_equiv"])
        },
        "reminders": [
            "若任一关键层 FDR 后未通过 ⇒ 总体失败（2.2）",
            "建议对低样本(low_n)分层扩容或合并；报告中已标注",
            "功效为近似值；如需严格样本量，请在方案阶段用事前功效分析/保留效应论证"
        ]
    }
    with open(memo_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(memo, ensure_ascii=False, indent=2))

    print("\n=== 分层 TOST（含 FDR） ===")
    if not per.empty:
        print(per.to_string(index=False))
    print("\n=== 总体（后分层） ===")
    print(ov.to_string(index=False))
    print(f"\n已输出：{per_path} | {ov_path} | {memo_path}")

if __name__ == "__main__":
    main()
