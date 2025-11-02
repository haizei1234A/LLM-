# -*- coding: utf-8 -*-
"""
build_json_ok_from_logs.py  

输入：bench_ollama_3stage.py 产生的 runs/*.jsonl
输出：CSV（用于 TOST/合规模块的分数输入）

对齐要点：
- 优先信任 bench 日志中的 json_ok/json_violation（可用 --recompute 强制重算）
- 可选 --schema 进行 JSON Schema 硬校验（AnswerContract）；失败即 0，并记录 violation
- 分层口径：task(json/general) × ctx_bucket(S/M/L/XL) × path(light/std/enh) × complexity_bucket(L/M/H)
- ITT：拒答/重试不在此处区分，JSON 合规仅针对需要 JSON 的样本计算
- Warm-only：跳过 in_warmup==True
- 阈值化验收：输出严格失败率与是否达标（≤0.5%），可写入 --summary-json
- 兼容下游：主输出列固定为 6 列：[id, metric, score, stratum, cluster, time_idx]
- 可选诊断表 --diag，包含 need_json/json_ok_log/json_ok_calc/violation 等

依赖：pandas
可选：jsonschema（如需 --schema）
"""

import os, re, json, argparse, glob, hashlib
import pandas as pd

# ---------- 读取日志 ----------
def load_jsonl_many(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    if not paths:
        raise FileNotFoundError(f"未找到日志：{patterns}")
    rows = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # 跳过坏行
                    continue
    return rows

# ---------- 分层桶 ----------
def bucket_ctx(prompt_len: int) -> str:
    L = int(prompt_len or 0)
    if L < 64: return "ctxS"
    if L < 256: return "ctxM"
    if L < 1024: return "ctxL"
    return "ctxXL"

def bucket_complexity(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v < 0.45: return "compL"
    if v < 0.90: return "compM"
    return "compH"

# ---------- 任务识别 ----------
def is_json_task(o):
    if o.get("need_json") is True:
        return True
    p = (o.get("prompt","") or "").lower()
    if "json" in p or "{" in p or "[" in p:
        return True
    return False

# ---------- 纯文本/结构判定 ----------
def json_bracket_like(t: str) -> bool:
    t = (t or "").strip()
    return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))

def try_json_load(t: str):
    try:
        return True, json.loads((t or "").strip()), ""
    except Exception as e:
        return False, None, str(e)[:120]

# ---------- 可选：Schema 校验 ----------
try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None

def make_validator(schema_path: str):
    if not schema_path:
        return None, None
    if Draft7Validator is None:
        raise RuntimeError("需要 jsonschema 依赖：pip install jsonschema")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    version = schema.get("$id") or schema.get("version") or os.path.basename(schema_path)
    return Draft7Validator(schema), str(version)

def validate_with_schema(obj, validator):
    try:
        validator.validate(obj)
        return True, ""
    except Exception as e:
        return False, f"schema:{str(e)[:160]}"

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, nargs="+",
                    help=r"一个或多个日志 glob，例如 runs/*.jsonl runs/baseline_*.jsonl")
    ap.add_argument("--out", required=True, help="输出分数 CSV，如 scores_gateway_json.csv")
    ap.add_argument("--json-only", action="store_true", help="仅输出 JSON 任务的记录（严格口径）")
    ap.add_argument("--cluster-col", default=None, help="cluster 列（默认使用日志中的 mode；如有 tenant/session 可指定）")
    ap.add_argument("--schema", default=None, help="JSON Schema（AnswerContract）路径，可选")
    ap.add_argument("--recompute", action="store_true",
                     help="忽略日志中的 json_ok，强制按本脚本重算（默认优先使用日志结果以保持一致性）")
    ap.add_argument("--skip-warmup-n", type=int, default=0,
                    help="强制跳过最早 N 条样本（按 time_idx；无则用 id）。与日志内 in_warmup 叠加。")
    ap.add_argument("--summary-json", default=None, help="写出严格子集失败率与指纹到此 JSON 文件")
    ap.add_argument("--diag", default=None, help="可选：输出诊断明细 CSV（含 violation 等）")
    ap.add_argument("--fail-threshold", type=float, default=0.005, help="验收阈值：JSON 失败率上限（默认 0.5%）")
    args = ap.parse_args()
    logs = load_jsonl_many(args.log)
    validator, schema_version = make_validator(args.schema)
    rows = []
    diag_rows = []
    for o in logs:
        def _is_forced_warmup(obj, n):
            if not n or n <= 0:
                return False
            ti = obj.get("time_idx", obj.get("id", 0))
            try:
                ti = int(ti or 0)
            except Exception:
                ti = 0
            return ti < n
        if o.get("in_warmup", False) or _is_forced_warmup(o, args.skip_warmup_n):
            continue

        # 任务属性
        need_json = is_json_task(o)
        if args.json_only and not need_json:
            continue

        # 文本与 bench 标注
        text = o.get("text") or o.get("response") or o.get("output") or ""
        json_ok_log = o.get("json_ok")
        json_violation_log = o.get("json_violation")

        # 计算或复用
        if (json_ok_log is not None) and (not args.recompute):
            ok = 1.0 if bool(json_ok_log) else 0.0
            violation = str(json_violation_log or ("ok" if ok == 1.0 else "unknown"))
            source = "log"
        else:
            # 以脚本口径重算
            source = "calc"
            if not need_json:
                ok, violation = 1.0, "not_required"
            else:
                if not text:
                    ok, violation = 0.0, "empty_output"
                else:
                    # 先结构，再解析，再 Schema
                    if not json_bracket_like(text):
                        ok, violation = 0.0, "not_bracket_json"
                    else:
                        parsed_ok, obj, err = try_json_load(text)
                        if not parsed_ok:
                            ok, violation = 0.0, f"not_json:{err}"
                        else:
                            if validator is not None:
                                v_ok, v_err = validate_with_schema(obj, validator)
                                ok = 1.0 if v_ok else 0.0
                                violation = "ok" if v_ok else (v_err or "schema_error")
                            else:
                                ok, violation = 1.0, "ok"

        # 分层
        task   = "json" if need_json else "general"
        ctxb   = bucket_ctx(o.get("prompt_len", 0))
        path   = str(o.get("final_path", "unk"))
        compb  = bucket_complexity(o.get("plan_complexity"))
        stratum = f"{task}|{ctxb}|{path}|{compb}"

        # cluster / time_idx
        cluster = o.get(args.cluster_col) if args.cluster_col else (o.get("mode", "session"))
        time_idx = o.get("time_idx", o.get("id", 0))

        rows.append({
            "id": int(o.get("id", 0) or 0),
            "metric": "json_ok",
            "score": float(ok),
            "stratum": stratum,
            "cluster": str(cluster),
            "time_idx": int(time_idx),
        })

        if args.diag:
            diag_rows.append({
                "id": int(o.get("id", 0) or 0),
                "need_json": bool(need_json),
                "json_ok_log": None if json_ok_log is None else bool(json_ok_log),
                "json_ok_calc": bool(ok == 1.0),
                "violation": violation,
                "source": source,
                "prompt_len": int(o.get("prompt_len", 0) or 0),
                "final_path": path,
                "plan_complexity": o.get("plan_complexity"),
            })

    # 写主输出（窄表）
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("没有样本；检查 --warmup / --json-only / 日志内容。")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] 写出分数表：{args.out}  共 {len(df)} 条")

    # 诊断可选
    if args.diag:
        ddf = pd.DataFrame(diag_rows)
        ddf.to_csv(args.diag, index=False, encoding="utf-8")
        print(f"[OK] 写出诊断表：{args.diag}  共 {len(ddf)} 条")

    # 严格子集失败率（仅 JSON 任务）
    df_json = df[df["stratum"].str.startswith("json|")]
    fail_rate = float("nan")
    pass_flag = None
    if not df_json.empty:
        fail_rate = 1.0 - float(df_json["score"].mean())
        pass_flag = bool(fail_rate <= float(args.fail_threshold))
        print(f"[JSON 严格失败率] ≈ {fail_rate*100:.3f}%   门限={args.fail_threshold*100:.2f}%   {'PASS' if pass_flag else 'FAIL'}")
    else:
        print("[提示] 当前样本中没有 JSON 任务（严格口径无法计算失败率）")

    # 审计指纹与摘要
    if args.summary_json:
        summary = {
            "n_total": int(len(df)),
            "n_json": int(len(df_json)),
            "json_fail_rate": None if pd.isna(fail_rate) else float(fail_rate),
            "threshold": float(args.fail_threshold),
            "pass": None if pass_flag is None else bool(pass_flag),
            "schema_version": schema_version,
            "validation_mode": ("schema+parse+brackets" if validator else "parse+brackets"),
            "source_preference": ("log" if not args.recompute else "recompute"),
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] 写出摘要：{args.summary_json}")

if __name__ == "__main__":
    main()

