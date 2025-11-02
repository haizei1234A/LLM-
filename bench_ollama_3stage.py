# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, time, math, uuid, glob, random, argparse, datetime, platform, re, hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable, Set

import numpy as np
import pandas as pd
import requests
import threading
from functools import lru_cache

# ===== HTTP 安全包装：每线程独立 Session；(connect, read) 双超时；keep-alive 可控 =====
_TL = threading.local()
_HTTP_KEEPALIVE_MODE = "auto"  # "auto"=Windows关/其它开；"on"=总开；"off"=总关

def set_http_keepalive(mode: str) -> None:
    global _HTTP_KEEPALIVE_MODE
    _HTTP_KEEPALIVE_MODE = mode or "auto"

def _resolve_keepalive(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    if _HTTP_KEEPALIVE_MODE == "on":
        return True
    if _HTTP_KEEPALIVE_MODE == "off":
        return False
    # auto：Windows 默认关，其它开
    return (os.name != "nt")

def _get_sess():
    s = getattr(_TL, "s", None)
    if s is None:
        s = requests.Session()
        _TL.s = s
    return s

def post_json(url: str, payload: dict, read_timeout_s: float, keepalive: Optional[bool] = None):
    sess = _get_sess()
    use_keepalive = _resolve_keepalive(keepalive)
    headers = {} if use_keepalive else {"Connection": "close"}
    # 连接超时=5s，读取超时=read_timeout_s（建议 60s）
    return sess.post(url, json=payload, headers=headers, timeout=(5, float(read_timeout_s)))
# =========================================================================================

# ---------- 可选依赖 ----------
try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None

try:
    from tdigest import TDigest
    import tdigest as _td_mod
    TDIGEST_VERSION = getattr(_td_mod, "__version__", "unknown")
except Exception:
    TDigest = None
    TDIGEST_VERSION = "unknown"

try:
    import tiktoken
except Exception:
    tiktoken = None

# ---------- 常量 ----------
EPS = 1e-12
MIN_JSON_LEN = 40
DEFAULT_P95_DELTA = 0.10
DEFAULT_JSON_FAIL_MAX = 0.005

BILLING_VERSION = "vNext-5part"
TDIGEST_DEFAULT_COMPRESSION = 200

# ---------- Gold 规则（质量标签） ----------
def load_gold_rules(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return {}
    rules: Dict[int, Dict[str, Any]] = {}
    bad = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            try:
                rid = int(obj.get("id", 0))
            except Exception:
                rid = 0
            if rid <= 0:
                bad += 1
                continue
            must_include = [str(s).lower() for s in obj.get("must_include", []) if s]
            regex_any = [str(p) for p in obj.get("regex_any", []) if p]
            rules[rid] = {
                "must_include": must_include,
                "regex_any": regex_any,
            }
    if bad:
        print(f"[WARN] load_gold_rules: skipped {bad} invalid line(s) in {path}")
    return rules


def _judge_by_rule(text: str, rule: Dict[str, Any]) -> float:
    if not rule:
        return float("nan")
    t_lower = (text or "").lower()
    for s in rule.get("must_include", []):
        if s and s in t_lower:
            return 1.0
    for pat in rule.get("regex_any", []):
        try:
            if re.search(pat, text or "", flags=re.I | re.S):
                return 1.0
        except re.error:
            continue
    return 0.0

# ---------- 环境 ----------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_KEEPALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
LIGHT_MODEL = os.getenv("OLLAMA_LIGHT", "qwen3:4b")
STD_MODEL   = os.getenv("OLLAMA_STD",   "qwen3:8b")
ENH_MODEL   = os.getenv("OLLAMA_ENH",   "qwen3:14b")

# ---------- 工具函数：分位数 ----------
def _hd(values: Iterable[float], q: float) -> float:
    arr = np.array([float(v) for v in values if not pd.isna(v)], dtype=float)
    n = arr.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(arr[0])
    q = max(EPS, min(1.0 - EPS, float(q)))
    weights = _hd_weights(n, q)
    return float(np.dot(weights, np.sort(arr)))


@lru_cache(maxsize=None)
def _hd_weights(n: int, q: float) -> np.ndarray:
    # Clamp q again inside the cached helper to ensure consistent keys.
    q = max(EPS, min(1.0 - EPS, float(q)))
    i = np.arange(1, n + 1, dtype=float)
    a = i
    b = n - i + 1.0
    log_q = math.log(q)
    log_1q = math.log1p(-q)
    log_beta = np.fromiter(
        (math.lgamma(aa) + math.lgamma(bb) - math.lgamma(aa + bb) for aa, bb in zip(a, b)),
        dtype=float,
        count=n,
    )
    log_weights = (i - 1.0) * log_q + (n - i) * log_1q - log_beta
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return weights

def q50(values: Iterable[float]) -> float:
    return _hd(values, 0.5)

def q95(values: Iterable[float]) -> float:
    return _hd(values, 0.95)

def tdigest_quantile(values: Iterable[float], q: float, compression: int = TDIGEST_DEFAULT_COMPRESSION) -> float:
    vals = [float(v) for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    if TDigest is None:
        return _hd(vals, q)
    d = TDigest(compression=compression)
    for v in vals:
        d.update(v)
    return float(d.quantile(q))

# ---------- CO-corrected ----------
def co_correct_periodic_fill(lat_ms_list: Iterable[float], period_ms: int, max_fills: int = 120) -> List[float]:
    if not period_ms or period_ms <= 0:
        return [float(x) for x in lat_ms_list if not pd.isna(x)]
    out: List[float] = []
    for L in lat_ms_list:
        if pd.isna(L):
            continue
        L = float(L)
        out.append(L)
        if L <= period_ms:
            continue
        kmax = min(int((L - EPS) // period_ms), max_fills)
        for j in range(1, kmax + 1):
            over = L - j * period_ms
            if over > 0:
                out.append(over)
            else:
                break
    return out

# ---------- KM 分位（右删失） ----------
def km_quantile(times_ms: Iterable[float], observed: Iterable[bool], q: float = 0.95) -> float:
    arr = [(float(t), bool(o)) for t, o in zip(times_ms, observed) if not pd.isna(t)]
    if not arr:
        return np.nan
    arr.sort(key=lambda x: x[0])
    uniq = sorted(set(t for t, _ in arr))
    n_total = len(arr)
    s = 1.0
    idx = 0
    for t in uniq:
        while idx < n_total and arr[idx][0] < t:
            idx += 1
        n_i = n_total - idx
        d_i = sum(1 for tt, ob in arr if tt == t and ob)
        if n_i <= 0:
            continue
        s *= (1.0 - d_i / n_i)
        if 1.0 - s >= q:
            return float(t)
    return np.nan

# ---------- Winsor（敏感性） ----------
def winsorize(values: Iterable[float], p: float = 0.999) -> List[float]:
    xs = [float(v) for v in values if not pd.isna(v)]
    if not xs:
        return xs
    lo = _hd(xs, 1 - p); hi = _hd(xs, p)
    return [min(max(v, lo), hi) for v in xs]

# ---------- Bootstrap ----------
def _percentile_ci(samples: np.ndarray, ci: float) -> Tuple[float, float]:
    alpha = (1 - ci) / 2
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1 - alpha))
    return theta_hat, lo, hi

def _bca_ci(samples: np.ndarray, theta_hat: float, jack: np.ndarray, ci: float) -> Tuple[float, float]:
    prop = np.mean(samples < theta_hat) + EPS
    from math import erfinv
    z0 = math.sqrt(2) * erfinv(2 * prop - 1)
    jmean = float(np.mean(jack))
    num = np.sum((jmean - jack) ** 3)
    den = 6.0 * (np.sum((jmean - jack) ** 2) ** 1.5 + EPS)
    a = float(num / (den + EPS))
    def z(p: float) -> float:
        from math import erfinv
        return math.sqrt(2) * erfinv(2 * p - 1)
    alpha = (1 - ci) / 2
    zlo, zhi = z(alpha), z(1 - alpha)
    def pct(adj_z: float) -> float:
        from math import erf
        p = 0.5 * (1 + erf(((z0 + adj_z) / (1 - a * (z0 + adj_z))) / math.sqrt(2)))
        p = max(0.0, min(1.0, p))
        return float(np.quantile(samples, p))
    return pct(zlo), pct(zhi)

def bootstrap_stat_ci(
    sample: Iterable[float],
    stat_fn: Callable[[np.ndarray], float],
    B: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
    method: str = "bca",
) -> Tuple[float, float, float]:
    x = np.array([v for v in sample if not pd.isna(v)], dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    theta_hat = float(stat_fn(x))
    thetas = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        thetas[b] = float(stat_fn(x[idx]))
    if method == "percentile":
        lo, hi = _percentile_ci(thetas, ci)
        return theta_hat, lo, hi
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(stat_fn(np.delete(x, i)))
    lo, hi = _bca_ci(thetas, theta_hat, jack, ci)
    return theta_hat, lo, hi

# ---------- FDR（BH-95） ----------
def bh_adjust(pvals: List[Optional[float]], q: float = 0.10) -> Tuple[List[float], List[bool]]:
    p = np.array([pv if (pv is not None and not np.isnan(pv)) else 1.0 for pv in pvals], dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    qvals = p * n / ranks
    qvals_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
    q_final = np.empty(n, dtype=float)
    q_final[order] = np.minimum(qvals_sorted, 1.0)
    reject = q_final <= q
    return q_final.tolist(), reject.tolist()

# ---------- Token 计数 ----------
def count_tokens(text: str, mode: str = "auto") -> int:
    if not text:
        return 0
    if mode == "auto" and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.encoding_for_model("gpt-4")
        return int(len(enc.encode(text)))
    return max(1, int(len(text) / 4))  # ~4 chars / token

# ---------- Stratification ----------
def ctx_bucket_by_len(L: int) -> str:
    if L <= 120: return "short"
    if L <= 400: return "medium"
    return "long"

def stratify_row(event_row: Dict[str, Any], region: str, az: str) -> Dict[str, str]:
    # —— 收敛为“单个布尔/字符串标量”，避免 Series 混入 —— #
    def _as_bool_scalar(x) -> bool:
        import numpy as _np, pandas as _pd
        if isinstance(x, _pd.Series):
            return bool(x.iloc[0]) if len(x) else False
        if isinstance(x, (list, tuple, _np.ndarray)):
            return bool(x[0]) if len(x) else False
        try:
            return bool(x)
        except Exception:
            return False
    def _as_str_scalar(x) -> str:
        import numpy as _np, pandas as _pd
        if isinstance(x, _pd.Series):
            return str(x.iloc[0]) if len(x) else ""
        if isinstance(x, (list, tuple, _np.ndarray)):
            return str(x[0]) if len(x) else ""
        return "" if x is None else str(x)

    prompt     = _as_str_scalar(event_row.get("prompt", ""))
    task       = _as_str_scalar(event_row.get("task", "general")) or "general"
    final_path = _as_str_scalar(event_row.get("final_path", ""))

    tool_use = _as_bool_scalar(event_row.get("tool_use", False))
    in_warm  = _as_bool_scalar(event_row.get("in_warmup", False))

    return {
        "task": task,
        "ctx_bucket": ctx_bucket_by_len(len(prompt)),
        "tool_use": str(tool_use),
        "provider_model": f"ollama/{final_path}",
        "region": region or "",
        "az": az or "",
        "cold_warm": "cold" if in_warm else "warm",
        "cache_hit_prompt": "False",
        "cache_hit_kv": "False",
    }

# ---------- Prompt 流式读取 ----------
class EmptyPromptFileError(RuntimeError):
    pass

class PromptCycler:
    def __init__(self, path: str) -> None:
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    def take(self, n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        rewound_without_prompts = False
        with open(self.path, "r", encoding="utf-8-sig") as f:
            while len(out) < n:
                line = f.readline()
                if not line:
                    if out:
                        f.seek(0); continue
                    if rewound_without_prompts:
                        raise EmptyPromptFileError(f"{self.path} yielded no usable prompts after a full pass")
                    f.seek(0); rewound_without_prompts = True; continue
                if line.startswith("#") or line.startswith("//"):
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict): out.append(obj)
                    else: out.append({"prompt": str(obj)})
                except Exception:
                    out.append({"prompt": line})
                rewound_without_prompts = False
        return out

# ---------- Ollama Client ----------
class OllamaClient:
    def __init__(self, base: str, keep_alive: str) -> None:
        self.base = base.rstrip("/")
        self.keep_alive = keep_alive
        self.session = requests.Session()  # 保留但不强依赖

    def generate(
        self,
        model: str,
        prompt: str,
        num_ctx: int,
        num_predict: int,
        timeout_s: float,
        temperature: float,
        top_p: float,
        need_json: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": int(num_ctx),
                "num_predict": int(num_predict),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        }
        if need_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            r = post_json(url, payload, timeout_s)  # keep-alive 策略由全局开关决定
            wall_ms = (time.perf_counter() - t0) * 1000.0
            r.raise_for_status()
            data = r.json()
            total_ms = (data.get("total_duration", 0) / 1e6) or wall_ms
            return {
                "text": data.get("response", "") or "",
                "e2e_ms": float(total_ms),
                "eval_tokens": int(data.get("eval_count", 0)),
                "right_censored": False,
                "error": "",
                "error_kind": "",
            }
        except requests.exceptions.Timeout as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "e2e_ms": float(wall_ms),
                "eval_tokens": 0,
                "right_censored": True,
                "error": f"timeout:{str(e)}",
                "error_kind": "timeout",
            }
        except requests.exceptions.RequestException as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "e2e_ms": float(wall_ms),
                "eval_tokens": 0,
                "right_censored": False,
                "error": f"request_error:{str(e)}",
                "error_kind": "request",
            }
        except Exception as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "e2e_ms": float(wall_ms),
                "eval_tokens": 0,
                "right_censored": False,
                "error": f"error:{str(e)}",
                "error_kind": "unknown",
            }

# ---------- Planner/QAR/执行 ----------
KEYWORDS_HEAVY = [
    "法律","合规","财务","医学","诊断","复杂算法","伪代码","正则","多步骤",
    "RAG","检索","对比","排序","解释原因","鲁棒","优化","贝叶斯","激励",
    "博弈","控制","对齐","跨模态","跨语言","对抗","量子","黑洞","暗物质",
    "隐私","差分隐私","多方计算","安全多方计算","同态加密","零知识","脑机接口",
    "聚变","可再生","MPC","需求响应","对撞机","疫苗","抗菌素","基因组",
    "侧信道","零信任","Rollup","数据可用性","软体机器人","蛋白质","生成模型",
    "负排放","碳汇","海洋","多智能体","供应链","量子互联网","法律检索",
    "序贯","数字疗法","基因编辑","精准灌溉","数字孪生","版权","水印",
]
CODE_HINTS = ["代码","regex","正则","python","js","java","sql","shell","示例代码","伪代码"]

TIERS = ["light", "std", "enh"]

def _norm_tier(tier: Optional[str]) -> str:
    t = str(tier or "light").lower()
    return t if t in TIERS else "light"

def tier_max(a: Optional[str], b: Optional[str]) -> str:
    ta = _norm_tier(a)
    tb = _norm_tier(b)
    return ta if TIERS.index(ta) >= TIERS.index(tb) else tb

SEMANTIC_TAGS = {
    "style_transfer": ["风格化","风格迁移","模仿文风","以…风格","Socratic","苏格拉底","柏拉图"],
    "refutation": ["反驳","辩难","驳斥","质疑","反证","反例","悖论","反驳…观点","驳…论"],
    "proof": ["证明","推导","归纳","演绎","公理","定理","不动点","复杂度证明","猜想","公设"],
    "debate": ["立场","观点","支持与反对","两方辩论","pros and cons"],
    "abstraction": [
        "抽象","归纳概括","范畴","理型论","本体","形而上","框架设计","体系构建","机制设计",
        "协同设计","多目标优化","策略组合","系统级权衡","治理框架","架构","蓝图","路线图",
        "鲁棒策略","层级贝叶斯","多信使数据","跨模态评测","跨语言协同","零信任架构",
    ],
}
RISK_DOMAIN_TAGS = {
    "legal":   ["法律","法务","合同","条款","合规","诉讼","判例","刑法","民法","GDPR"],
    "medical": ["医学","诊断","病理","药物","剂量","不良反应","临床","指南","ICD"],
    "finance": ["财务","会计","税","发票","估值","股权","期权","投资","风险披露"]
}

# === 新增：短而难关键词与判定（与论文“长度非充分、难度优先”一致） ===
SHORT_HARD_MAX_LEN = 160
SHORT_HARD_TAGS = [
    "P=NP", "NP 完全", "NP完全", "NP-complete", "不可判定", "停机", "停机问题",
    "哥德尔", "图灵", "莱斯", "Rice 定理", "不可计算", "计算性", "复杂度",
    "黎曼", "黎曼猜想", "ζ 函数", "ζ函数", "连续统", "连续统假设", "集合论", "公理化", "选择公理",
    "哥德尔不完备", "全息", "全息原理", "AdS/CFT", "黑洞信息", "信息悖论",
    "黑洞", "宇宙常数", "暗物质", "暗能量", "费米悖论", "热寂", "生命起源", "RNA 世界", "RNA世界",
    "宇宙有边界", "时间是离散", "因果可逆转", "自由意志", "意识", "意识可计算", "意识有量化",
    "意识上传", "真随机", "随机性", "强AI", "强 AI", "AGI", "AGI 可验证", "AGI可验证", "对齐可证明", "对齐可证明吗",
    "语言起源", "普适语法", "归纳能被证明", "自由能原理", "货币", "最优形态", "市场完全有效",
    "气候临界点", "聚变", "净增益", "人类寿命上限", "量子引力", "量子测量", "坍缩", "塌缩",
    "多世界", "宏观叠加", "全息原理普适", "时空是涌现", "数学是被发明", "道德可客观", "外星生命",
    "暗物质究竟是什么", "暗能量本质", "热寂能被避免", "量子互联网", "噬菌体", "负排放",
    "海洋铁肥", "零信任", "Rollup", "数据要素", "差分隐私", "安全多方计算", "同态加密",
    "零知识证明", "脑机接口", "火星基地", "锂电", "太空太阳能", "自动驾驶", "可达集",
    "受控升级", "多智能体博弈", "激励机制", "后量子", "侧信道", "跨模态评测",
    "序贯自适应试验", "抗菌素耐药", "数字疗法", "城市数字孪生", "版权治理",
]
SHORT_HARD_TAGS_LOWER = [
    "p=np", "np-complete", "halting", "turing", "godel", "gödel", "rice theorem",
    "riemann", "riemann hypothesis", "zeta", "continuum hypothesis", "continuum",
    "holography", "ads/cft", "quantum gravity", "quantum measurement", "wavefunction collapse",
    "many worlds", "macro superposition", "black hole information", "cosmological constant",
    "black hole", "dark matter", "dark energy", "fermi paradox", "heat death", "origin of life", "rna world",
    "universe boundary", "discrete time", "reversible causality", "free will", "computable consciousness",
    "measure consciousness", "consciousness upload", "true random", "strong ai", "agi",
    "alignment proof", "verify agi", "language origin", "universal grammar", "problem of induction",
    "free energy principle", "optimal money", "efficient market", "climate tipping point",
    "fusion net gain", "maximum human lifespan", "quantize consciousness", "emergent spacetime",
    "math invented", "moral objective", "extraterrestrial life", "panspermia", "dark matter nature",
    "dark energy nature", "avoid heat death", "quantum internet", "zero trust", "rollup",
    "data availability", "differential privacy", "secure multiparty", "homomorphic encryption",
    "zero-knowledge", "brain computer interface", "mars base", "lithium battery", "space solar",
    "autonomous driving", "reachable set", "incentive mechanism", "post-quantum", "side-channel",
    "adaptive trial", "antimicrobial resistance", "digital therapeutics", "digital twin", "copyright governance",
]

def is_short_hard(prompt: str) -> bool:
    p = prompt or ""
    if len(p) > SHORT_HARD_MAX_LEN:
        return False
    pl = p.lower()
    if any(tag in p for tag in SHORT_HARD_TAGS):
        return True
    if any(tag in pl for tag in SHORT_HARD_TAGS_LOWER):
        return True
    return False

_TEMPLATE_HINTS = ["作为一个AI","作为一名 AI","不能提供","无法提供","以下是一些建议","首先","其次","最后"]
_CONTRA_PATTERNS = [r"同时.*但是", r"然而.*但是", r"一方面.*另一方面.*但是"]

def detect_semantic(prompt: str) -> Tuple[List[str], List[str]]:
    p = (prompt or "")
    sem_hits = [k for k, kws in SEMANTIC_TAGS.items() if any(kw in p for kw in kws)]
    risk_hits= [k for k, kws in RISK_DOMAIN_TAGS.items() if any(kw in p for kw in kws)]
    return sem_hits, risk_hits

def is_templatey(text: str) -> bool:
    """只按套话关键词判断，避免误杀短纲要。"""
    t = text or ""
    return any(h in t for h in _TEMPLATE_HINTS)

def is_json_task(prompt: str) -> bool:
    p = (prompt or "")
    pl = p.lower()
    if any(k in pl for k in ["仅输出 json","只输出 json","format: json","return json"]):
        return True
    s = p.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

def looks_like_code(prompt: str) -> bool:
    p = (prompt or "").lower()
    if any(k in p for k in ["regex","python","java","sql","shell","code","snippet"]):
        return True
    return any(k in (prompt or "") for k in CODE_HINTS)

def plan(prompt: str) -> Dict[str, Any]:
    L = len(prompt or "")
    marks = sum((prompt or "").count(x) for x in ["?","？",":","：","\n"])
    heavy = any(k.lower() in (prompt or "").lower() for k in KEYWORDS_HEAVY)
    score = (L/200.0) + (marks/6.0) + (0.8 if heavy else 0.0)  # ~[0,1.6]
    return {"len": L, "marks": marks, "heavy_kw": heavy, "complexity": max(0.0, min(1.6, score))}

def probe_difficulty(client, args, prompt: str) -> Tuple[float, int, str]:
    tpl = ("你是难度评估器。对下面任务给出一个0到1之间的小数，"
           "表示完成高质量回答所需的推理复杂度。只输出数字，不要解释。\n任务：\n")
    probe_prompt = tpl + (prompt or "")
    h = hashlib.sha1(probe_prompt.encode("utf-8")).hexdigest()[:10]
    out = client.generate(LIGHT_MODEL, probe_prompt, args.num_ctx, max(16, args.probe_cap),
                          min(30, args.timeout), 0.0, 1.0, need_json=False)
    txt = (out.get("text") or "").strip()
    m = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", txt)
    val = float(m.group(0)) if m else 0.0
    val = max(0.0, min(1.0, val))
    return val, int(out.get("eval_tokens", 0)), h

def preflight_outline(out: dict) -> tuple[bool, str, int]:
    """
    预检：遇到拒绝/明显自相矛盾/几乎没内容 → 建议上调到 std。
    返回：(need_upgrade, reason, eval_tokens)
    """
    t = (out.get("text") or "")
    reason = ""
    if is_refusal(t):
        reason = "refusal"
    elif re.search("|".join(_CONTRA_PATTERNS), t) is not None:
        reason = "self_contradiction"
    elif len(t.strip()) < 8:
        reason = "too_short_outline"
    return (reason != ""), reason, int(out.get("eval_tokens", 0))

def route(score: float, thr_std: float, thr_enh: float) -> Tuple[str, str, Dict[str, float]]:
    s = float(score)
    ts = float(thr_std)
    te = float(thr_enh)
    margins = {"d_std": s - ts, "d_enh": s - te}
    if s < ts:
        return "light", f"complexity={s:.2f} < thr_std={ts:.2f}", margins
    if s < te:
        return "std", f"complexity={s:.2f} < thr_enh={te:.2f}", margins
    return "enh", f"complexity={s:.2f} >= thr_enh={te:.2f}", margins

def json_contract_ok(text: str, schema_validator: Optional[Draft7Validator]) -> Tuple[bool, str]:
    try:
        obj = json.loads((text or "").strip())
    except Exception:
        return False, "not_json"
    if schema_validator is None:
        return True, ""
    try:
        schema_validator.validate(obj)
        return True, ""
    except Exception as e:
        return False, f"schema:{str(e)[:200]}"

def needs_upgrade(prompt: str, text: str, need_json: bool, cur_path: str) -> str:
    t = (text or "").strip()
    if need_json:
        if not ((t.startswith("{") or t.startswith("[")) and (t.endswith("}") or t.endswith("]"))):
            return "expect_json_but_not_json"
        if cur_path == "light" and len(t) < MIN_JSON_LEN:
            return "too_short"
    return ""

_REFUSAL_HINTS = ["抱歉","无法帮助","不便提供","sorry","cannot help","cannot comply"]
def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _REFUSAL_HINTS)

def sleep_to_period(t_start: float, period_ms: int) -> None:
    if period_ms and period_ms > 0:
        elap = (time.perf_counter() - t_start) * 1000.0
        rest = period_ms - elap
        if rest > 0:
            time.sleep(rest / 1000.0)

class Gateway:
    def __init__(self, client: OllamaClient, args: argparse.Namespace) -> None:
        self.client = client
        self.args = args
        self.model_map = {"light": LIGHT_MODEL, "std": STD_MODEL, "enh": ENH_MODEL}
        self.req_seq = 0
        self.order = TIERS

    def controlled_execute(self, first_path: str, prompt: str, plan_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.req_seq += 1
        need_json = is_json_task(prompt)

        # 楼层兜底（before）
        floor = _norm_tier(self.args.def_floor)
        floor_sources: List[str] = []
        floor_seen: Set[str] = set()

        def note(reason: str) -> None:
            r = str(reason or "").strip()
            if r and r not in floor_seen:
                floor_sources.append(r)
                floor_seen.add(r)

        if plan_ctx:
            preset = plan_ctx.get("route_floor_reco")
            if preset:
                floor = _norm_tier(preset)
            preset_reason = plan_ctx.get("route_floor_reason")
            if preset_reason:
                for r in str(preset_reason).split("|"):
                    note(r)
            try:
                if plan_ctx.get("short_hard"):
                    nf = tier_max(floor, "std")
                    if nf != floor:
                        floor = nf; note("short_hard")
            except Exception:
                pass
            try:
                if len(plan_ctx.get("semantic_heavy", [])) > 0:
                    nf = tier_max(floor, "std")
                    if nf != floor:
                        floor = nf; note("semantic_heavy")
            except Exception:
                pass
            try:
                if len(plan_ctx.get("risk_domain", [])) > 0:
                    nf = tier_max(floor, "std")
                    if nf != floor:
                        floor = nf; note("risk_domain")
            except Exception:
                pass

        # 新增：短而难至少 std（兜底）
        try:
            if (not need_json) and is_short_hard(prompt):
                nf = tier_max(floor, "std")
                if nf != floor:
                    floor = nf; note("short_hard")
        except Exception:
            pass

        floor_before = floor

        if need_json:
            nf = tier_max(floor, self.args.json_floor)
            if nf != floor:
                floor = nf; note("json_contract")
        elif looks_like_code(prompt):
            nf = tier_max(floor, self.args.code_floor)
            if nf != floor:
                floor = nf; note("code_task")

        floor_after = floor

        # —— 轻量探索采样：只改 first_path，不改楼层！——
        if (not need_json) and (not looks_like_code(prompt)):
            k = int(getattr(self.args, "explore_light_every_k", 0) or 0)
            if k > 0 and (self.req_seq % k == 0):
                first_path = "light"

        # preflight：体检小样触发则把楼层升到 std
        preflight_used = False
        preflight_reason = ""
        preflight_tokens = 0
        if (self.args.preflight == "auto") and plan_ctx:
            try:
                short_hard = bool(plan_ctx.get("short_hard", False))
                trig = short_hard or (len(plan_ctx.get("semantic_heavy", [])) > 0) or (
                    float(plan_ctx.get("probe_difficulty",0.0)) >= float(self.args.probe_threshold)
                )
            except Exception:
                trig = False
            if trig:
                preflight_used = True
                pout = self.client.generate(
                    self.model_map["light"], prompt,
                    self.args.num_ctx,
                    max(8, min(self.args.preflight_cap, 48)),
                    min(30, self.args.timeout),
                    0.0, 1.0, need_json=False
                )
                do_up, why, pftok = preflight_outline(pout)
                preflight_reason, preflight_tokens = (why or ""), int(pftok)
                if do_up and self.order.index(floor) < self.order.index("std"):
                    floor = "std"
                    note("preflight")

        # 与首选路径比较（after）
        cur = first_path
        if self.order.index(_norm_tier(cur)) < self.order.index(floor):
            cur = floor

        # JSON 任务严格提示与 cap
        cap = max(self.args.cap_enh, 192) if need_json else self.args.cap_enh
        prompt_eff = ("只输出 JSON，不要解释与多余文本。\n" + (prompt or "")) if need_json else (prompt or "")

        attempts = 0
        upgrades = 0
        repaired = False
        last_out: Dict[str, Any] = {"text":"", "e2e_ms":np.nan, "eval_tokens":0,
                                    "right_censored":False, "error":"", "error_kind":""}
        reason = ""

        while attempts < (self.args.max_upgrades + 1):
            last_out = self.client.generate(
                self.model_map[cur], prompt_eff, self.args.num_ctx, cap,
                self.args.timeout, self.args.temperature, self.args.top_p, need_json=need_json
            )
            attempts += 1
            if last_out.get("right_censored", False):
                break
            reason = needs_upgrade(prompt_eff, last_out.get("text",""), need_json, cur)
            if (not reason) or cur == "enh":
                break
            i = self.order.index(cur)
            if i < (len(self.order) - 1) and upgrades < self.args.max_upgrades:
                upgrades += 1
                cur = self.order[i+1]
            else:
                break

        return {
            "final_path": cur,
            "attempts": attempts,
            "upgrades": upgrades,
            "repaired": repaired,
            "route_floor_before": floor_before,
            "route_floor_after": floor_after,
            "route_floor_reason": "|".join(floor_sources),
            "preflight_used": preflight_used,
            "preflight_reason": preflight_reason,
            "preflight_tokens": preflight_tokens,
            "upgrade_reason": reason,
            **last_out
        }

# ---------- 质量打分（粗评） ----------
GOLD_RULES: Dict[int, Dict[str, Any]] = {}


def quality_score(event_row: Dict[str, Any], mode: str = "auto") -> float:
    if mode == "none":
        return np.nan
    obj = event_row.get("_prompt_obj")
    if not isinstance(obj, dict):
        return np.nan
    out_raw = (event_row.get("output") or "").strip()
    if not out_raw:
        return 0.0
    out_lower = out_raw.lower()
    gold = obj.get("label") or obj.get("answer") or None
    answers = obj.get("answers")
    if gold:
        g = str(gold).strip().lower()
        if mode in ("auto","exact"):
            return 1.0 if out_lower == g else 0.0
        if mode == "substring":
            return 1.0 if g in out_lower else 0.0
    if answers and isinstance(answers, list) and answers:
        al = [str(a).strip().lower() for a in answers]
        if mode in ("auto","exact"):
            return 1.0 if out_lower in al else 0.0
        if mode == "substring":
            return 1.0 if any(a in out_lower for a in al) else 0.0

    prompt_id = obj.get("id") or event_row.get("prompt_id")
    try:
        prompt_id_int = int(prompt_id)
    except Exception:
        prompt_id_int = 0
    if prompt_id_int and prompt_id_int in GOLD_RULES:
        score = _judge_by_rule(out_raw, GOLD_RULES[prompt_id_int])
        if not math.isnan(score):
            return float(score)

    return 1.0 if out_raw else 0.0

# ---------- 统计聚合 ----------
@dataclass
class StatsResult:
    summary: Dict[str, Any]
    warm_df: pd.DataFrame
    df_all: pd.DataFrame

def compute_stats_and_summary(rows: List[Dict[str, Any]], args: argparse.Namespace,
                              tdigest_version: str = TDIGEST_VERSION) -> StatsResult:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no rows collected")
    warm_df = df[~df["in_warmup"]].copy()

    # 原始 P50/P95
    overall_lat = df["e2e_ms"].tolist()
    warm_lat = warm_df["e2e_ms"].tolist()
    overall_p50 = q50(overall_lat); overall_p95 = q95(overall_lat)
    warm_p50 = q50(warm_lat); warm_p95 = q95(warm_lat)

    # CO-corrected
    if args.co_correction == "periodic_fill":
        overall_lat_co = co_correct_periodic_fill(overall_lat, args.period_ms, args.max_co_fills)
        warm_lat_co = co_correct_periodic_fill(warm_lat, args.period_ms, args.max_co_fills)
    else:
        overall_lat_co = overall_lat; warm_lat_co = warm_lat
    overall_p95_co = tdigest_quantile(overall_lat_co, 0.95, compression=args.tdigest_compression)
    warm_p95_co = tdigest_quantile(warm_lat_co, 0.95, compression=args.tdigest_compression)

    # 删失：KM 与 SLO-cap（warm-only）
    warm_obs = ~(warm_df["right_censored"].astype(bool))
    warm_lat_list = warm_df["e2e_ms"].tolist()
    warm_p95_km = km_quantile(warm_lat_list, warm_obs.tolist(), q=0.95)
    timeout_cap_ms = args.timeout_cap_ms if args.timeout_cap_ms else (args.timeout * 1000)
    warm_lat_cap = [(timeout_cap_ms if (not ob) else t) for t, ob in zip(warm_lat_list, warm_obs.tolist())]
    warm_p95_cap = q95(warm_lat_cap)
    timeouts_rate = float((~warm_obs).mean() * 100.0)
    censoring_mode = "KM / SLO-cap"

    # 路由分布
    dist = warm_df["final_path"].value_counts(normalize=True) * 100.0
    light_pct = float(dist.get("light", 0.0))
    std_pct = float(dist.get("std", 0.0))
    enh_pct = float(dist.get("enh", 0.0))
    r_costly_pct = float((warm_df["final_path"] == "enh").mean() * 100.0)

    # JSON 失败率
    need_json_mask = warm_df["need_json"] == True
    if need_json_mask.any() and "json_ok" in warm_df:
        json_fail_rate = float((~warm_df.loc[need_json_mask, "json_ok"]).mean() * 100.0)
    else:
        json_fail_rate = 0.0

    # ITT 衍生率
    retry_rate_pct = float((warm_df.get("retry_count", pd.Series([0]*len(warm_df))).astype(int) > 0).mean() * 100.0)
    refusal_rate_pct = float((warm_df.get("refusal", False) == True).mean() * 100.0)

    # 分层标签与权重
    strat_rows = []
    for _, r in warm_df.iterrows():
        s = stratify_row(r, args.region, args.az)
        s["stratum"] = "|".join([s["task"], s["ctx_bucket"], s["tool_use"], s["provider_model"], s["region"],
                                 s["az"], s["cold_warm"], s["cache_hit_prompt"], s["cache_hit_kv"]])
        strat_rows.append(s)
    strat_df = pd.DataFrame(strat_rows)
    warm_df = pd.concat([warm_df.reset_index(drop=True), strat_df.reset_index(drop=True)], axis=1)
    # —— 去重列名，避免下游再出现重复字段 —— #
    warm_df = warm_df.loc[:, ~warm_df.columns.duplicated(keep="last")]
    warm_df["weight"] = 1.0
    if getattr(args, "poststrat_weights", None):
        try:
            wdf = pd.read_csv(args.poststrat_weights)
            weight_map = {str(r["stratum"]): float(r["weight"]) for _, r in wdf.iterrows()}
            warm_df["weight"] = warm_df["stratum"].map(weight_map).fillna(1.0)
        except Exception:
            pass

    # 质量分（可选）
    warm_df["q_primary"] = warm_df.apply(lambda r: quality_score(dict(r), mode=args.judge_mode), axis=1)

    summary = {
        "mode": args.mode,
        "platform": platform.platform(),
        "light_model": LIGHT_MODEL, "std_model": STD_MODEL, "enh_model": ENH_MODEL,
        "N_total": int(len(df)), "N_eval": int(len(warm_df)),
        "num_ctx": int(args.num_ctx), "cap_enh": int(args.cap_enh),
        "temperature": float(args.temperature), "top_p": float(args.top_p),

        "overall_p50_ms": float(overall_p50), "overall_p95_ms": float(overall_p95),
        "overall_p95_ms_co": float(overall_p95_co),
        "warm_p50_ms": float(warm_p50), "warm_p95_ms": float(warm_p95),
        "warm_p95_ms_co": float(warm_p95_co),

        "warm_p95_ms_km": float(warm_p95_km),
        "warm_p95_ms_cap": float(warm_p95_cap),
        "timeouts_rate_pct": float(timeouts_rate),
        "censoring_mode": censoring_mode,

        "avg_eval_ms": float(warm_df["e2e_ms"].mean()),
        "sum_eval_ms": float(warm_df["e2e_ms"].sum()),
        "avg_work_ms_approx": float((warm_df["e2e_ms"] * warm_df["attempts"]).mean()),
        "sum_work_ms_approx": float((warm_df["e2e_ms"] * warm_df["attempts"]).sum()),
        "attempts_mean": float(warm_df["attempts"].mean()),
        "attempts_p95": float(np.percentile(warm_df["attempts"], 95)),

        "light_pct": float(light_pct),
        "std_pct": float(std_pct),
        "enh_pct": float(enh_pct),
        "r_costly_pct": float(r_costly_pct),
        "path_pct_sum": float(light_pct + std_pct + enh_pct),

        "json_fail_rate_pct": float(json_fail_rate),
        "retry_rate_pct": float(retry_rate_pct),
        "refusal_rate_pct": float(refusal_rate_pct),

        "sla": args.sla,
        "period_ms": int(args.period_ms),
        "sampler_period_ms": int(args.period_ms),
        "co_correction": str(args.co_correction),
        "tdigest_compression": int(args.tdigest_compression),
        "tdigest_version": tdigest_version or "unknown",
        "warmup_window_ms": int(args.warmup_window_ms),
        "periodic_load": True,
        "hdr_sigfig": None,
        "seed": args.seed if args.seed is not None else None,
        "engine": "ollama",
        "quant_config": None,
        "model_build": None, "router_build": None, "planner_build": None,

        "thr_std": float(args.thr_std), "thr_enh": float(args.thr_enh),
        "json_floor": str(args.json_floor), "code_floor": str(args.code_floor), "def_floor": str(args.def_floor),
        "max_upgrades": int(args.max_upgrades),
        "force_light_len": int(args.force_light_len), "force_light_score": float(args.force_light_score),
    }

    return StatsResult(summary=summary, warm_df=warm_df, df_all=df)

# ---------- 家族检验 ----------
def family_C_json_guard(warm_df: pd.DataFrame, q: float) -> List[Dict[str, Any]]:
    g = warm_df.copy()
    g["json_need_and_fail"] = (g["need_json"] == True) & (g["json_ok"] == False)
    tests: List[Dict[str, Any]] = []
    for s, sdf in g.groupby("stratum"):
        n_need = int((sdf["need_json"] == True).sum())
        x_fail = int((sdf["json_need_and_fail"]).sum())
        if n_need == 0:
            pval = None
        else:
            p0 = DEFAULT_JSON_FAIL_MAX
            phat = x_fail / max(1, n_need)
            sd = math.sqrt(p0 * (1 - p0) / max(1, n_need))
            z = (phat - p0) / (sd + EPS)
            from math import erf
            pval = 0.5 * (1 + erf(z / math.sqrt(2)))  # 单侧
        tests.append({"family":"C","stratum":s,"n_need":n_need,"x_fail":x_fail,"raw_p":pval})
    qvals, reject = bh_adjust([t["raw_p"] for t in tests], q=q)
    for t, qv, rj in zip(tests, qvals, reject):
        t["q_value"] = qv; t["reject"] = bool(rj)
    return tests

def _bootstrap_guardrail_p95(g_lat: List[float], b_lat: List[float], delta: float,
                             B: int, compression: int, seed: Optional[int]) -> float:
    if len(g_lat) == 0 or len(b_lat) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    cnt = 0; ng, nb = len(g_lat), len(b_lat)
    for _ in range(B):
        gi = rng.integers(0, ng, ng); bi = rng.integers(0, nb, nb)
        g95_b = tdigest_quantile([g_lat[i] for i in gi], 0.95, compression=compression)
        b95_b = tdigest_quantile([b_lat[i] for i in bi], 0.95, compression=compression)
        if g95_b <= (1.0 + delta) * b95_b:
            cnt += 1
    pval = 1.0 - (cnt / max(1, B))
    return float(pval)

def _family_B_run_one(item: Tuple[int, str, List[float], List[float]],
                      p95_delta: float, bootstrap_B: int, tdigest_compression: int,
                      seed: Optional[int]) -> Tuple[int, str, float, float, float]:
    idx, s, g_lat, b_lat = item
    g95s = tdigest_quantile(g_lat, 0.95, compression=tdigest_compression)
    b95s = tdigest_quantile(b_lat, 0.95, compression=tdigest_compression)
    pval = _bootstrap_guardrail_p95(g_lat, b_lat, p95_delta, bootstrap_B, tdigest_compression, seed)
    return idx, s, g95s, b95s, pval

def family_B_p95_guardrail(warm_df: pd.DataFrame,
                           base_warm_df: Optional[pd.DataFrame],
                           args: argparse.Namespace) -> List[Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    if base_warm_df is None or base_warm_df.empty:
        return tests

    base_warm = base_warm_df.copy()
    # —— baseline 去重列名，防止重名索引导致 Series —— #
    base_warm = base_warm.loc[:, ~base_warm.columns.duplicated(keep="last")].copy()

    # 用 stratify_row 统一产出分层字段，避免把 Series 拼进 join
    def _stratum_from_row(r: pd.Series) -> str:
        s = stratify_row(r, args.region, args.az)
        return "|".join([
            s["task"], s["ctx_bucket"], s["tool_use"], s["provider_model"],
            s["region"], s["az"], s["cold_warm"], s["cache_hit_prompt"], s["cache_hit_kv"]
        ])

    base_warm["stratum"] = base_warm.apply(_stratum_from_row, axis=1)

    # 每个层级的 guardrail 工作条目
    work_items: List[Tuple[int, str, List[float], List[float]]] = []
    for idx, (sname, sdf) in enumerate(warm_df.groupby("stratum")):
        g_lat = sdf["e2e_ms"].tolist()
        b_lat = base_warm.loc[base_warm["stratum"] == sname, "e2e_ms"].tolist()
        if args.co_correction == "periodic_fill":
            g_lat = co_correct_periodic_fill(g_lat, args.period_ms, args.max_co_fills)
            b_lat = co_correct_periodic_fill(b_lat, args.period_ms, args.max_co_fills)
        work_items.append((idx, sname, g_lat, b_lat))

    # 计算 p95 与 p 值
    results: List[Tuple[int, str, float, float, float]] = []
    if args.workers and args.workers > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(_family_B_run_one, it, args.p95_delta,
                              args.bootstrap_B, args.tdigest_compression, args.seed)
                    for it in work_items]
            for fu in as_completed(futs):
                results.append(fu.result())
    else:
        for it in work_items:
            results.append(_family_B_run_one(it, args.p95_delta,
                                             args.bootstrap_B, args.tdigest_compression, args.seed))
    results.sort(key=lambda x: x[0])

    tests = [{"family": "B", "stratum": sname, "p95_guard_delta": args.p95_delta,
              "g95": g95s, "b95": b95s, "raw_p": pval}
             for _, sname, g95s, b95s, pval in results]

    # FDR 调整
    qvals, reject = bh_adjust([t["raw_p"] for t in tests], q=args.fdr_q)
    for t, qv, rj in zip(tests, qvals, reject):
        t["q_value"] = qv
        t["reject"] = bool(rj)
    return tests

# -------- 家族A：TOST 非劣（基于 q_primary，需存在 baseline 与 gold） --------
def family_A_tost(
    warm_df: pd.DataFrame,
    base_warm_df: Optional[pd.DataFrame],
    epsilon_q: float,
    B: int = 2000,
    seed: Optional[int] = None,
    require_family: bool = True,
    min_n: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    meta = {"status": "skipped", "reason": "", "eligible_strata": 0, "tested_strata": 0}
    if base_warm_df is None or base_warm_df.empty:
        meta.update({"reason": "baseline missing"})
        return tests, meta

    if "stratum" not in warm_df:
        raise ValueError("warm_df 缺少 stratum 列，无法执行家族A")

    # 只在存在质量分时做
    if not warm_df["q_primary"].notna().any():
        for s, _ in warm_df.groupby("stratum"):
            tests.append({
                "family": "A",
                "stratum": s,
                "epsilon_q": epsilon_q,
                "raw_p": None,
                "note": "no quality labels",
            })
        meta.update({
            "status": ("blocked" if require_family else "inconclusive"),
            "reason": "no quality labels",
            "eligible_strata": int(warm_df.groupby("stratum").ngroups),
            "tested_strata": 0,
        })
        return tests, meta
    rng = np.random.default_rng(seed)
    tested = 0
    eligible = 0
    for s, sdf in warm_df.groupby("stratum"):
        g = sdf["q_primary"].dropna().to_numpy(dtype=float)
        b = base_warm_df[base_warm_df["stratum"] == s]["q_primary"].dropna().to_numpy(dtype=float)
        if g.size == 0 or b.size == 0:
            tests.append({
                "family": "A",
                "stratum": s,
                "epsilon_q": epsilon_q,
                "raw_p": None,
                "note": "empty stratum",
                "n_gateway": int(g.size),
                "n_baseline": int(b.size),
            })
            continue
        eligible += 1
        if min_n and min(g.size, b.size) < int(min_n):
            tests.append({
                "family": "A",
                "stratum": s,
                "epsilon_q": epsilon_q,
                "raw_p": None,
                "note": f"too few labels (g={g.size}, b={b.size})",
                "n_gateway": int(g.size),
                "n_baseline": int(b.size),
            })
            continue
        # 差值 d = mean(g) - mean(b)
        d_hat = float(np.mean(g) - np.mean(b))
        cnt_lo = 0
        cnt_hi = 0
        ng, nb = g.size, b.size
        for _ in range(B):
            gi = rng.integers(0, ng, ng)
            bi = rng.integers(0, nb, nb)
            d_b = float(np.mean(g[gi]) - np.mean(b[bi]))
            if d_b <= -epsilon_q:
                cnt_lo += 1
            if d_b >= epsilon_q:
                cnt_hi += 1
        p_lo = cnt_lo / max(1, B)
        p_hi = cnt_hi / max(1, B)
        raw_p = max(p_lo, p_hi)
        tests.append({
            "family": "A",
            "stratum": s,
            "epsilon_q": float(epsilon_q),
            "mean_g": float(np.mean(g)),
            "mean_b": float(np.mean(b)),
            "diff_g_minus_b": float(d_hat),
            "raw_p": float(raw_p),
            "n_gateway": int(g.size),
            "n_baseline": int(b.size),
        })

    if tests:
        qvals, reject = bh_adjust([t["raw_p"] for t in tests], q=0.10)
        for t, qv, rj in zip(tests, qvals, reject):
            t["q_value"] = qv
            t["reject"] = bool(rj)
            t["pass_equiv"] = bool(rj)

    meta.update({
        "status": "ok" if tested > 0 else ("blocked" if require_family else "inconclusive"),
        "reason": "" if tested > 0 else ("no eligible strata" if eligible == 0 else "min_n filter"),
        "eligible_strata": int(eligible),
        "tested_strata": int(tested),
    })
    return tests, meta

# ---------- 落盘 ----------
def latest(path_glob: str) -> Optional[str]:
    cand = sorted(glob.glob(path_glob), key=os.path.getmtime, reverse=True)
    return cand[0] if cand else None

def write_outputs(
    df_all: pd.DataFrame, summary: Dict[str, Any],
    tests_A: List[Dict[str, Any]], tests_B: List[Dict[str, Any]], tests_C: List[Dict[str, Any]],
    args: argparse.Namespace
) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.mode}_{ts}"
    os.makedirs("runs", exist_ok=True)

    # 事件明细
    events_path = os.path.join("runs", f"{base}_events.jsonl")
    df_all.to_json(events_path, orient="records", lines=True, force_ascii=False)

    # 汇总 JSON
    summary_path = os.path.join("runs", f"{base}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    # metrics_summary
    metrics_df = pd.DataFrame([summary])
    try:
        metrics_df.to_parquet(os.path.join("runs", f"{base}_metrics_summary.parquet"), index=False)
    except Exception:
        metrics_df.to_csv(os.path.join("runs", f"{base}_metrics_summary.csv"), index=False)

    # 家族检验 CSV
    if tests_A:
        pd.DataFrame(tests_A).to_csv(os.path.join("runs", f"{base}_tests_family_A.csv"), index=False)
    if tests_B:
        pd.DataFrame(tests_B).to_csv(os.path.join("runs", f"{base}_tests_family_B.csv"), index=False)
    if tests_C:
        pd.DataFrame(tests_C).to_csv(os.path.join("runs", f"{base}_tests_family_C.csv"), index=False)

    # bootstrap_meta.json & seed_manifest
    meta = {
        "B": int(args.bootstrap_B),
        "ci_method": args.bootstrap_ci,
        "seed": int(args.seed) if args.seed is not None else None,
        "resample_unit": "request",
        "block_len": None,
        "censoring_mode": summary.get("censoring_mode", ""),
        "co_params": {"method": args.co_correction, "period_ms": args.period_ms, "max_fills": args.max_co_fills},
        "estimator": {"offline": "HD+Bootstrap", "online": "t-digest",
                      "tdigest_compression": args.tdigest_compression, "tdigest_version": TDIGEST_VERSION},
        "FDR": {"method": "BH-95", "q": args.fdr_q},
        "stratification": ["task","ctx_bucket","tool_use","provider/model","region/az","cold_warm","cache_hit"],
        "weights_file": args.poststrat_weights
    }
    with open(os.path.join("runs", f"{base}_bootstrap_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join("runs", f"{base}_seed_manifest.txt"), "w", encoding="utf-8") as f:
        f.write(str(args.seed if args.seed is not None else "None") + "\n")

    return base

# ---------- CLI 工具 ----------
def _parse_bool_flag(text) -> bool:
    if isinstance(text, bool):
        return bool(text)
    t = str(text).strip().lower()
    if t in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {text}")

# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline_enh","gateway_3stage"], required=True)

    # 运行规模
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prompts", default="prompts.jsonl")

    # 推理开关
    ap.add_argument("--cap_enh", type=int, default=192)
    ap.add_argument("--num_ctx", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP读取超时秒；连接超时固定5秒")
    ap.add_argument("--sla", choices=["complete","json","none"], default="complete")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--period-ms", dest="period_ms", type=int, default=0)
    ap.add_argument("--http-keepalive", choices=["auto","on","off"], default="auto",
                    help="HTTP 连接复用：auto=Windows关/其它开；on=总开；off=总关")

    # 路由阈值 + 楼层 + 升级次数
    ap.add_argument("--thr_std", type=float, default=0.40)
    ap.add_argument("--thr_enh", type=float, default=0.85)
    ap.add_argument("--json_floor", default="std")
    ap.add_argument("--code_floor", default="std")
    ap.add_argument("--def_floor", default="light")
    ap.add_argument("--max_upgrades", type=int, default=2)

    # 轻臂“快车道”
    ap.add_argument("--force_light_len", type=int, default=120)
    ap.add_argument("--force_light_score", type=float, default=0.30)

    # 保底探索采样（首跳 light）
    ap.add_argument("--explore-light-every-k", type=int, default=0, dest="explore_light_every_k",
                    help="每 K 个请求先试 light（0=关闭；preflight/probe 仍可上调到 std/enh）")

    # 探针与 bonus
    ap.add_argument("--disable-probe", action="store_true")
    ap.add_argument("--probe-cap", type=int, default=24)
    ap.add_argument("--probe-weight", type=float, default=0.40)
    ap.add_argument("--semantic-bonus", type=float, default=0.25)
    ap.add_argument("--risk-bonus", type=float, default=0.20)
    ap.add_argument("--probe-threshold", type=float, default=0.45)

    # preflight
    ap.add_argument("--preflight", choices=["off","auto"], default="auto")
    ap.add_argument("--preflight-cap", type=int, default=48)

    # 度量/指纹
    ap.add_argument("--co-correction", choices=["none","periodic_fill"], default="periodic_fill")
    ap.add_argument("--tdigest-compression", type=int, default=TDIGEST_DEFAULT_COMPRESSION, dest="tdigest_compression")
    ap.add_argument("--warmup-window-ms", type=int, default=None, dest="warmup_window_ms")
    ap.add_argument("--json-schema", default=None, dest="json_schema")
    ap.add_argument("--max-co-fills", type=int, default=120, dest="max_co_fills")
    ap.add_argument("--gold", default="gold.jsonl", help="可选：质量 gold 规则(jsonl)")

    # 统计协议 & FDR
    ap.add_argument("--bootstrap-B", type=int, default=1000, dest="bootstrap_B")
    ap.add_argument("--bootstrap-ci", choices=["bca","percentile"], default="bca")
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--epsilon-q", type=float, default=0.01)
    ap.add_argument("--p95-delta", type=float, default=DEFAULT_P95_DELTA)
    ap.add_argument("--judge-mode", choices=["auto","exact","substring","none"], default="auto")
    ap.add_argument("--poststrat-weights", default=None)
    ap.add_argument("--min-n-per-stratum", type=int, default=0,
                    help="质量 TOST 每个分层的最小有效样本数（0=关闭）")
    ap.add_argument("--require-family-A", type=_parse_bool_flag, default=True,
                    help="质量门：True=缺就阻塞；False=允许标记 inconclusive")
    ap.add_argument("--workers", type=int, default=0, help="家族B Bootstrap 并行 worker 数")

    # 审计/环境
    ap.add_argument("--region", default="local")
    ap.add_argument("--az", default="local-az")
    ap.add_argument("--user-tenant", default="bench")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tokenizer", choices=["auto","approx"], default="auto")

    # 超时封顶（SLO-cap 口径）
    ap.add_argument("--timeout-cap-ms", type=int, default=None)
    return ap

# ---------- 主流程 ----------
def main() -> None:
    args = build_arg_parser().parse_args()
    set_http_keepalive(args.http_keepalive)

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    random.seed(args.seed); np.random.seed(args.seed)

    global GOLD_RULES
    GOLD_RULES = load_gold_rules(args.gold)

    # warmup 窗口
    if args.warmup_window_ms is None:
        args.warmup_window_ms = int(max(0, args.warmup) * max(0, args.period_ms))

    os.makedirs("runs", exist_ok=True)

    # prompts
    empty_prompt_exc: Optional[EmptyPromptFileError] = None
    try:
        prompts = PromptCycler(args.prompts).take(args.n)
    except EmptyPromptFileError as exc:
        empty_prompt_exc = exc; prompts = []
    if not prompts:
        if empty_prompt_exc is not None:
            raise RuntimeError("prompts.jsonl 为空") from empty_prompt_exc

    # JSON Schema（可选）
    schema_validator = None
    if args.json_schema and Draft7Validator is not None:
        try:
            with open(args.json_schema, "r", encoding="utf-8") as f:
                schema_validator = Draft7Validator(json.load(f))
        except Exception:
            schema_validator = None

    client = OllamaClient(OLLAMA_BASE, OLLAMA_KEEPALIVE)
    gw = Gateway(client, args)

    rows: List[Dict[str, Any]] = []
    session_id = str(uuid.uuid4())
    t_session0 = time.perf_counter()

    for idx, obj in enumerate(prompts, 1):
        if isinstance(obj, dict) and "id" not in obj:
            obj["id"] = idx
        t_period = time.perf_counter()
        prompt = obj.get("prompt") if isinstance(obj, dict) else str(obj)
        prompt = prompt or ""
        need_json = is_json_task(prompt)
        in_warmup = ((time.perf_counter() - t_session0) * 1000.0 <= args.warmup_window_ms) if args.warmup_window_ms else (idx <= args.warmup)

        # —— 规划 / QAR —— baseline_enh 走 enh；gateway_3stage 走路由
        short_hard_flag = False

        if args.mode == "baseline_enh":
            chosen = "enh"; route_decision = "always_enh"
            plan_report = {"complexity": None, "complexity_legacy": None}
            sem_heavy, risk_domain = [], []
            probe_score, probe_tokens, probe_hash = 0.0, 0, ""
            out = client.generate(ENH_MODEL, prompt, args.num_ctx, args.cap_enh,
                                  args.timeout, args.temperature, args.top_p, need_json=need_json)
            final_info = {
                "final_path": "enh", "attempts": 1, "upgrades": 0, "repaired": False,
                "route_floor_before": "enh", "route_floor_after": "enh", "route_floor_reason": "baseline",
                "preflight_used": False, "preflight_reason": "", "preflight_tokens": 0,
                "upgrade_reason": "", **out
            }
        else:
            try:
                short_hard_flag = bool(is_short_hard(prompt))
            except Exception:
                short_hard_flag = False
            plan_report = plan(prompt)
            legacy_score = float(plan_report.get("complexity", 0.0))
            sem_heavy, risk_domain = detect_semantic(prompt)
            heavy_hit = (len(sem_heavy) > 0)
            risk_hit  = (len(risk_domain) > 0)

            route_floor = _norm_tier(args.def_floor)
            floor_reasons: List[str] = []

            def lift_floor(tier: Optional[str], reason: str) -> None:
                nonlocal route_floor
                new_floor = tier_max(route_floor, tier)
                if new_floor != route_floor:
                    route_floor = new_floor
                    if reason not in floor_reasons:
                        floor_reasons.append(reason)

            if need_json:
                lift_floor(args.json_floor, "json_contract")
            elif looks_like_code(prompt):
                lift_floor(args.code_floor, "code_task")
            if heavy_hit:
                lift_floor("std", "semantic_heavy")
            if risk_hit:
                lift_floor("std", "risk_domain")
            if short_hard_flag:
                lift_floor("std", "short_hard")

            probe_score, probe_tokens, probe_hash = (0.0, 0, "")
            if not args.disable_probe:
                try:
                    probe_score, probe_tokens, probe_hash = probe_difficulty(client, args, prompt)
                except Exception:
                    probe_score = 0.0

            score_prime = legacy_score \
                          + args.probe_weight * float(probe_score) \
                          + (args.semantic_bonus if heavy_hit else 0.0) \
                          + (args.risk_bonus if risk_hit else 0.0)

            chosen, route_decision, route_margins = route(score_prime, args.thr_std, args.thr_enh)
            plan_report["complexity_legacy"] = legacy_score
            plan_report["complexity"] = score_prime
            plan_report["router_counterfactual"] = chosen
            plan_report["router_margin_std"] = float(route_margins.get("d_std", np.nan))
            plan_report["router_margin_enh"] = float(route_margins.get("d_enh", np.nan))
            plan_report["route_floor_reco"] = route_floor
            plan_report["route_floor_reason"] = "|".join(floor_reasons)
            plan_report["short_hard"] = bool(short_hard_flag)

            # “快车道”：短且易 → light（非 JSON/非代码）——短而难禁用快车道
            if (not need_json) and (not looks_like_code(prompt)) and (not short_hard_flag):
                if args.force_light_len > 0 and len(prompt) <= args.force_light_len:
                    if (plan_report.get("complexity") or 0.0) <= args.force_light_score:
                        chosen = "light"
                        route_decision += " | fastlane"

            final_info = gw.controlled_execute(chosen, prompt, plan_ctx={
                "semantic_heavy": sem_heavy, "risk_domain": risk_domain,
                "probe_difficulty": probe_score, "short_hard": short_hard_flag,
                "route_floor_reco": route_floor, "route_floor_reason": "|".join(floor_reasons),
            })

        text = final_info.get("text", "")
        ok_json, json_err = (True,"")
        if need_json:
            ok_json, json_err = json_contract_ok(text, schema_validator)

        row = {
            "session_id": session_id,
            "idx": idx,
            "prompt_id": obj.get("id") if isinstance(obj, dict) else idx,
            "prompt": prompt,
            "_prompt_obj": obj,
            "need_json": bool(need_json),
            "json_ok": bool(ok_json),
            "json_err": json_err,
            "output": text,
            "e2e_ms": float(final_info.get("e2e_ms", np.nan)),
            "eval_tokens": int(final_info.get("eval_tokens", 0)),
            "right_censored": bool(final_info.get("right_censored", False)),
            "error": final_info.get("error", ""),
            "error_kind": final_info.get("error_kind", ""),
            "final_path": final_info.get("final_path", ""),
            "attempts": int(final_info.get("attempts", 1)),
            "upgrades": int(final_info.get("upgrades", 0)),
            "repaired": bool(final_info.get("repaired", False)),
            "route_floor_before": final_info.get("route_floor_before", ""),
            "route_floor_after": final_info.get("route_floor_after", ""),
            "preflight_used": bool(final_info.get("preflight_used", False)),
            "preflight_reason": final_info.get("preflight_reason", ""),
            "preflight_tokens": int(final_info.get("preflight_tokens", 0)),
            "upgrade_reason": final_info.get("upgrade_reason", ""),
            "planner_complexity": plan_report.get("complexity", None),
            "planner_complexity_legacy": plan_report.get("complexity_legacy", None),
            "semantic_heavy": "|".join(sem_heavy) if sem_heavy else "",
            "risk_domain": "|".join(risk_domain) if risk_domain else "",
            "probe_score": float(probe_score),
            "probe_tokens": int(probe_tokens),
            "probe_hash": probe_hash,
            "route_decision": route_decision,
            "router_reason": route_decision,
            "router_counterfactual": plan_report.get("router_counterfactual", ""),
            "router_margin_std": float(plan_report.get("router_margin_std", np.nan)),
            "router_margin_enh": float(plan_report.get("router_margin_enh", np.nan)),
            "route_floor_reco": plan_report.get("route_floor_reco", ""),
            "route_floor_reason": final_info.get("route_floor_reason", ""),
            "short_hard": bool(short_hard_flag) if args.mode != "baseline_enh" else False,
            "tool_use": False,
            "task": "general",
            "in_warmup": bool(in_warmup),
            "retry_count": 0,
            "refusal": bool(is_refusal(text)),
            "token_est_prompt": int(count_tokens(prompt, args.tokenizer)),
            "token_est_output": int(count_tokens(text, args.tokenizer)),
            "billing_version": BILLING_VERSION
        }
        rows.append(row)

        # 周期采样：补足载荷节律
        if idx < len(prompts):
            sleep_to_period(t_period, args.period_ms)
    # ---- 统计与输出 ----
    stats = compute_stats_and_summary(rows, args)
    # baseline（若存在）——用于家族A/B
    base_events_path = latest(os.path.join("runs", "baseline_enh_*_events.jsonl"))
    base_warm_df = None
    if base_events_path:
        try:
            base_df_all = pd.read_json(base_events_path, lines=True)
            base_warm_df = base_df_all[~base_df_all["in_warmup"]].copy()
            # 保障字段：q_primary 与 stratum
            base_warm_df["q_primary"] = base_warm_df.apply(lambda r: quality_score(dict(r), mode=args.judge_mode), axis=1)
            if "stratum" not in base_warm_df:
                strat_rows = []
                for _, r in base_warm_df.iterrows():
                    s = stratify_row(r, args.region, args.az)
                    s["stratum"] = "|".join([s["task"], s["ctx_bucket"], s["tool_use"], s["provider_model"], s["region"],
                                             s["az"], s["cold_warm"], s["cache_hit_prompt"], s["cache_hit_kv"]])
                    strat_rows.append(s)
                strat_df = pd.DataFrame(strat_rows)
                base_warm_df = pd.concat([base_warm_df.reset_index(drop=True), strat_df.reset_index(drop=True)], axis=1)
            # —— baseline 去重列名（关键） —— #
            base_warm_df = base_warm_df.loc[:, ~base_warm_df.columns.duplicated(keep="last")]
        except Exception:
            base_warm_df = None

    tests_A, family_A_meta = family_A_tost(
        stats.warm_df,
        base_warm_df,
        epsilon_q=args.epsilon_q,
        B=max(1000, args.bootstrap_B),
        seed=args.seed,
        require_family=args.require_family_A,
        min_n=args.min_n_per_stratum,
    )
    stats.summary["family_A_status"] = family_A_meta.get("status")
    if family_A_meta.get("reason"):
        stats.summary["family_A_reason"] = family_A_meta.get("reason")
    stats.summary["family_A_eligible_strata"] = family_A_meta.get("eligible_strata")
    stats.summary["family_A_tested_strata"] = family_A_meta.get("tested_strata")
    stats.summary["family_A_min_n"] = int(args.min_n_per_stratum)
    tests_B = family_B_p95_guardrail(stats.warm_df, base_warm_df, args)
    tests_C = family_C_json_guard(stats.warm_df, q=args.fdr_q)

    base_tag = write_outputs(stats.df_all, stats.summary, tests_A, tests_B, tests_C, args)

    # 控制台摘要
    print(f"\n==> run tag: {base_tag}")
    print(json.dumps(stats.summary, ensure_ascii=False, indent=2))
    if tests_A:
        print("\n[Family A / TOST] entries:", len(tests_A))
    if tests_B:
        print("[Family B / P95 Guardrail] entries:", len(tests_B))
    if tests_C:
        print("[Family C / JSON Fail] entries:", len(tests_C))

if __name__ == "__main__":
    main()
