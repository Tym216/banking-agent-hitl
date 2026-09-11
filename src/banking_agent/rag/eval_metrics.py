"""检索质量评估纯函数：匹配与指标聚合（不依赖检索器/LLM/DB，便于单测）。

匹配规则（对齐 v4 验证集修正后 schema）：
- fact.terms 元素：str → AND 必须出现；list → OR 组内任一出现即可
- 样本 match_policy：all（缺省）| any，聚合该样本全部 fact 的命中
- canon() = loader.normalize_text(s).lower()，检索正文与 terms 用同一规则
- any_of / doc_refs / golden_snippets 等字段一律忽略，不参与判定
指标口径：
- 仅 facts 非空的样本计入分母（out_of_scope / 无标注样本单独标记）
- Fact Recall@K = 命中 fact 数 / fact 总数；Sample Recall@K = 完全命中样本数 / 样本数
"""

from __future__ import annotations

from dataclasses import dataclass, field

from banking_agent.rag.loader import normalize_text


def canon(text: str) -> str:
    """归一化（全角→半角/标点统一/空白折叠）+ ASCII 小写。

    normalize_text 本身不含大小写处理，英文词（Red / Vino Volo 等）
    的稳定匹配依赖这里的 .lower()。
    """
    return normalize_text(text).lower()


def _entry_label(term: object) -> str:
    alts = term if isinstance(term, list) else [term]
    return "/".join(str(a) for a in alts)


def _entry_hit(term: object, merged: str) -> bool:
    alts = term if isinstance(term, list) else [term]
    return any(canon(str(a)) in merged for a in alts)


@dataclass
class FactCheck:
    fact_id: str
    hit: bool
    hit_terms: list[str] = field(default_factory=list)
    miss_terms: list[str] = field(default_factory=list)


@dataclass
class SampleCheck:
    sample_id: str
    oos: bool = False                  # facts 为空（含 out_of_scope / 未标注）
    match_policy: str = "all"
    sample_hit: bool = False
    facts: list[FactCheck] = field(default_factory=list)


def check_fact(fact: dict, merged_canon: str) -> FactCheck:
    """单个 fact 判定：terms 全满足（AND），list 元素组内任一即可（OR）。"""
    hit_terms: list[str] = []
    miss_terms: list[str] = []
    for term in fact.get("terms") or []:
        if _entry_hit(term, merged_canon):
            hit_terms.append(_entry_label(term))
        else:
            miss_terms.append(_entry_label(term))
    return FactCheck(
        fact_id=str(fact.get("fact_id", "")),
        hit=not miss_terms,
        hit_terms=hit_terms,
        miss_terms=miss_terms,
    )


def check_sample(facts: list[dict], merged_canon: str,
                 match_policy: str | None = None) -> SampleCheck:
    """样本级判定：facts 为空视为 oos（不进分母）；否则按 match_policy 聚合。"""
    if not facts:
        return SampleCheck(sample_id="", oos=True)
    checks = [check_fact(f, merged_canon) for f in facts]
    policy = (match_policy or "all").lower()
    if policy == "any":
        sample_hit = any(c.hit for c in checks)
    else:  # 缺省 all
        sample_hit = all(c.hit for c in checks)
    return SampleCheck(
        sample_id="", oos=False, match_policy=policy,
        sample_hit=sample_hit, facts=checks,
    )


@dataclass
class KSummary:
    k: int
    n_samples: int
    n_facts: int
    hit_samples: int
    hit_facts: int

    @property
    def sample_recall(self) -> float:
        return self.hit_samples / self.n_samples if self.n_samples else 0.0

    @property
    def fact_recall(self) -> float:
        return self.hit_facts / self.n_facts if self.n_facts else 0.0


def first_relevant_rank(chunk_canons: list[str], facts: list[dict]) -> int:
    """第一个单独完整满足任一 fact 的 chunk 排名（1-based）；无则 0。

    MRR 局限性：只统计"单个 chunk 独立覆盖某个 fact"的情形；需要多个
    chunk 合并才能命中的 fact 对该指标不可见，因此 MRR 会系统性偏低，
    仅作为辅助排序指标。
    """
    for i, ct in enumerate(chunk_canons, 1):
        if any(check_fact(f, ct).hit for f in facts):
            return i
    return 0


def summarize(samples: list[SampleCheck]) -> KSummary:
    """对一个 K 值下所有样本的检查结果聚合（调用方保证已排除 oos）。"""
    return KSummary(
        k=0,
        n_samples=len(samples),
        n_facts=sum(len(s.facts) for s in samples),
        hit_samples=sum(1 for s in samples if s.sample_hit),
        hit_facts=sum(1 for s in samples for f in s.facts if f.hit),
    )
