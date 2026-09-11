"""检索评估纯函数测试：AND/嵌套 OR/策略聚合/规范化/oos 排除/切片。"""

from __future__ import annotations

from banking_agent.rag.eval_metrics import (
    canon,
    check_fact,
    check_sample,
    first_relevant_rank,
    summarize,
)

_MERGED = canon("""
HSBC EveryMile 信用卡：在底特律机场的 Vino Volo 餐厅可用免费餐食，meal set 或 免費套餐。
Red 信用卡永久豁免年费。Visa Signature 卡年费 1800 元。
""")


def test_canon_fullwidth_case_whitespace():
    # 全角→半角、标点映射、空白折叠、ASCII 小写；繁体字不做简繁转换
    assert canon("ＡＢＣ，Red　卡 年費") == "abc,red 卡 年費"
    assert "vino volo" in canon("Vino Volo")


def test_fact_and_and_nested_or():
    fact = {
        "fact_id": "detroit_dining",
        "terms": [["Detroit", "底特律"], "Vino Volo", ["meal set", "免費套餐"]],
    }
    c = check_fact(fact, _MERGED)
    assert c.hit is True
    assert c.miss_terms == []
    assert c.hit_terms == ["Detroit/底特律", "Vino Volo", "meal set/免費套餐"]


def test_fact_missing_term_reports_miss():
    fact = {"fact_id": "x", "terms": ["Vino Volo", "上海虹桥机场"]}
    c = check_fact(fact, _MERGED)
    assert c.hit is False
    assert c.hit_terms == ["Vino Volo"]
    assert c.miss_terms == ["上海虹桥机场"]


def test_fact_any_of_field_ignored():
    """any_of 已从格式移除，即使残留也不参与判定。"""
    fact = {
        "fact_id": "y",
        "terms": ["不存在词A", "不存在词B"],
        "any_of": ["Vino Volo"],  # 若被错误读取，此 fact 会通过
    }
    c = check_fact(fact, _MERGED)
    assert c.hit is False


def test_sample_policy_all_default_and_any():
    facts = [
        {"fact_id": "f1", "terms": ["Vino Volo"]},
        {"fact_id": "f2", "terms": ["不存在的关键词"]},
    ]
    # 缺省 all
    s = check_sample(facts, _MERGED)
    assert s.match_policy == "all"
    assert s.sample_hit is False
    # any
    s = check_sample(facts, _MERGED, match_policy="any")
    assert s.sample_hit is True


def test_oos_sample_excluded():
    s = check_sample([], _MERGED)
    assert s.oos is True
    s2 = check_sample([{"fact_id": "a", "terms": ["Vino Volo"]}], _MERGED)
    assert s2.oos is False


def test_terms_can_be_str_instead_of_list():
    fact = {"fact_id": "s", "terms": "Vino Volo"}  # 容错：单个字符串
    assert check_fact(fact, _MERGED).hit is True


def test_summarize_counts():
    ok = check_sample([{"fact_id": "a", "terms": ["Vino Volo"]}], _MERGED)
    half = check_sample(
        [{"fact_id": "a", "terms": ["Vino Volo"]},
         {"fact_id": "b", "terms": ["不存在"]}],
        _MERGED,
    )
    sum_ = summarize([ok, half])
    assert sum_.n_samples == 2
    assert sum_.n_facts == 3
    assert sum_.hit_samples == 1
    assert sum_.hit_facts == 2
    assert sum_.sample_recall == 0.5
    assert abs(sum_.fact_recall - 2 / 3) < 1e-9


def test_chunk_merge_then_slice_by_k():
    """模拟：检索 top_k=50 的 hits 按 K 切片后合文本文。"""
    from banking_agent.rag.loader import Chunk

    hits = [
        (Chunk("a.md", 0, "Vino Volo 免费餐食"), 0.9),
        (Chunk("b.md", 0, "Red 卡年费"), 0.5),
        (Chunk("c.md", 0, "无关内容"), 0.1),
    ]
    # K=1 只含 a → 两个 fact 都缺一个 → miss
    merged1 = canon("\n".join(c.text for c, _ in hits[:1]))
    fact_ab = {"fact_id": "ab", "terms": ["Vino Volo", "Red"]}
    assert check_fact(fact_ab, merged1).hit is False
    # K=2 含 a+b → AND 全命中
    merged2 = canon("\n".join(c.text for c, _ in hits[:2]))
    assert check_fact(fact_ab, merged2).hit is True


def test_first_relevant_rank():
    facts = [
        {"fact_id": "f1", "terms": ["Vino Volo", "免费餐食"]},   # 单 chunk a 完整覆盖
        {"fact_id": "f2", "terms": ["Red", "年费"]},            # 单 chunk b 完整覆盖
    ]
    texts = [canon(t) for t in
             ["Vino Volo 餐厅提供免费餐食", "Red 卡年费 480", "无关内容"]]
    assert first_relevant_rank(texts, facts) == 1

    texts2 = [canon(t) for t in
              ["随便聊聊", "Vino Volo 餐厅提供免费餐食", "Red 卡年费 480"]]
    assert first_relevant_rank(texts2, facts) == 2


def test_mrr_zero_when_fact_needs_merge():
    """fact 需跨 chunk 合并才命中 → 单 chunk 均不相关 → MRR 归零（局限性）。"""
    facts = [{"fact_id": "f1", "terms": ["Vino Volo", "Red"]}]  # 分散在 a、b 两个 chunk
    texts = [canon(t) for t in ["Vino Volo 餐厅", "Red 卡年费", "无关"]]
    assert first_relevant_rank(texts, facts) == 0
