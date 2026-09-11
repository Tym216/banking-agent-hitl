"""检索管线测试：混合检索 / 融合 / min_score / 元数据过滤 / 精排。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from banking_agent.config import RAGConfig
from banking_agent.rag.embeddings import MockEmbedder
from banking_agent.rag.loader import load_knowledge_base
from banking_agent.rag.retriever import Retriever
from banking_agent.rag.store import VectorStore

_DOCS = {
    "a.md": "信用卡年费政策：标准信用卡年费为每年 200 元，金卡 480 元。消费满 6 笔可免次年年费。",
    "b.json": (
        '[{"id": "x1", "title": "转账限额", "content": "个人手机银行单笔转账限额 5 万元。",'
        ' "language": "zh-HK"}]'
    ),
    "c.md": "理财产品风险提示：投资有风险，过往业绩不代表未来表现。",
}


def _build(cfg: RAGConfig) -> Retriever:
    tmp = tempfile.mkdtemp()
    for name, content in _DOCS.items():
        (Path(tmp) / name).write_text(content, encoding="utf-8")
    chunks = load_knowledge_base(Path(tmp), cfg.chunk_size, cfg.chunk_overlap,
                                 metadata_blacklist=cfg.metadata_blacklist)
    store = VectorStore(MockEmbedder(512))
    store.build(chunks, Path(tmp) / "meta.db")
    return Retriever(store, cfg)


def _names(hits) -> list[str]:
    return [c.doc_id for c, _ in hits]


def test_single_embedding_retriever():
    r = _build(RAGConfig(chunk_size=200, overlap=20, retrievers={"type": "embedding"}))
    res = r.retrieve("信用卡年费是多少")
    assert not res.is_empty()
    assert res.hits[0][0].doc_id == "a.md"


def test_hybrid_rrf_fusion_both_sources_contribute():
    r = _build(RAGConfig(
        chunk_size=200, overlap=20,
        retrievers=[
            {"type": "embedding", "weight": 0.6},
            {"type": "bm25", "weight": 0.4},
        ],
    ))
    res = r.retrieve("信用卡年费是多少")
    assert not res.is_empty()
    # a.md 同时被两路召回，RRF 分数高于单路召回的 b.json
    top = res.hits[0]
    assert top[0].doc_id == "a.md"
    assert top[1] > 0.03  # 两路 1/(60+1) 之和 ≈ 0.0328


def test_hybrid_weighted_sum():
    r = _build(RAGConfig(
        chunk_size=200, overlap=20,
        fusion={"method": "weighted_sum"},
        retrievers=[
            {"type": "embedding", "weight": 0.6},
            {"type": "bm25", "weight": 0.4},
        ],
    ))
    res = r.retrieve("转账限额")
    assert not res.is_empty()
    assert res.hits[0][0].doc_id == "b.json"


def test_min_score_filters_embedding_only():
    """embedding 低于 min_score 的候选被丢弃；BM25 不受该阈值影响。"""
    r = _build(RAGConfig(
        chunk_size=200, overlap=20,
        retrievers=[
            {"type": "embedding", "min_score": 0.9},  # mock 分数域远低于 0.9 → embedding 全灭
            {"type": "bm25"},
        ],
    ))
    res = r.retrieve("信用卡年费是多少")
    assert not res.is_empty()  # BM25 仍能召回
    assert res.hits[0][0].doc_id == "a.md"

    r2 = _build(RAGConfig(chunk_size=200, overlap=20,
                          retrievers={"type": "embedding", "min_score": 0.9}))
    assert r2.retrieve("信用卡年费是多少").is_empty()


def test_metadata_filter_on_off():
    """过滤开关：开→限定范围；关→全量。"""
    cfg_on = RAGConfig(chunk_size=200, overlap=20,
                       enable_metadata_filter=True,
                       retrievers={"type": "embedding"})
    r_on = _build(cfg_on)
    # b.json 的 metadata 含 title="转账限额"，精确匹配命中
    res = r_on.retrieve("转账限额", filters={"title": "转账限额"})
    assert _names(res.hits) == ["b.json"]
    # 过滤条件不匹配其他文档
    res = r_on.retrieve("年费", filters={"title": "转账限额"})
    assert all(n == "b.json" for n in _names(res.hits))

    cfg_off = RAGConfig(chunk_size=200, overlap=20,
                        enable_metadata_filter=False,
                        retrievers={"type": "embedding"})
    r_off = _build(cfg_off)
    res = r_off.retrieve("年费", filters={"title": "转账限额"})  # 开关关 → 过滤不生效
    assert "a.md" in _names(res.hits)


def test_metadata_filter_empty_falls_back():
    """过滤后候选为空 → 回退未过滤结果。"""
    r = _build(RAGConfig(chunk_size=200, overlap=20,
                         enable_metadata_filter=True,
                         retrievers={"type": "embedding"}))
    res = r.retrieve("信用卡年费", filters={"title": "不存在的标题"})
    assert not res.is_empty()
    assert "a.md" in _names(res.hits)


def test_rerank_mock_reorders_and_truncates():
    """Mock 精排：按字符重叠重排，取 rerank.top_k。"""
    r = _build(RAGConfig(
        chunk_size=200, overlap=20, top_k=1,
        rerank={"enabled": True, "provider": "mock", "top_k": 2},
        retrievers={"type": "embedding"},
    ))
    # 查询与 b.json 的 content 高度重叠 → 精排后应升到第一
    res = r.retrieve("转账限额 5 万元")
    assert not res.is_empty()
    assert len(res.hits) == 1  # top_k=1
    assert res.hits[0][0].doc_id == "b.json"


def test_retrieve_empty_gives_refuse_log():
    """无关查询在 min_score 粗滤下为空 → refuse 日志。mock 真实命中 ~0.4-0.7，噪声 ~0.016。"""
    r = _build(RAGConfig(chunk_size=200, overlap=20,
                         retrievers={"type": "embedding", "min_score": 0.1}))
    res = r.retrieve("火星移民贷款业务")
    assert res.is_empty()
    log = res.to_log_dict()
    assert log["decision"] == "refuse"
    assert log["hits"] == []


def test_bm25_min_score_forbidden():
    import pytest

    with pytest.raises(ValueError):
        RAGConfig(retrievers=[{"type": "bm25", "min_score": 0.1}])


def test_retrieve_top_k_override():
    """评估用：cfg.top_k=2 但传 top_k=50 时返回全部候选；默认仍截断到配置值。"""
    cfg = RAGConfig(chunk_size=200, overlap=20, top_k=2,
                    retrievers={"type": "embedding"})
    tmp = tempfile.mkdtemp()
    for name, content in _DOCS.items():
        (Path(tmp) / name).write_text(content, encoding="utf-8")
    chunks = load_knowledge_base(Path(tmp), cfg.chunk_size, cfg.chunk_overlap)
    store = VectorStore(MockEmbedder(512))
    store.build(chunks, Path(tmp) / "meta.db")
    r = Retriever(store, cfg)

    default = r.retrieve("信用卡年费是多少")
    assert len(default.hits) == 2  # cfg.top_k=2

    overridden = r.retrieve("信用卡年费是多少", top_k=50)
    assert len(overridden.hits) == len(_DOCS)  # 全部 3 个候选
    assert overridden.hits[0][0].doc_id == default.hits[0][0].doc_id


def test_rerank_candidate_pool_limits_input(monkeypatch):
    """candidate_pool 只精排融合后的前 N 个候选；None 时精排全部。"""
    import banking_agent.rag.retriever as retriever_mod

    seen: list[int] = []

    class SpyReranker:
        def rerank(self, query: str, chunks) -> list[float]:
            seen.append(len(chunks))
            return [float(len(chunks) - i) for i in range(len(chunks))]

    monkeypatch.setattr(retriever_mod, "create_reranker", lambda cfg: SpyReranker())
    tmp = tempfile.mkdtemp()
    for name, content in _DOCS.items():
        (Path(tmp) / name).write_text(content, encoding="utf-8")
    chunks = load_knowledge_base(Path(tmp), 200, 20)
    store = VectorStore(MockEmbedder(512))
    store.build(chunks, Path(tmp) / "meta.db")

    cfg_all = RAGConfig(chunk_size=200, overlap=20,
                        rerank={"enabled": True, "provider": "mock", "top_k": 10})
    r_all = Retriever(store, cfg_all)
    res_all = r_all.retrieve("信用卡年费是多少", top_k=50)
    assert seen and seen[0] == len(_DOCS)      # 默认精排全部 3 个

    cfg_pool = RAGConfig(chunk_size=200, overlap=20,
                         rerank={"enabled": True, "provider": "mock",
                                 "top_k": 10, "candidate_pool": 2})
    r_pool = Retriever(store, cfg_pool)
    res_pool = r_pool.retrieve("信用卡年费是多少", top_k=50)
    assert seen[-1] == 2                        # 只送 2 个进精排
    assert len(res_pool.hits) <= 2              # 结果不会超出候选池
