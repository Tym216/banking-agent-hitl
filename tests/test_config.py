"""配置加载：环境变量覆盖 YAML 文件（load_config 优先级契约）。"""

from __future__ import annotations

import os

from banking_agent.config import load_config


def test_env_overrides_yaml(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "storage:\n  db_path: \"data/from_yaml.db\"\n  checkpoint_db_path: \"data/ckpt_yaml.db\"\n"
        "rag:\n  high_threshold: 0.9\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BANKING_AGENT_STORAGE__DB_PATH", "data/from_env.db")
    monkeypatch.setenv("BANKING_AGENT_RAG__HIGH_THRESHOLD", "0.55")

    cfg = load_config(yaml_path)

    # env 覆盖 YAML 中已有的键
    assert cfg.storage.db_path == "data/from_env.db"
    assert cfg.rag.high_threshold == 0.55
    # 未被 env 覆盖的键保持 YAML 值
    assert cfg.storage.checkpoint_db_path == "data/ckpt_yaml.db"


def test_yaml_used_when_no_env(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("rag:\n  low_threshold: 0.3\n", encoding="utf-8")
    for key in list(os.environ):
        if key.startswith("BANKING_AGENT_"):
            monkeypatch.delenv(key)

    cfg = load_config(yaml_path)

    assert cfg.rag.low_threshold == 0.3
    assert cfg.llm.provider == "mock"  # 未配置的走默认值
