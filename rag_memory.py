"""
可选的 RAG 记忆层：LanceDB 存向量 + Ollama 嵌入 API 生成向量。
未安装 lancedb 或 WORLD_RAG=0 时，create_protagonist_rag() 返回 None。

Ollama 0.3.4+ 使用 POST /api/embed + JSON 字段 input
本模块优先尝试 /api/embed，失败再回退 /api/embeddings。

环境变量：
  WORLD_RAG=0           关闭 RAG
  WORLD_RAG_PATH=...    LanceDB 目录（默认 .world_data/lance_zero）
  OLLAMA_EMBED_MODEL    嵌入模型名（默认 nomic-embed-text，需 ollama pull）
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Set

import requests

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _ollama_base_url() -> str:
    u = (OLLAMA_API_URL or "").strip()
    for suf in ("/api/generate", "/api/chat", "/v1/chat/completions"):
        if u.endswith(suf):
            return u[: -len(suf)].rstrip("/")
    if "/api/" in u:
        return u.split("/api/", 1)[0].rstrip("/")
    return u.rstrip("/") or "http://localhost:11434"


def ollama_embed(text: str, timeout: int = 90) -> List[float]:
    """
    调用 Ollama 嵌入接口。
    优先 /api/embed（body: model + input），再尝试 /api/embeddings（body: model + prompt）。
    """
    base = _ollama_base_url()
    candidates = [
        (f"{base}/api/embed", {"model": OLLAMA_EMBED_MODEL, "input": text}),
        (f"{base}/api/embeddings", {"model": OLLAMA_EMBED_MODEL, "prompt": text}),
    ]
    last_err: Optional[Exception] = None
    for url, payload in candidates:
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "embeddings" in data and data["embeddings"]:
                emb = data["embeddings"][0]
            else:
                emb = data.get("embedding")
            if not emb:
                raise ValueError(f"无向量字段: {str(data)[:200]}")
            return [float(x) for x in emb]
        except Exception as exc:
            last_err = exc
            continue
    raise last_err if last_err else RuntimeError("ollama_embed: 无可用嵌入端点")


class ProtagonistRAG:
    """
    将零的记忆文本同步为向量行；按查询句检索 top-k 片段供 Planner / 决策使用。
    """

    def __init__(self, db_path: str, session_id: str) -> None:
        import lancedb  # noqa: PLC0415

        self.db_path = db_path
        self.session_id = session_id
        self.seen_ids: Set[str] = set()
        self._db = lancedb.connect(db_path)
        self._table_name = f"zero_memories_{session_id}"
        self._table: Any = None
        self._embed_disabled = False
        self._embed_disable_message_printed = False

    def _ensure_table(self, first_row: Dict[str, Any]) -> None:
        if self._table is not None:
            return
        if self._table_name in self._db.table_names():
            self._table = self._db.open_table(self._table_name)
        else:
            self._table = self._db.create_table(self._table_name, data=[first_row])

    def sync_from_memory_system(self, memory) -> None:
        """把 MemorySystem 里尚未索引的条目写入 LanceDB。"""
        if self._embed_disabled:
            return
        for mid, item in memory.memories.items():
            if mid in self.seen_ids:
                continue
            text = (item.content or "").strip()
            if not text:
                self.seen_ids.add(mid)
                continue
            try:
                vec = ollama_embed(text[:2000])
            except Exception as exc:
                self._embed_disabled = True
                if not self._embed_disable_message_printed:
                    print(
                        "[RAG] Ollama 嵌入不可用，已停止向量同步与检索。"
                        f" 原因：{exc}\n"
                        "      请确认：1) ollama serve 已运行  2) 已执行 "
                        f"`ollama pull {OLLAMA_EMBED_MODEL}`  3) 新版 Ollama 使用 /api/embed（本程序已自动尝试）。"
                    )
                    self._embed_disable_message_printed = True
                return
            row = {"id": mid, "text": text[:4000], "vector": vec}
            if self._table is None:
                self._ensure_table(row)
            else:
                self._table.add([row])
            self.seen_ids.add(mid)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._embed_disabled or not query.strip() or self._table is None:
            return []
        try:
            qv = ollama_embed(query[:1500])
            return self._table.search(qv).limit(k).to_list()
        except Exception:
            return []

    def format_snippets(self, query: str, k: int = 5, max_chars: int = 1200) -> str:
        rows = self.search(query, k=k)
        if not rows:
            return ""
        parts: List[str] = []
        total = 0
        for row in rows:
            rid = row.get("id") or ""
            if rid == "__seed__":
                continue
            t = (row.get("text") or "").strip()
            if not t:
                continue
            line = f"- {t}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        return "\n".join(parts)


def _normalize_session_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\\-]", "_", (raw or "").strip())
    return cleaned[:48] if cleaned else f"s{int(time.time())}"


def create_protagonist_rag(session_id: str = "") -> Optional[ProtagonistRAG]:
    if os.getenv("WORLD_RAG", "1").lower() in ("0", "false", "no", "off"):
        return None
    try:
        import lancedb  # noqa: F401, PLC0415
    except ImportError:
        print("[RAG] 未安装 lancedb，已禁用向量记忆。安装：pip install lancedb")
        return None
    try:
        db_path = os.getenv("WORLD_RAG_PATH", os.path.join(".world_data", "lance_zero"))
        os.makedirs(db_path, exist_ok=True)
        sid = _normalize_session_id(session_id or f"s{int(time.time())}")
        return ProtagonistRAG(db_path, sid)
    except Exception as exc:
        print(f"[RAG] 初始化失败，已禁用向量记忆。原因：{exc}")
        return None
