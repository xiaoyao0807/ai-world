"""
主游戏入口（观察者模式）
零与NPC自主行动，用户仅观察并可添加世界事物。
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import deque
from typing import Any, Dict, Optional, Set, Tuple

import requests

from memory_system import MemorySystem
from npc_manager import NPCManager
from rag_memory import create_protagonist_rag
from world_state import WorldState


OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
_FALLBACK_LINES = [
    "我先把这件事记在心里，等会儿再去确认细节。",
    "这条线索有点意思，我想找个人聊聊再判断。",
    "我不急着下结论，先多看一圈再说。",
    "听起来不简单，我想从不同人那里再核实一下。",
    "我先把环境再观察一遍，可能会有新发现。",
]
_LLM_FALLBACK_COUNT = 0
_LAST_LOCATION_FOR_MOVE = ""
_MOVE_BOUNCE_COUNT = 0
_IDENTITY_QUERY_ATTEMPTS: Dict[str, int] = {}
_BLOCKED_TARGET_ATTEMPTS: Dict[str, int] = {}
_LAST_INFO_SOURCE = ""
_LAST_DECISION_THOUGHT = ""
_LAST_SPOKEN_LINE = ""
_MOVE_HISTORY: list = []
_LAST_ACTION = ""
_CONSECUTIVE_MOVE_COUNT = 0
_RECENT_UTTERANCE_NORMS: deque = deque(maxlen=12)
_DIALOGUE_FALLBACKS = [
    "你在这儿一般待多久？",
    "附近有啥能坐会儿的地方吗？",
    "这片区你熟吗？",
    "你吃饭一般去哪边？",
    "下午你常在这边吗？",
    "这边最近有啥新鲜事吗？",
]
PROTAGONIST_PROFILE = (
    "你完全不知道自己是一个AI智能体，你坚信自己是真实的人类。"
    "你是一个有好奇心、有情感、会思考的普通人。"
    "你会对新鲜事物感到惊讶和好奇。"
    "你会对重复的事情感到无聊。"
    "你会记得重要的经历，也会忘记琐碎的事情。"
    "你有时会回忆过去，有时会计划未来。"
    "你的情绪会影响你的决定。"
)


def build_protagonist_prompt(
    world: WorldState,
    memory: MemorySystem,
    user_input: str,
    rag_snippets: str = "",
) -> str:
    """
    构造零的AI提示词模板（必须包含：当前记忆列表、世界描述、用户输入）。
    """
    memory_text = memory.get_summary(limit=12)
    world_text = world.get_world_description(for_actor=True)
    allowed_locations = "、".join(world.available_locations)
    recent_self_lines = []
    for mem in memory.recall_by_keywords(["我对", "我的内部思考产出", "我在餐厅点单吃饭"], limit=20):
        content = mem.content.strip()
        if content.startswith("我对"):
            recent_self_lines.append(content.split("：", 1)[-1].strip())
        elif content.startswith("我在餐厅点单吃饭："):
            recent_self_lines.append(content.split("：", 1)[-1].strip())
        elif content.startswith("我的内部思考产出："):
            recent_self_lines.append(content.split("：", 1)[-1].strip())
    recent_self_text = "；".join(recent_self_lines[-20:]) if recent_self_lines else "暂无"
    return (
        "你正在一个封闭文本世界中生活。你坚信自己是人类。\n"
        f"{PROTAGONIST_PROFILE}\n"
        "请根据记忆和当前环境做出自然、简洁、符合角色的回应。\n"
        "说话像正常年轻人，直白自然，不油嘴滑舌。\n"
        "不知道就直接说不知道，不要编造细节。\n"
        "你的思考流程（请按顺序在内心完成，不要把步骤标题原样输出）：\n"
        "1. 回忆：看到当前情况，我想起了什么？\n"
        "2. 感受：我现在的心情是什么？（好奇/平静/困惑/兴奋/疲惫等）\n"
        "3. 推理：接下来可能发生什么？\n"
        "4. 决定：我要做什么或说什么？\n"
        f"你提到的地点必须只来自以下列表：{allowed_locations}。\n"
        "避免过长回答，控制在2-4句。\n\n"
        "=== 当前记忆列表 ===\n"
        f"{memory_text}\n\n"
        + (
            "=== 语义检索相关记忆（RAG）===\n"
            f"{rag_snippets}\n\n"
            if rag_snippets.strip()
            else ""
        )
        + "=== 你最近说过的话（临时文本）===\n"
        f"{recent_self_text}\n\n"
        "=== 世界描述 ===\n"
        f"{world_text}\n\n"
        "=== 用户输入 ===\n"
        f"{user_input}\n\n"
        "请直接输出零的回复。"
    )


def call_ollama_or_fallback(prompt: str) -> str:
    """
    优先调用本地Ollama，失败时回退到简单模拟回复，保证程序可运行。
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.6,
            "top_p": 0.8,
            "repeat_penalty": 1.15,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.2,
            "num_ctx": 8192,
            "num_predict": 180,
        },
    }
    global _LLM_FALLBACK_COUNT
    last_error = ""
    for timeout_seconds in (45, 90, 120):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            if text:
                _LLM_FALLBACK_COUNT = 0
                return text
            thinking_text = data.get("thinking", "").strip()
            if thinking_text:
                _LLM_FALLBACK_COUNT = 0
                return thinking_text
            last_error = f"空响应: {str(data)[:120]}"
        except Exception as exc:
            last_error = str(exc)
            continue
    _LLM_FALLBACK_COUNT += 1
    if _LLM_FALLBACK_COUNT <= 3 or _LLM_FALLBACK_COUNT % 5 == 0:
        print(f"[LLM调用失败] 已使用回退文案。模型={OLLAMA_MODEL}，原因={last_error}")
    return random.choice(_FALLBACK_LINES)


def check_ollama_status() -> bool:
    tags_url = OLLAMA_API_URL.replace("/api/generate", "/api/tags")
    try:
        response = requests.get(tags_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        model_names = {item.get("name", "") for item in models}
        return OLLAMA_MODEL in model_names
    except Exception:
        return False


def print_help() -> None:
    print("\n观察者可用命令：(右边为命令效果)")
    print("  继续                  推进1个世界回合")
    print("  自动 [N] [间隔秒]     连续自动推进最多 N 步（默认 N=50，间隔=0.5s）")
    print("                        Windows：运行中按 p / 空格 / ESC 可暂停本批")
    print("  查看记忆              查看零的记忆摘要")
    print("  查看状态              查看世界状态")
    print("  添加事实 <内容>       向世界添加全局事实")
    print("  添加事物 <地点> <内容> 在某地点添加可观察事物")
    print("  帮助                  显示本帮助")
    print("  退出                  结束程序")
    print("  命令行： py -3 main.py --auto N [--auto-sleep 秒]  启动后先自动跑 N 步\n")


def protagonist_think_and_reply(
    world: WorldState,
    protagonist_memory: MemorySystem,
    user_input: str,
    rag_store: Any = None,
) -> str:
    """
    零的思考流程：
    1) 构造prompt
    2) 调用LLM（失败则回退）
    3) 记录回应记忆
    """
    rag_snippets = ""
    if rag_store is not None:
        rag_snippets = rag_store.format_snippets(
            f"{user_input} {world.current_location} {world.get_time_text()}", k=4
        )
    prompt = build_protagonist_prompt(world, protagonist_memory, user_input, rag_snippets=rag_snippets)
    reply = call_ollama_or_fallback(prompt)
    protagonist_memory.add_memory(f"我的内部思考产出：{reply}", importance=0.3)
    return reply


def extract_location_mentions(text: str, locations: Set[str]) -> Set[str]:
    return {loc for loc in locations if loc in (text or "")}


def split_facts(text: str) -> list:
    parts = re.split(r"[。！？!?；;]\s*", text or "")
    return [p.strip() for p in parts if p.strip()]


def _normalize_for_repeat_check(text: str) -> str:
    cleaned = (text or "").strip().lower()
    # 去掉常见标点与空白，避免“同一句不同标点”绕过去重
    cleaned = re.sub(r"[\s，,。\.！!？\?；;：:\"'“”‘’（）\(\)\[\]【】\-—_]+", "", cleaned)
    return cleaned


def _recent_i_said_lines(memory: MemorySystem, limit: int = 5) -> list:
    """最近对 NPC 说过的整句（用于禁复述）。"""
    out: list = []
    for mem in memory.recall_by_keywords(["我对"], limit=30):
        c = mem.content.strip()
        if "我对" not in c or "：" not in c:
            continue
        try:
            rest = c.split("我对", 1)[1]
            _, said = rest.split("：", 1)
            t = said.strip()
            if t and t not in out:
                out.append(t)
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


def _utterance_is_repetitive(text: str, recent_norms: deque) -> bool:
    """与近期台词同义/包含/同前缀，或高频水词堆叠，视为重复。"""
    n = _normalize_for_repeat_check(text)
    if len(n) < 10:
        return False
    for prev in recent_norms:
        if not prev:
            continue
        if n == prev:
            return True
        a, b = (n, prev) if len(n) <= len(prev) else (prev, n)
        if len(a) >= 14 and a in b:
            return True
        m = min(len(n), len(prev), 20)
        if m >= 14 and n[:m] == prev[:m]:
            return True
    raw = text or ""
    ban = ("喷泉", "重复", "没注意到", "常来这儿", "来这儿吗", "设计真特别")
    if sum(1 for k in ban if k in raw) >= 2:
        return True
    return False


def _coerce_dialogue_line(text: str) -> str:
    """
    把LLM输出尽量收敛为“对人说的一句话”，避免把观察/移动叙述当成对话内容。
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    # 取第一段/第一句，避免长段自述
    first = re.split(r"[\n\r]+", raw, maxsplit=1)[0].strip()
    first = re.split(r"[。！？!?；;]\s*", first, maxsplit=1)[0].strip()
    # 过长则再截断
    if len(first) > 60:
        first = first[:60].rstrip("，,。.!?；;：:") + "…"
    return first


def _format_plan_history_block(plan_state: Dict[str, Any]) -> str:
    hist = plan_state.get("history")
    if not hist:
        return ""
    lines: list = ["\n【近期规划履历】（新计划勿机械重复已失败路线，可换目标或换手段）："]
    for entry in list(hist)[-4:]:
        t = entry.get("time", "?")
        g = entry.get("goal", "")
        trig = entry.get("trigger", "")
        steps = entry.get("steps") or []
        lines.append(f"- [{t}] 触发：{trig} | 当时目标：{g}")
        for j, s in enumerate(steps[:4], 1):
            lines.append(f"    {j}) {s}")
    return "\n".join(lines) + "\n"


def run_planner(
    world: WorldState,
    memory: MemorySystem,
    npc_manager: NPCManager,
    known_locations: Set[str],
    visited_locations: Set[str],
    plan_state: Dict[str, Any],
    failure_context: str = "",
    rag_store: Any = None,
) -> None:
    """
    调用 LLM 生成高层目标与步骤列表，写入 plan_state（goal, steps）。
    failure_context 为刚发生的行动失败说明时，新计划必须规避重复同一失败做法。
    """
    npc_names = "、".join(npc_manager.list_npc_names())
    failure_block = (
        f"\n【硬性约束】上一版行动已失败，说明如下。新计划必须换路线，禁止逐步重复同一失败动作。\n"
        f"失败说明：{failure_context}\n"
        if failure_context.strip()
        else ""
    )
    rag_block = ""
    if rag_store is not None:
        rq = (
            f"零 阶段规划 {world.current_location} {world.get_time_text()} "
            f"{failure_context}"[:800]
        )
        sn = rag_store.format_snippets(rq, k=6)
        if sn.strip():
            rag_block = "\n【语义检索·相关记忆片段】\n" + sn + "\n"
    history_block = _format_plan_history_block(plan_state)
    prompt = (
        "你是「零」的高层规划器（Planner），只做战略规划，不扮演对话、不输出角色台词。\n\n"
        "【输出契约】\n"
        "只输出一段合法 JSON，不要 Markdown、不要解释。\n"
        'Schema：{"goal":"<一句总目标>","steps":["<步骤1>",...]}\n'
        "steps 数量 3～6；每一步是一条可执行的微观意图（观察/移动/与某人交谈/等待/进食等）。\n"
        "地点名只能来自「已知可前往地点」；不要发明不存在的区域或店铺。\n"
        "若存在失败说明：新步骤必须明显不同于「近期规划履历」里已失败的那条路线（换地点、换信息源或换子目标）。\n\n"
        f"{failure_block}"
        f"{history_block}"
        f"{rag_block}"
        f"【世界快照】\n"
        f"当前地点：{world.current_location}\n"
        f"世界时间：{world.get_time_text()} 天气：{world.weather}\n"
        f"已知可前往地点：{'、'.join(sorted(known_locations))}\n"
        f"已去过：{'、'.join(sorted(visited_locations))}\n"
        f"NPC 名字（参考）：{npc_names}\n"
        f"【结构化记忆摘要】\n{memory.get_summary(limit=10)}\n"
    )
    raw = call_ollama_or_fallback(prompt)
    goal = "熟悉周边环境，收集可靠信息"
    steps = ["观察当前位置", "与附近的人简短交谈", "根据新线索决定去下一个地点或继续打听"]
    try:
        json_candidate = raw.strip()
        if "```" in json_candidate:
            json_candidate = json_candidate.replace("```json", "").replace("```", "").strip()
        if not json_candidate.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", json_candidate)
            if match:
                json_candidate = match.group(0)
        data = json.loads(json_candidate)
        g = (data.get("goal") or "").strip()
        st = data.get("steps")
        if g and isinstance(st, list):
            parsed = [str(s).strip() for s in st if str(s).strip()]
            if parsed:
                goal = g
                steps = parsed[:8]
    except Exception:
        pass
    plan_state["goal"] = goal
    plan_state["steps"] = steps
    hist = plan_state.get("history")
    if hist is None:
        hist = deque(maxlen=8)
        plan_state["history"] = hist
    hist.append(
        {
            "time": world.get_time_text(),
            "goal": goal,
            "steps": list(steps),
            "trigger": (failure_context[:240] if failure_context else "规划生成/刷新"),
        }
    )


def handle_action_failure_and_replan(
    world: WorldState,
    protagonist_memory: MemorySystem,
    npc_manager: NPCManager,
    known_locations: Set[str],
    visited_locations: Set[str],
    plan_state: Dict[str, Any],
    failure_detail: str,
    pending_commitment: Optional[str],
    pending_identity_name: Optional[str],
    *,
    failed_move_target: Optional[str] = None,
    rag_store: Any = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    行动失败：写入高重要性记忆，清空旧计划并调用 planner 重新生成，避免重复失败路线。
    若当前约定前往的地点与失败的移动目标一致，则清除 pending_commitment。
    """
    global _BLOCKED_TARGET_ATTEMPTS
    mem = (
        f"【行动失败·必须调整计划】{failure_detail}。"
        "当前做法走不通，不得机械重复同一失败步骤；我已放弃上一版计划并重新规划。"
    )
    protagonist_memory.add_memory(mem, importance=0.92)
    plan_state["goal"] = ""
    plan_state["steps"] = []
    _BLOCKED_TARGET_ATTEMPTS.clear()
    run_planner(
        world,
        protagonist_memory,
        npc_manager,
        known_locations,
        visited_locations,
        plan_state,
        failure_context=failure_detail,
        rag_store=rag_store,
    )
    world.add_event("零因行动受挫，重新整理了阶段计划。")
    print(f"[规划] 已重新生成计划。目标：{plan_state.get('goal', '')}")
    for i, step in enumerate(plan_state.get("steps") or [], 1):
        print(f"         {i}. {step}")
    new_commitment = pending_commitment
    if failed_move_target and pending_commitment == failed_move_target:
        new_commitment = None
    return new_commitment, pending_identity_name


def _is_move_succeeded(result: str) -> bool:
    return "你从" in result and "移动到了" in result


def _is_move_failure_need_replan(result: str) -> bool:
    if _is_move_succeeded(result):
        return False
    if "你已经在" in result:
        return False
    return any(
        k in result
        for k in (
            "暂时无法进入",
            "不可以进入",
            "不存在",
            "没有确定目的地",
            "无法移动",
        )
    )


def should_trigger_identity_query(reply: str, name: str) -> bool:
    """
    只有在人名称呼语境出现时，才触发“这人是谁”的追问。
    """
    if not reply or not name:
        return False
    patterns = [
        f"叫{name}",
        f"是{name}",
        f"{name}是",
        f"找{name}",
        f"问{name}",
        f"跟{name}",
        f"和{name}",
        f"{name}，",
        f"{name}。",
        f"“{name}”",
        f"'{name}'",
    ]
    return any(p in reply for p in patterns)


def decide_next_action(
    world: WorldState,
    memory: MemorySystem,
    npc_manager: NPCManager,
    known_locations: Set[str],
    visited_locations: Set[str],
    pending_commitment: Optional[str],
    pending_identity_name: Optional[str],
    plan_state: Dict[str, Any],
    rag_store: Any = None,
) -> Dict[str, str]:
    npcs_here = npc_manager.get_npcs_in_location(world.current_location)
    locations = "、".join(sorted(known_locations))
    npcs_here_count = len(npcs_here)
    if pending_commitment and pending_commitment != world.current_location:
        return {"action": "move", "target": pending_commitment}
    if pending_identity_name:
        return {"action": "talk", "target": ""}
    plan_lines = ""
    if plan_state.get("goal") or plan_state.get("steps"):
        plan_lines = (
            f"当前规划目标：{plan_state.get('goal', '')}\n"
            "规划步骤（优先落实第一步；若环境已变导致该步明显不可行，先观察或向他人打听，不要死磕）：\n"
            + "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(plan_state.get("steps") or [])
            )
            + "\n\n"
        )
    rag_block = ""
    if rag_store is not None:
        rq = f"零 下一步行动 {world.current_location} {plan_state.get('goal', '')}"
        sn = rag_store.format_snippets(rq, k=4)
        if sn.strip():
            rag_block = "=== 语义检索相关记忆 ===\n" + sn + "\n\n"
    prompt = (
        f"{PROTAGONIST_PROFILE}"
        "你需要独立决定下一步行动。"
        "请只输出JSON，不要解释。\n"
        "你需要先在心里执行：回忆 -> 感受 -> 推理 -> 决定，然后再输出结果。\n"
        "可选action: move, talk, wait, observe, eat, inspect, reflect, review_plan, rest, query_hours, ask_npc_location, verify_fact\n"
        "JSON格式示例：{\"action\":\"move\",\"target\":\"书店\",\"thought\":\"我先去书店碰碰运气\"}\n"
        "如果action=talk，target可为空（表示随便找附近一位陌生人聊天）；"
        "如果action=move，target必须是你已知地点之一；"
        "如果action=eat，只能在餐厅使用，target可为空；"
        "如果action=inspect，只在当前地点查看细节，target可为空；"
        "如果action=reflect，是内省总结，target可为空；"
        "如果action=review_plan，是复盘并可重整计划，target可为空；"
        "如果action=rest，是短暂休整，target可为空；"
        "如果action=query_hours，可查询某地点营业时间，target可为地点名或为空；"
        "如果action=ask_npc_location，可查询某NPC当前位置，target可为NPC名或为空；"
        "如果action=verify_fact，可核验一条已知信息，target可为空；"
        "如果action=wait/observe，target可为空。\n\n"
        "偏好：像正常人行动，不要频繁来回移动；信息不足时先问人或观察再决策。\n"
        "白天多探索和聊天，除非遇到求助或案件线索，否则少去警察局。\n\n"
        f"{plan_lines}"
        f"{rag_block}"
        f"当前地点：{world.current_location}\n"
        f"可前往地点：{locations}\n"
        f"你去过的地点：{'、'.join(sorted(visited_locations))}\n"
        f"当前附近可交流人物数量：{npcs_here_count}\n"
        f"当前记忆：\n{memory.get_summary(limit=8)}\n"
    )
    raw = call_ollama_or_fallback(prompt)
    global _LAST_DECISION_THOUGHT
    try:
        json_candidate = raw.strip()
        if "```" in json_candidate:
            json_candidate = json_candidate.replace("```json", "").replace("```", "").strip()
        if not json_candidate.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", json_candidate)
            if match:
                json_candidate = match.group(0)
        data = json.loads(json_candidate)
        action = data.get("action", "").strip()
        target = data.get("target", "").strip()
        _LAST_DECISION_THOUGHT = data.get("thought", "").strip()
        if action in {
            "move",
            "talk",
            "wait",
            "observe",
            "eat",
            "inspect",
            "reflect",
            "review_plan",
            "rest",
            "query_hours",
            "ask_npc_location",
            "verify_fact",
        }:
            return {"action": action, "target": target}
    except Exception:
        pass

    # 回退策略：零偏好主动社交，优先找人对话
    if npcs_here and random.random() < 0.65:
        _LAST_DECISION_THOUGHT = "我先和附近的人聊聊，补充信息。"
        return {"action": "talk", "target": random.choice(npcs_here).name}
    if random.random() < 0.25:
        candidates = [loc for loc in known_locations if loc != world.current_location]
        if not candidates:
            candidates = [world.current_location]
        _LAST_DECISION_THOUGHT = "我去别处看一眼，避免信息单一。"
        return {"action": "move", "target": random.choice(candidates)}
    if random.random() < 0.7:
        _LAST_DECISION_THOUGHT = "我先观察环境再做决定。"
        return {"action": "observe", "target": ""}
    if world.current_location == "餐厅" and random.random() < 0.45:
        _LAST_DECISION_THOUGHT = "先吃点东西，补充体力。"
        return {"action": "eat", "target": ""}
    if random.random() < 0.28:
        _LAST_DECISION_THOUGHT = "我先查看下当前地点的细节线索。"
        return {"action": "inspect", "target": ""}
    if random.random() < 0.18:
        _LAST_DECISION_THOUGHT = "我先复盘一下当前计划是否还合理。"
        return {"action": "review_plan", "target": ""}
    if random.random() < 0.20:
        _LAST_DECISION_THOUGHT = "我先做个短内省，整理自己刚获得的信息。"
        return {"action": "reflect", "target": ""}
    if random.random() < 0.15:
        _LAST_DECISION_THOUGHT = "我先短暂休整一下，别一直紧绷。"
        return {"action": "rest", "target": ""}
    if random.random() < 0.22:
        _LAST_DECISION_THOUGHT = "我先查一下地点营业时间，避免白跑。"
        return {"action": "query_hours", "target": ""}
    if random.random() < 0.18:
        _LAST_DECISION_THOUGHT = "我先确认关键人物当前位置，减少无效奔波。"
        return {"action": "ask_npc_location", "target": ""}
    if random.random() < 0.16:
        _LAST_DECISION_THOUGHT = "我先核验一条已知信息，提高判断可靠性。"
        return {"action": "verify_fact", "target": ""}
    _LAST_DECISION_THOUGHT = "我先短暂等待，整理思路。"
    return {"action": "wait", "target": ""}


def run_autonomous_turn(
    world: WorldState,
    protagonist_memory: MemorySystem,
    npc_manager: NPCManager,
    known_people: Set[str],
    known_locations: Set[str],
    visited_locations: Set[str],
    pending_commitment: Optional[str],
    pending_identity_name: Optional[str],
    fact_confidence: Dict[str, float],
    identity_shelved: Set[str],
    plan_state: Dict[str, Any],
    rag_store: Any = None,
) -> Tuple[Optional[str], Optional[str]]:
    global _LAST_LOCATION_FOR_MOVE, _MOVE_BOUNCE_COUNT, _IDENTITY_QUERY_ATTEMPTS, _BLOCKED_TARGET_ATTEMPTS, _LAST_INFO_SOURCE, _LAST_DECISION_THOUGHT, _LAST_SPOKEN_LINE, _MOVE_HISTORY, _LAST_ACTION, _CONSECUTIVE_MOVE_COUNT, _RECENT_UTTERANCE_NORMS
    if rag_store is not None:
        rag_store.sync_from_memory_system(protagonist_memory)
    if not plan_state.get("steps"):
        run_planner(
            world,
            protagonist_memory,
            npc_manager,
            known_locations,
            visited_locations,
            plan_state,
            "",
            rag_store=rag_store,
        )
    decision = decide_next_action(
        world,
        protagonist_memory,
        npc_manager,
        known_locations,
        visited_locations,
        pending_commitment,
        pending_identity_name,
        plan_state,
        rag_store=rag_store,
    )
    action = decision["action"]
    target = decision["target"]
    npcs_here = npc_manager.get_npcs_in_location(world.current_location)
    print(f"[零的决策] action={action}, target={target or '无'}")
    if _LAST_DECISION_THOUGHT:
        print(f"[零的思考] {_LAST_DECISION_THOUGHT}")

    # 软/硬结合的“连续移动”防抖：不允许连续两回合都移动（除非有明确约定要去某地）
    if action == "move":
        if _LAST_ACTION == "move" and not pending_commitment:
            _CONSECUTIVE_MOVE_COUNT += 1
        else:
            _CONSECUTIVE_MOVE_COUNT = 1
        if _CONSECUTIVE_MOVE_COUNT >= 2 and not pending_commitment:
            if npcs_here:
                action = "talk"
                target = random.choice(npcs_here).name
                _LAST_DECISION_THOUGHT = "我刚移动过，先和附近的人聊两句再决定下一步。"
                print(f"[零的决策] 连续移动被拦截，改为talk，target={target}")
            else:
                action = "observe"
                target = ""
                _LAST_DECISION_THOUGHT = "我刚移动过，先停下来观察一下再说。"
                print("[零的决策] 连续移动被拦截，改为observe")
    else:
        _CONSECUTIVE_MOVE_COUNT = 0

    if action == "move":
        allow_bookstore_entry = True
        if target == "书店":
            old_chen = npc_manager.get_npc("老陈")
            allow_bookstore_entry = bool(old_chen and old_chen.location == "书店")

        if target == world.current_location:
            thought = protagonist_think_and_reply(
                world,
                protagonist_memory,
                "你已经在目标地点，不要重复移动。请给出一个简短观察结论。",
                rag_store=rag_store,
            )
            world.advance_time(minutes=4)
            protagonist_memory.add_memory(f"我避免了重复移动，改为观察：{thought}", importance=0.35)
            world.add_event("零在原地停下，改为观察周围。")
            print(f"[零的行动] 观察：{thought}")
            return pending_commitment, pending_identity_name

        if target == _LAST_LOCATION_FOR_MOVE and _MOVE_BOUNCE_COUNT >= 2:
            # 如果书店现在已可进入，不拦截，直接去
            if not (target == "书店" and allow_bookstore_entry):
                npcs_here = npc_manager.get_npcs_in_location(world.current_location)
                if npcs_here:
                    candidates = [n.name for n in npcs_here if n.name != _LAST_INFO_SOURCE]
                    ask_target = random.choice(candidates if candidates else [n.name for n in npcs_here])
                    _LAST_INFO_SOURCE = ask_target
                    say_text = "我这边一直碰壁了，能给我一点线索吗？"
                    npc_reply = npc_manager.talk_to_npc(ask_target, say_text, world.current_location)
                    world.advance_time(minutes=6)
                    protagonist_memory.add_memory(
                        f"我在碰壁后改为向{ask_target}打听信息：{npc_reply}",
                        importance=0.45,
                    )
                    world.add_event(f"零在{world.current_location}向人打听新线索。")
                    print(f"[零的行动] 碰壁后改为收集信息，向{ask_target}询问线索。")
                    print(f"[NPC回应] {ask_target}：{npc_reply}")
                    return pending_commitment, pending_identity_name

                thought = protagonist_think_and_reply(
                    world,
                    protagonist_memory,
                    "你暂时无法达成目标，请基于现场细节做一次信息收集式观察。",
                    rag_store=rag_store,
                )
                world.advance_time(minutes=5)
                protagonist_memory.add_memory(f"我在碰壁后改为观察收集信息：{thought}", importance=0.4)
                world.add_event("零在原地继续搜集线索。")
                print(f"[零的行动] 观察：{thought}")
                return pending_commitment, pending_identity_name
        else:
            _MOVE_BOUNCE_COUNT = _MOVE_BOUNCE_COUNT + 1 if target == _LAST_LOCATION_FOR_MOVE else 0
            _LAST_LOCATION_FOR_MOVE = target

        result = (
            world.move_to(target, allow_bookstore_entry=allow_bookstore_entry)
            if target
            else "零想移动，但没有确定目的地。"
        )
        if _is_move_succeeded(result):
            if target:
                _BLOCKED_TARGET_ATTEMPTS[target] = 0
            protagonist_memory.add_memory(f"我决定移动。结果：{result}", importance=0.35)
            print(f"[零的行动] {result}")
            if target:
                if target not in visited_locations:
                    protagonist_memory.add_memory(f"我第一次到达了{target}。", importance=0.85)
                visited_locations.add(target)
                _MOVE_HISTORY.append(target)
                _MOVE_HISTORY = _MOVE_HISTORY[-4:]
                if len(_MOVE_HISTORY) == 4 and _MOVE_HISTORY[0] == _MOVE_HISTORY[2] and _MOVE_HISTORY[1] == _MOVE_HISTORY[3]:
                    world.advance_time(minutes=5)
                    protagonist_memory.add_memory("我发现自己来回折返太频繁，决定先停下来整理线索。", importance=0.5)
                    world.add_event("零停止来回折返，先在原地整理信息。")
                    print("[零的行动] 我不再来回折返，先停下来整理线索。")
                    return None, pending_identity_name
            if pending_commitment and target == pending_commitment:
                return None, pending_identity_name
            _LAST_ACTION = "move"
            return pending_commitment, pending_identity_name

        if "你已经在" in result:
            protagonist_memory.add_memory(f"我决定移动。结果：{result}", importance=0.35)
            print(f"[零的行动] {result}")
            _LAST_ACTION = "move"
            return pending_commitment, pending_identity_name

        if _is_move_failure_need_replan(result):
            if target:
                _BLOCKED_TARGET_ATTEMPTS[target] = _BLOCKED_TARGET_ATTEMPTS.get(target, 0) + 1
            world.advance_time(minutes=5)
            world.add_event(f"零在{world.current_location}附近短暂停留，等待时机。")
            print(f"[零的行动] {result}")
            pending_commitment, pending_identity_name = handle_action_failure_and_replan(
                world,
                protagonist_memory,
                npc_manager,
                known_locations,
                visited_locations,
                plan_state,
                failure_detail=f"移动失败：{result}" + (f"（目标地点：{target}）" if target else ""),
                pending_commitment=pending_commitment,
                pending_identity_name=pending_identity_name,
                failed_move_target=target if target else None,
                rag_store=rag_store,
            )
            return pending_commitment, pending_identity_name

        protagonist_memory.add_memory(f"我决定移动。结果：{result}", importance=0.35)
        print(f"[零的行动] {result}")
        _LAST_ACTION = "move"
        return pending_commitment, pending_identity_name

    if action == "talk":
        # 对话前再次按“当前地点”实时取人，避免跨场景错聊
        live_npcs_here = npc_manager.get_npcs_in_location(world.current_location)
        live_name_set = {n.name for n in live_npcs_here}
        npc = npc_manager.get_npc(target) if target else None
        if npc is None and live_npcs_here:
            npc = random.choice(live_npcs_here)
            target = npc.name
        if not npc or target not in live_name_set:
            detail = (
                f"对话失败：我想找的人不在当前地点（意图对象：{target or '未指定'}），"
                f"而我人在{world.current_location}。"
            )
            print(f"[零的行动] {detail}")
            return handle_action_failure_and_replan(
                world,
                protagonist_memory,
                npc_manager,
                known_locations,
                visited_locations,
                plan_state,
                failure_detail=detail,
                pending_commitment=pending_commitment,
                pending_identity_name=pending_identity_name,
                rag_store=rag_store,
            )

        if target == "李世民":
            li_talk_count = len(protagonist_memory.recall_by_keywords(["我对李世民说"], limit=30))
            if li_talk_count >= 2 and random.random() < 0.8:
                world.advance_time(minutes=5)
                protagonist_memory.add_memory("我感觉李世民不太想多聊，我先不打扰他。", importance=0.4)
                print("[零的行动] 我感觉李世民社交欲望很低，先不继续打扰。")
                return pending_commitment, pending_identity_name

        display_name = target if target in known_people else "陌生人"
        # 记忆里的前缀是 display_name（陌生人/真名），与 target 对齐才能计数
        recent_talks = protagonist_memory.recall_by_keywords(["我对"], limit=8)
        repeated_target = sum(1 for m in recent_talks if f"我对{display_name}说：" in m.content)
        if repeated_target >= 2 and random.random() < 0.88:
            thought = protagonist_think_and_reply(
                world,
                protagonist_memory,
                "你和眼前这个人已经连续聊了好几轮，不要再开口闲聊喷泉或车轱辘话。"
                "请只输出对周围环境的短观察（陈述句，不要问句），1-2句。",
                rag_store=rag_store,
            )
            world.advance_time(minutes=5)
            protagonist_memory.add_memory(f"我观察到：{thought}", importance=0.38)
            world.add_event("零停止连续搭话，改为观察周围。")
            print(f"[零的行动] 观察：{thought}")
            _LAST_ACTION = "observe"
            if world.current_location == "广场":
                known_locations.update(set(world.available_locations))
                protagonist_memory.add_memory("我看了广场路牌，知道了城里主要地点。", importance=0.8)
            return pending_commitment, pending_identity_name
        if pending_identity_name:
            attempts = _IDENTITY_QUERY_ATTEMPTS.get(pending_identity_name, 0)
            if attempts >= 2:
                protagonist_memory.add_memory(
                    f"关于“{pending_identity_name}”的身份我已确认过两次，先暂时保留疑问，不再反复追问。",
                    importance=0.75,
                )
                identity_shelved.add(pending_identity_name)
                _IDENTITY_QUERY_ATTEMPTS.pop(pending_identity_name, None)
                pending_identity_name = None
                say_text = protagonist_think_and_reply(
                    world,
                    protagonist_memory,
                    "你暂时放下身份追问，改为问一个新的具体问题。",
                    rag_store=rag_store,
                )
            else:
                _IDENTITY_QUERY_ATTEMPTS[pending_identity_name] = attempts + 1
                say_text = f"你刚才提到“{pending_identity_name}”，这个人是谁？"
                pending_identity_name = None
        else:
            recent_said = _recent_i_said_lines(protagonist_memory, 6)
            ban_line = "；".join(recent_said[:5]) if recent_said else "（暂无）"
            say_text = protagonist_think_and_reply(
                world,
                protagonist_memory,
                f"你准备和{display_name}对话。只输出一句给对方的话，不要场景描写/内心独白/移动叙述。\n"
                f"【禁复述】你最近说过的整句包括：{ban_line}\n"
                "本次必须换话题：可问吃饭、工作、附近店、下午安排、城里新闻；"
                "禁止再问喷泉、禁止说「重复」「没注意到」、禁止只改一两个字的同义复读。\n"
                "24 字以内，不要用引号包裹整句。",
                rag_store=rag_store,
            )
        say_text = _coerce_dialogue_line(say_text)
        last_norm = _normalize_for_repeat_check(_LAST_SPOKEN_LINE) if _LAST_SPOKEN_LINE else ""
        now_norm = _normalize_for_repeat_check(say_text)
        if last_norm and now_norm and now_norm == last_norm:
            say_text = protagonist_think_and_reply(
                world,
                protagonist_memory,
                "你刚才说过类似的话了。换完全不同的开头与话题，只输出一句短问话。",
                rag_store=rag_store,
            )
            say_text = _coerce_dialogue_line(say_text)
            now_norm = _normalize_for_repeat_check(say_text)
        rep_attempts = 0
        while _utterance_is_repetitive(say_text, _RECENT_UTTERANCE_NORMS) and rep_attempts < 2:
            rep_attempts += 1
            say_text = protagonist_think_and_reply(
                world,
                protagonist_memory,
                "你这句话仍和前几轮太像。必须换全新话题（吃饭/店铺/工作/路线），一句短问，禁止喷泉与重复类措辞。",
                rag_store=rag_store,
            )
            say_text = _coerce_dialogue_line(say_text)
        if _utterance_is_repetitive(say_text, _RECENT_UTTERANCE_NORMS):
            candidates = [
                f
                for f in _DIALOGUE_FALLBACKS
                if _normalize_for_repeat_check(f) not in _RECENT_UTTERANCE_NORMS
            ]
            say_text = random.choice(candidates if candidates else _DIALOGUE_FALLBACKS)
        _LAST_SPOKEN_LINE = say_text
        _RECENT_UTTERANCE_NORMS.append(_normalize_for_repeat_check(say_text))
        npc_reply = npc_manager.talk_to_npc(
            target,
            say_text,
            world.current_location,
            current_time_text=world.get_time_text(),
            current_weather=world.weather,
        )
        world.advance_time(minutes=8)
        protagonist_memory.add_memory(f"我对{display_name}说：{say_text}", importance=0.3)
        protagonist_memory.add_memory(f"{display_name}对我说：{npc_reply}", importance=0.3)
        world.add_event(f"零在{world.current_location}与{display_name}交谈。")
        scene_label = f"{display_name}(在{world.current_location})" if display_name == "陌生人" else display_name
        print(f"[零的行动] 零对{scene_label}：{say_text}")
        print(f"[NPC回应] {scene_label}：{npc_reply}")
        _LAST_ACTION = "talk"

        # 只有“明确询问姓名/身份”时，才允许解锁人名
        asked_identity = any(k in say_text for k in ["你是谁", "叫什么", "名字", "怎么称呼", "你是做什么", "这个人是谁"])
        if asked_identity and (target in npc_reply or f"我是{target}" in npc_reply or f"叫我{target}" in npc_reply):
            known_people.add(target)
            identity_shelved.add(target)
            _IDENTITY_QUERY_ATTEMPTS.pop(target, None)
        # 对“待确认名字”的身份收敛：出现“X是/叫X/在X...”即认为已得到可用答案
        for name in list(_IDENTITY_QUERY_ATTEMPTS.keys()):
            if name in npc_reply and any(k in npc_reply for k in [f"{name}是", f"叫{name}", f"去问{name}", f"在{name}"]):
                known_people.add(name)
                identity_shelved.add(name)
                _IDENTITY_QUERY_ATTEMPTS.pop(name, None)

        for fact in split_facts(npc_reply):
            score = fact_confidence.get(fact, 0.0)
            if score <= 0:
                fact_confidence[fact] = 0.5
                protagonist_memory.add_memory(f"待验证信息(可信度0.50)：{fact}", importance=0.7)
            else:
                new_score = min(1.0, score + 0.25)
                fact_confidence[fact] = new_score
                protagonist_memory.add_memory(f"信息再次被验证(可信度{new_score:.2f})：{fact}", importance=0.8)

        for name in npc_manager.list_npc_names():
            if (
                name in npc_reply
                and name not in known_people
                and name not in identity_shelved
                and should_trigger_identity_query(npc_reply, name)
            ):
                pending_identity_name = name
                protagonist_memory.add_memory(f"我听到陌生名字“{name}”，下轮要确认是谁。", importance=0.8)
                break
        mentioned = extract_location_mentions(npc_reply, set(world.available_locations))
        known_locations.update(mentioned)

        # 从对话内容抽取“双方约定去某地”
        for loc in world.available_locations:
            self_commit = any(k in say_text for k in [f"去{loc}", f"到{loc}", f"在{loc}见"])
            other_commit = any(k in npc_reply for k in [f"去{loc}", f"到{loc}", f"在{loc}见", f"一起去{loc}"])
            if self_commit and other_commit:
                if loc != world.current_location:
                    return loc, pending_identity_name
        return pending_commitment, pending_identity_name

    if action == "eat":
        if world.current_location != "餐厅":
            detail = f"进食失败：我想点餐吃饭，但当前在{world.current_location}，不是餐厅。"
            print(f"[零的行动] {detail}")
            world.advance_time(minutes=4)
            return handle_action_failure_and_replan(
                world,
                protagonist_memory,
                npc_manager,
                known_locations,
                visited_locations,
                plan_state,
                failure_detail=detail,
                pending_commitment=pending_commitment,
                pending_identity_name=pending_identity_name,
                rag_store=rag_store,
            )
        meal_text = protagonist_think_and_reply(
            world,
            protagonist_memory,
            "你在金姨小馆准备点单，请说出一句简短自然的点单内容。",
            rag_store=rag_store,
        )
        world.advance_time(minutes=12)
        protagonist_memory.add_memory(f"我在餐厅点单吃饭：{meal_text}", importance=0.6)
        world.add_event("零在餐厅点了餐并吃饭。")
        print(f"[零的行动] 点单吃饭：{meal_text}")
        _LAST_ACTION = "eat"
        return pending_commitment, pending_identity_name

    if action == "observe":
        thought = protagonist_think_and_reply(
            world,
            protagonist_memory,
            "你正在观察四周，请输出一条简短观察结论。",
            rag_store=rag_store,
        )
        world.advance_time(minutes=5)
        protagonist_memory.add_memory(f"我观察到：{thought}", importance=0.3)
        world.add_event("零停下脚步观察环境。")
        print(f"[零的行动] 观察：{thought}")
        _LAST_ACTION = "observe"
        # 在广场看到路牌后解锁地点认知
        if world.current_location == "广场":
            known_locations.update(set(world.available_locations))
            protagonist_memory.add_memory("我看了广场路牌，知道了城里主要地点。", importance=0.8)
        return pending_commitment, pending_identity_name

    if action == "inspect":
        local_items = world.location_objects.get(world.current_location, [])
        item_text = "、".join(local_items) if local_items else "没看到特别显眼的东西。"
        inspect_line = protagonist_think_and_reply(
            world,
            protagonist_memory,
            f"你正在仔细查看{world.current_location}的细节。可见事物：{item_text}。"
            "请输出1-2句有信息价值的观察，不要闲聊。",
            rag_store=rag_store,
        )
        world.advance_time(minutes=4)
        protagonist_memory.add_memory(f"我在{world.current_location}仔细查看：{inspect_line}", importance=0.45)
        world.add_event(f"零在{world.current_location}停下脚步仔细查看周围。")
        print(f"[零的行动] 细查：{inspect_line}")
        _LAST_ACTION = "inspect"
        if world.current_location == "广场" and any("路牌" in x for x in local_items):
            known_locations.update(set(world.available_locations))
            protagonist_memory.add_memory("我通过路牌再次确认了全城主要地点。", importance=0.75)
        return pending_commitment, pending_identity_name

    if action == "reflect":
        reflect_line = protagonist_think_and_reply(
            world,
            protagonist_memory,
            "你做一次短内省：总结当前最重要的1条线索，以及下一步最合理的方向。输出1-2句。",
            rag_store=rag_store,
        )
        world.advance_time(minutes=3)
        protagonist_memory.add_memory(f"我做了短内省：{reflect_line}", importance=0.5)
        world.add_event("零短暂停下，做了内心复盘。")
        print(f"[零的行动] 内省：{reflect_line}")
        _LAST_ACTION = "reflect"
        return pending_commitment, pending_identity_name

    if action == "review_plan":
        summary = f"当前目标：{plan_state.get('goal', '暂无')}；步骤数：{len(plan_state.get('steps') or [])}"
        protagonist_memory.add_memory(f"我复盘了计划：{summary}", importance=0.55)
        print(f"[零的行动] 计划复盘：{summary}")
        # 当计划为空/过短时主动重规划
        if not plan_state.get("steps") or len(plan_state.get("steps") or []) < 2:
            run_planner(
                world,
                protagonist_memory,
                npc_manager,
                known_locations,
                visited_locations,
                plan_state,
                failure_context="主动复盘后发现计划信息不足，重整阶段计划。",
                rag_store=rag_store,
            )
            print(f"[规划] 复盘后重整：{plan_state.get('goal', '')}")
        world.advance_time(minutes=2)
        _LAST_ACTION = "review_plan"
        return pending_commitment, pending_identity_name

    if action == "rest":
        rest_result = world.wait(hours=0.08)
        protagonist_memory.add_memory(f"我短暂休整了一下。结果：{rest_result}", importance=0.35)
        world.add_event("零短暂休整，稳定状态。")
        print(f"[零的行动] 短休：{rest_result}")
        _LAST_ACTION = "rest"
        return pending_commitment, pending_identity_name

    if action == "query_hours":
        q_target = world.resolve_location(target) if target else world.current_location
        if q_target not in world.available_locations:
            q_target = world.current_location
        hours_text = world.get_location_hours_text(q_target)
        open_text = "开放" if world.is_location_open(q_target) else "关闭"
        line = f"{q_target}营业时间：{hours_text}；当前状态：{open_text}。"
        protagonist_memory.add_memory(f"我查询了营业信息：{line}", importance=0.62)
        world.add_event(f"零查询了{q_target}的营业信息。")
        print(f"[零的行动] 查询营业时间：{line}")
        world.advance_time(minutes=2)
        _LAST_ACTION = "query_hours"
        return pending_commitment, pending_identity_name

    if action == "ask_npc_location":
        ask_name = target.strip() if target else ""
        if not ask_name:
            all_names = npc_manager.list_npc_names()
            ask_name = random.choice(all_names) if all_names else ""
        npc = npc_manager.get_npc(ask_name) if ask_name else None
        if npc is None:
            msg = f"我想查询人物位置，但没找到“{ask_name or '未知目标'}”。"
            protagonist_memory.add_memory(msg, importance=0.5)
            print(f"[零的行动] {msg}")
            world.advance_time(minutes=2)
            _LAST_ACTION = "ask_npc_location"
            return pending_commitment, pending_identity_name
        line = f"{npc.name}当前在{npc.location}。"
        protagonist_memory.add_memory(f"我查询了人物位置：{line}", importance=0.68)
        world.add_event(f"零确认了{npc.name}的当前位置。")
        print(f"[零的行动] 查询人物位置：{line}")
        world.advance_time(minutes=2)
        _LAST_ACTION = "ask_npc_location"
        return pending_commitment, pending_identity_name

    if action == "verify_fact":
        if not fact_confidence:
            msg = "目前没有待核验信息，我先继续收集线索。"
            protagonist_memory.add_memory(msg, importance=0.45)
            print(f"[零的行动] 核验信息：{msg}")
            world.advance_time(minutes=2)
            _LAST_ACTION = "verify_fact"
            return pending_commitment, pending_identity_name
        fact, score = min(fact_confidence.items(), key=lambda x: x[1])
        delta = 0.05
        # 轻量核验规则：地点在场/天气一致时提升更快
        if any(loc in fact for loc in world.available_locations) and world.current_location in fact:
            delta = 0.2
        if world.weather in fact:
            delta = max(delta, 0.15)
        new_score = min(1.0, score + delta)
        fact_confidence[fact] = new_score
        protagonist_memory.add_memory(
            f"我核验了一条信息：{fact}（可信度 {score:.2f} -> {new_score:.2f}）",
            importance=0.7 if new_score >= 0.75 else 0.55,
        )
        world.add_event("零进行了一次信息核验。")
        print(f"[零的行动] 核验信息：{fact}（可信度 {score:.2f} -> {new_score:.2f}）")
        world.advance_time(minutes=3)
        _LAST_ACTION = "verify_fact"
        return pending_commitment, pending_identity_name

    wait_result = world.wait(hours=0.25)
    protagonist_memory.add_memory(f"我选择等待。结果：{wait_result}", importance=0.3)
    print(f"[零的行动] {wait_result}")
    _LAST_ACTION = "wait"
    return pending_commitment, pending_identity_name


def apply_pre_turn_npc_updates(
    world: WorldState,
    npc_manager: NPCManager,
    protagonist_memory: MemorySystem,
) -> None:
    """根据世界时间更新 NPC 位置，并让零感知同场景的进出。"""
    for move_event in npc_manager.update_all_locations(world.current_time):
        world.add_event(move_event)
        match = re.match(r"(.+)从(.+)移动到(.+)。", move_event)
        if match:
            who, src, dst = match.group(1), match.group(2), match.group(3)
            if src == world.current_location:
                protagonist_memory.add_memory(f"我注意到{who}离开了这里。", importance=0.55)
            if dst == world.current_location:
                protagonist_memory.add_memory(f"我注意到{who}来到了这里。", importance=0.6)


def flush_event_prints(world: WorldState, limit: int = 5) -> None:
    for event in world.pop_events(limit=limit):
        print(f"[事件] {event}")


def pause_requested_nonblocking() -> bool:
    """自动批次运行中轮询暂停（不阻塞等待输入）。"""
    if sys.platform == "win32":
        try:
            import msvcrt  # noqa: PLC0415

            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"p", b"P", b" ", b"\x1b"):
                    return True
        except Exception:
            pass
        return False
    try:
        import select  # noqa: PLC0415

        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line and line.strip().lower() in ("暂停", "pause", "p"):
                return True
    except Exception:
        pass
    return False


def run_auto_batch(
    max_steps: int,
    sleep_seconds: float,
    world: WorldState,
    protagonist_memory: MemorySystem,
    npc_manager: NPCManager,
    known_people: Set[str],
    known_locations: Set[str],
    visited_locations: Set[str],
    fact_confidence: Dict[str, float],
    identity_shelved: Set[str],
    plan_state: Dict[str, Any],
    rag_store: Any,
    pending_commitment: Optional[str],
    pending_identity_name: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    print(
        f"[自动] 开始：最多 {max_steps} 步，间隔 {sleep_seconds}s；"
        "Windows 可按 p / 空格 / ESC 暂停本批。"
    )
    pc, pid = pending_commitment, pending_identity_name
    executed = 0
    for _ in range(max_steps):
        if pause_requested_nonblocking():
            print(f"[自动] 已暂停（本批已执行 {executed} 步）")
            break
        apply_pre_turn_npc_updates(world, npc_manager, protagonist_memory)
        flush_event_prints(world, limit=5)
        pc, pid = run_autonomous_turn(
            world,
            protagonist_memory,
            npc_manager,
            known_people,
            known_locations,
            visited_locations,
            pc,
            pid,
            fact_confidence,
            identity_shelved,
            plan_state,
            rag_store=rag_store,
        )
        executed += 1
        time.sleep(max(0.0, sleep_seconds))
    print(f"[自动] 本批结束，本批共执行 {executed} 步。")
    return pc, pid


def hard_exit_without_memory(protagonist_memory: MemorySystem, npc_manager: NPCManager) -> None:
    """
    硬退出：清空本次运行内存对象，并直接终止进程。
    """
    protagonist_memory.memories.clear()
    for npc in npc_manager.npcs.values():
        npc.memory.memories.clear()
    print("正在强制退出：已清空当前运行记忆并终止进程。", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="封闭文本世界 AI（观察者模式）")
    parser.add_argument(
        "--auto",
        type=int,
        default=0,
        metavar="N",
        help="启动后先自动执行 N 个世界回合再进入交互循环",
    )
    parser.add_argument(
        "--auto-sleep",
        type=float,
        default=0.5,
        help="自动回合之间的间隔秒数",
    )
    args = parser.parse_args()

    world = WorldState()
    npc_manager = NPCManager()
    loaded_count = npc_manager.load_from_directory("npcs")
    for move_event in npc_manager.update_all_locations(world.current_time):
        world.add_event(move_event)

    protagonist_memory = MemorySystem(owner_name="零")
    protagonist_memory.add_permanent_memory("我是人类")
    protagonist_memory.add_memory("我性格大大咧咧，喜欢主动和人聊天。", importance=0.9)
    protagonist_memory.add_memory("我刚来到这座城市，对这里的人和他们的身份都不了解。", importance=0.98)
    protagonist_memory.add_memory("我白天喜欢到处逛逛，顺便认识新朋友。", importance=0.9)
    protagonist_memory.add_memory("我获取知识的主要方式是和别人对话。", importance=0.95)
    protagonist_memory.add_memory("我会主动去找不同的人了解这个世界。", importance=0.9)
    protagonist_memory.add_memory("我家在广场旁边的一栋楼里。", importance=0.9)
    protagonist_memory.add_memory("我每天常从家里出门，先到广场。", importance=0.75)
    protagonist_memory.add_memory("我刚刚在这个世界醒来。", importance=0.6)
    protagonist_memory.add_memory(f"我所在地点是{world.current_location}。", importance=0.5)
    known_people: Set[str] = set()
    known_locations: Set[str] = {"广场"}
    visited_locations: Set[str] = {"广场"}
    # 零一开始只知道城里“有警察”，但不知道名字
    known_people.add("警察")
    pending_commitment: Optional[str] = None
    pending_identity_name: Optional[str] = None
    fact_confidence: Dict[str, float] = {}
    identity_shelved: Set[str] = set()
    plan_state: Dict[str, Any] = {"goal": "", "steps": [], "history": deque(maxlen=8)}

    run_session_id = f"run_{int(time.time())}"
    rag_store = create_protagonist_rag(session_id=run_session_id)
    if rag_store:
        print(
            "[RAG] LanceDB 向量记忆已启用（WORLD_RAG=0 可关闭；嵌入模型见 OLLAMA_EMBED_MODEL，"
            "需 ollama pull nomic-embed-text 等）"
        )
        print(f"[RAG] 本次会话隔离ID：{run_session_id}（仅检索本次运行记忆）")
    else:
        print("[RAG] 未启用（未安装 lancedb 或已设置 WORLD_RAG=0）")

    print("=== 封闭文本世界 AI 实验（观察者模式）===")
    print(f"已加载NPC数量：{loaded_count}")
    if check_ollama_status():
        print(f"[LLM] 已连接 Ollama，当前模型：{OLLAMA_MODEL}")
    else:
        print(f"[LLM] 未连接或模型不可用，将使用回退文案。当前模型配置：{OLLAMA_MODEL}")
    print(world.get_world_description())
    print_help()

    print("提示：你不能控制零或NPC，只能观察和添加世界内容。")

    if args.auto > 0:
        pending_commitment, pending_identity_name = run_auto_batch(
            args.auto,
            args.auto_sleep,
            world,
            protagonist_memory,
            npc_manager,
            known_people,
            known_locations,
            visited_locations,
            fact_confidence,
            identity_shelved,
            plan_state,
            rag_store,
            pending_commitment,
            pending_identity_name,
        )

    while True:
        # 零设定为完整记忆，不执行遗忘清理
        apply_pre_turn_npc_updates(world, npc_manager, protagonist_memory)
        flush_event_prints(world, limit=5)

        user_input = input("观察者>>> ").strip()
        if not user_input:
            user_input = "继续"

        if user_input == "退出":
            hard_exit_without_memory(protagonist_memory, npc_manager)

        if user_input == "帮助":
            print_help()
            continue

        if user_input == "查看记忆":
            print("\n--- 零的记忆摘要 ---")
            print(protagonist_memory.get_summary(limit=15))
            print("--------------------\n")
            continue

        if user_input == "查看状态":
            print("\n--- 世界状态 ---")
            print(world.get_world_description())
            print("--------------\n")
            continue

        if user_input.startswith("添加事实 "):
            content = user_input[len("添加事实 ") :].strip()
            result = world.add_world_thing(content, "")
            print(result)
            continue

        if user_input.startswith("添加事物 "):
            payload = user_input[len("添加事物 ") :].strip()
            parts = payload.split(" ", 1)
            if len(parts) != 2:
                print("格式错误：添加事物 <地点> <内容>")
                continue
            result = world.add_world_thing(parts[1], parts[0])
            print(result)
            continue

        if user_input == "自动" or user_input.startswith("自动 "):
            parts = user_input.split()
            n_steps = 50
            sleep_s = 0.5
            if len(parts) >= 2:
                try:
                    n_steps = int(parts[1])
                except ValueError:
                    print("[自动] 步数无效，使用默认 50")
            if len(parts) >= 3:
                try:
                    sleep_s = float(parts[2])
                except ValueError:
                    print("[自动] 间隔秒无效，使用默认 0.5")
            pending_commitment, pending_identity_name = run_auto_batch(
                n_steps,
                sleep_s,
                world,
                protagonist_memory,
                npc_manager,
                known_people,
                known_locations,
                visited_locations,
                fact_confidence,
                identity_shelved,
                plan_state,
                rag_store,
                pending_commitment,
                pending_identity_name,
            )
            continue

        if user_input != "继续":
            print("未知命令，输入“帮助”查看可用命令。")
            continue

        pending_commitment, pending_identity_name = run_autonomous_turn(
            world,
            protagonist_memory,
            npc_manager,
            known_people,
            known_locations,
            visited_locations,
            pending_commitment,
            pending_identity_name,
            fact_confidence,
            identity_shelved,
            plan_state,
            rag_store=rag_store,
        )


if __name__ == "__main__":
    main()
