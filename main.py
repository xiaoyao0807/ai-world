"""
主游戏入口（观察者模式）
零与NPC自主行动，用户仅观察并可添加世界事物。
"""

import json
import os
import random
import re
import sys
from typing import Dict, Optional, Set, Tuple

import requests

from memory_system import MemorySystem
from npc_manager import NPCManager
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
PROTAGONIST_PROFILE = (
    "你完全不知道自己是一个AI智能体，你坚信自己是真实的人类。"
    "你是一个有好奇心、有情感、会思考的普通人。"
    "你会对新鲜事物感到惊讶和好奇。"
    "你会对重复的事情感到无聊。"
    "你会记得重要的经历，也会忘记琐碎的事情。"
    "你有时会回忆过去，有时会计划未来。"
    "你的情绪会影响你的决定。"
)


def build_protagonist_prompt(world: WorldState, memory: MemorySystem, user_input: str) -> str:
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
        "=== 你最近说过的话（临时文本）===\n"
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
            "temperature": 0.95,
            "top_p": 0.95,
            "repeat_penalty": 1.08,
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
    print("  查看记忆              查看零的记忆摘要")
    print("  查看状态              查看世界状态")
    print("  添加事实 <内容>       向世界添加全局事实")
    print("  添加事物 <地点> <内容> 在某地点添加可观察事物")
    print("  帮助                  显示本帮助")
    print("  退出                  结束程序\n")


def protagonist_think_and_reply(world: WorldState, protagonist_memory: MemorySystem, user_input: str) -> str:
    """
    零的思考流程：
    1) 构造prompt
    2) 调用LLM（失败则回退）
    3) 记录回应记忆
    """
    prompt = build_protagonist_prompt(world, protagonist_memory, user_input)
    reply = call_ollama_or_fallback(prompt)
    protagonist_memory.add_memory(f"我的内部思考产出：{reply}", importance=0.3)
    return reply


def extract_location_mentions(text: str, locations: Set[str]) -> Set[str]:
    return {loc for loc in locations if loc in (text or "")}


def split_facts(text: str) -> list:
    parts = re.split(r"[。！？!?；;]\s*", text or "")
    return [p.strip() for p in parts if p.strip()]


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
) -> Dict[str, str]:
    npcs_here = npc_manager.get_npcs_in_location(world.current_location)
    locations = "、".join(sorted(known_locations))
    npcs_here_count = len(npcs_here)
    if pending_commitment and pending_commitment != world.current_location:
        return {"action": "move", "target": pending_commitment}
    if pending_identity_name:
        return {"action": "talk", "target": ""}
    prompt = (
        f"{PROTAGONIST_PROFILE}"
        "你需要独立决定下一步行动。"
        "请只输出JSON，不要解释。\n"
        "你需要先在心里执行：回忆 -> 感受 -> 推理 -> 决定，然后再输出结果。\n"
        "可选action: move, talk, wait, observe, eat\n"
        "JSON格式示例：{\"action\":\"move\",\"target\":\"书店\",\"thought\":\"我先去书店碰碰运气\"}\n"
        "如果action=talk，target可为空（表示随便找附近一位陌生人聊天）；"
        "如果action=move，target必须是你已知地点之一；"
        "如果action=eat，只能在餐厅使用，target可为空；"
        "如果action=wait/observe，target可为空。\n\n"
        "偏好：像正常人行动，不要频繁来回移动；信息不足时先问人或观察再决策。\n"
        "白天多探索和聊天，除非遇到求助或案件线索，否则少去警察局。\n\n"
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
        if action in {"move", "talk", "wait", "observe", "eat"}:
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
    if world.current_location == "餐厅" and random.random() < 0.5:
        _LAST_DECISION_THOUGHT = "先吃点东西，补充体力。"
        return {"action": "eat", "target": ""}
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
) -> Tuple[Optional[str], Optional[str]]:
    global _LAST_LOCATION_FOR_MOVE, _MOVE_BOUNCE_COUNT, _IDENTITY_QUERY_ATTEMPTS, _BLOCKED_TARGET_ATTEMPTS, _LAST_INFO_SOURCE, _LAST_DECISION_THOUGHT, _LAST_SPOKEN_LINE, _MOVE_HISTORY
    decision = decide_next_action(
        world,
        protagonist_memory,
        npc_manager,
        known_locations,
        visited_locations,
        pending_commitment,
        pending_identity_name,
    )
    action = decision["action"]
    target = decision["target"]
    npcs_here = npc_manager.get_npcs_in_location(world.current_location)
    print(f"[零的决策] action={action}, target={target or '无'}")
    if _LAST_DECISION_THOUGHT:
        print(f"[零的思考] {_LAST_DECISION_THOUGHT}")

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
        if "暂时无法进入" in result or "现在没人值守" in result:
            if target:
                _BLOCKED_TARGET_ATTEMPTS[target] = _BLOCKED_TARGET_ATTEMPTS.get(target, 0) + 1
            # 目标暂未开放时推进一点时间，避免原地无限重复同一决策
            world.advance_time(minutes=5)
            world.add_event(f"零在{world.current_location}附近短暂停留，等待时机。")
            # 连续失败则暂时放下该目标，转去收集其他线索，避免重复循环
            if target and _BLOCKED_TARGET_ATTEMPTS.get(target, 0) >= 3:
                protagonist_memory.add_memory(
                    f"{target}连续无法进入，我先放下这个目标，去找其他线索。",
                    importance=0.6,
                )
                print(f"[零的行动] {target}暂时进不去，我先换个方向找线索。")
                return None, pending_identity_name
        elif target:
            _BLOCKED_TARGET_ATTEMPTS[target] = 0
        protagonist_memory.add_memory(f"我决定移动。结果：{result}", importance=0.35)
        print(f"[零的行动] {result}")
        if "移动到" in result and target:
            if target not in visited_locations:
                protagonist_memory.add_memory(f"我第一次到达了{target}。", importance=0.85)
            visited_locations.add(target)
            _MOVE_HISTORY.append(target)
            _MOVE_HISTORY = _MOVE_HISTORY[-4:]
            # 防ABAB左右横跳
            if len(_MOVE_HISTORY) == 4 and _MOVE_HISTORY[0] == _MOVE_HISTORY[2] and _MOVE_HISTORY[1] == _MOVE_HISTORY[3]:
                world.advance_time(minutes=5)
                protagonist_memory.add_memory("我发现自己来回折返太频繁，决定先停下来整理线索。", importance=0.5)
                world.add_event("零停止来回折返，先在原地整理信息。")
                print("[零的行动] 我不再来回折返，先停下来整理线索。")
                return None, pending_identity_name
        if pending_commitment and target == pending_commitment and "移动到" in result:
            return None, pending_identity_name
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
            msg = "零想对话，但目标NPC不在当前地点。"
            protagonist_memory.add_memory(msg, importance=0.3)
            print(f"[零的行动] {msg}")
            return pending_commitment, pending_identity_name

        if target == "李世民":
            li_talk_count = len(protagonist_memory.recall_by_keywords(["我对李世民说"], limit=30))
            if li_talk_count >= 2 and random.random() < 0.8:
                world.advance_time(minutes=5)
                protagonist_memory.add_memory("我感觉李世民不太想多聊，我先不打扰他。", importance=0.4)
                print("[零的行动] 我感觉李世民社交欲望很低，先不继续打扰。")
                return pending_commitment, pending_identity_name

        # 防止连续多轮只和同一NPC重复聊天
        recent_talks = protagonist_memory.recall_by_keywords(["我对"], limit=4)
        repeated_target = sum(1 for m in recent_talks if f"我对{target}说：" in m.content)
        if repeated_target >= 3 and random.random() < 0.75:
            alt_move = world.wait(hours=0.1)
            protagonist_memory.add_memory(
                f"我发现自己一直在和{target}重复聊天，所以先暂停一下。结果：{alt_move}",
                importance=0.35,
            )
            print(f"[零的行动] 我决定先不重复找{target}聊天，短暂停留后再行动。")
            return pending_commitment, pending_identity_name

        display_name = target if target in known_people else "陌生人"
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
                )
            else:
                _IDENTITY_QUERY_ATTEMPTS[pending_identity_name] = attempts + 1
                say_text = f"你刚才提到“{pending_identity_name}”，这个人是谁？"
                pending_identity_name = None
        else:
            say_text = protagonist_think_and_reply(
                world,
                protagonist_memory,
                f"你准备和{display_name}对话。请避免和前几轮重复，给出一句有新信息价值的话。",
            )
        if _LAST_SPOKEN_LINE and say_text.strip() == _LAST_SPOKEN_LINE.strip():
            say_text = protagonist_think_and_reply(
                world,
                protagonist_memory,
                "你刚才说过类似的话，请换一种更自然的新表达，不要重复。",
            )
        _LAST_SPOKEN_LINE = say_text
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
            msg = "这里不是餐厅，暂时没法点单吃饭。"
            protagonist_memory.add_memory(msg, importance=0.35)
            print(f"[零的行动] {msg}")
            world.advance_time(minutes=4)
            return pending_commitment, pending_identity_name
        meal_text = protagonist_think_and_reply(
            world,
            protagonist_memory,
            "你在金姨小馆准备点单，请说出一句简短自然的点单内容。",
        )
        world.advance_time(minutes=12)
        protagonist_memory.add_memory(f"我在餐厅点单吃饭：{meal_text}", importance=0.6)
        world.add_event("零在餐厅点了餐并吃饭。")
        print(f"[零的行动] 点单吃饭：{meal_text}")
        return pending_commitment, pending_identity_name

    if action == "observe":
        thought = protagonist_think_and_reply(
            world,
            protagonist_memory,
            "你正在观察四周，请输出一条简短观察结论。",
        )
        world.advance_time(minutes=5)
        protagonist_memory.add_memory(f"我观察到：{thought}", importance=0.3)
        world.add_event("零停下脚步观察环境。")
        print(f"[零的行动] 观察：{thought}")
        # 在广场看到路牌后解锁地点认知
        if world.current_location == "广场":
            known_locations.update(set(world.available_locations))
            protagonist_memory.add_memory("我看了广场路牌，知道了城里主要地点。", importance=0.8)
        return pending_commitment, pending_identity_name

    wait_result = world.wait(hours=0.25)
    protagonist_memory.add_memory(f"我选择等待。结果：{wait_result}", importance=0.3)
    print(f"[零的行动] {wait_result}")
    return pending_commitment, pending_identity_name


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

    print("=== 封闭文本世界 AI 实验（观察者模式）===")
    print(f"已加载NPC数量：{loaded_count}")
    if check_ollama_status():
        print(f"[LLM] 已连接 Ollama，当前模型：{OLLAMA_MODEL}")
    else:
        print(f"[LLM] 未连接或模型不可用，将使用回退文案。当前模型配置：{OLLAMA_MODEL}")
    print(world.get_world_description())
    print_help()

    print("提示：你不能控制零或NPC，只能观察和添加世界内容。")

    while True:
        # 零设定为完整记忆，不执行遗忘清理

        # 根据当前时间推进NPC作息位置
        for move_event in npc_manager.update_all_locations(world.current_time):
            world.add_event(move_event)
            # 当零所在场景有人进入/离开时，零会第一时间注意到
            match = re.match(r"(.+)从(.+)移动到(.+)。", move_event)
            if match:
                who, src, dst = match.group(1), match.group(2), match.group(3)
                if src == world.current_location:
                    protagonist_memory.add_memory(f"我注意到{who}离开了这里。", importance=0.55)
                if dst == world.current_location:
                    protagonist_memory.add_memory(f"我注意到{who}来到了这里。", importance=0.6)

        # 展示近期世界事件
        pending_events = world.pop_events(limit=5)
        for event in pending_events:
            print(f"[事件] {event}")

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
        )


if __name__ == "__main__":
    main()
