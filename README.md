🌍 楚门ai的世界 - 可扩展AI生命模拟框架

# Closed Text World AI (Observer Mode)

一个受《楚门的世界》启发的AI实验项目。AI智能体拥有记忆、遗忘机制和独立NPC系统。
**你可以观察他的思想和行为，但是不能操控。**  

主角「零」是一个基于 qwen3.5:4b 的纯血AI，但是他只认为自己是一位会自主思考、行动、对话、学习的人类；其他角色也各自有独立记忆与人格，他们只会根据自己的性格和记忆做出动作和回答，没有固定的回复框架！

---

可以为这个世界加入各种各样的角色，事件，场景。


| 特性 | 说明 |
|------|------|
| 👁️ **Observer-only** | 用户只负责观察和添加世界信息，不直接控制角色行为 |
| 🤖 **Autonomous agent** | 零每回合自主选择 move / talk / observe / wait / eat 等行为 |
| 🧠 **Memory model** | 基于艾宾浩斯遗忘曲线 `strength = importance * exp(-0.1 * hours_passed)` |
| 🔄 **Memory reinforcement** | 被回忆的记忆会刷新并轻微增强 |
| 🔒 **Permanent memory** | `importance=999` 的记忆永不衰减（例如「我是人类」） |
| 👥 **Independent NPC memory** | 每个 NPC 有自己的记忆系统，互不共享 |
| 💬 **LLM-driven dialogue** | 对话由人格 + 场景 + 记忆动态生成，不是固定模板 |
| 🔧 **Ollama fallback** | 本地模型不可用时自动回退，程序不中断 |
| 🛡️ **Anti-loop behavior** | 包含跨场景对话拦截与“左右横跳”防抖逻辑 |

| 新增特性 | 说明 |
|---------|------|
| 🧠 **RAG记忆系统** | 基于LanceDB的向量检索，长期记忆增强 |
| 📋 **Planner规划** | LLM生成结构化计划，支持失败重规划 |
| 🎯 **扩展行为** | inspect/reflect/review_plan/rest等新动作 |
| 🔄 **会话隔离** | 每次运行独立ID，防止历史污染 |
| 🤖 **自动批处理** | `--auto N` 连续运行，支持暂停 |
---

## Tech Stack

- Python 3.9+
- `requests`
- `lancedb`（可选，用于 RAG）
- Ollama 本地 API（`http://localhost:11434`）

安装依赖：

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
world/
├─ main.py
├─ memory_system.py
├─ world_state.py
├─ npc_manager.py
├─ rag_memory.py
├─ requirements.txt
├─ README.md
├─ UPDATE_LOG.md
├─ RECORDING_GUIDE.md
├─ run.bat
├─ run.sh
└─ npcs/
   ├─ bookstore_owner.json
   ├─ flower_shop_owner.json
   ├─ police_operator_yang_xiaoxiao.json
   ├─ restaurant_owner_jin_dahua.json
   └─ passerby_li_shimin.json
```
---

## Run

### Windows

```bat
run.bat
```

或：

```bat
py -3 main.py
```

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

---

## Observer Commands

- `继续`：推进 1 个自治回合（空输入也可继续）
- `自动 [N] [间隔秒]`：连续自动推进（默认 `N=50`，间隔 `0.5s`）
- `查看记忆`：查看零的记忆摘要
- `查看状态`：查看当前世界状态
- `添加事实 <内容>`：添加全局事实
- `添加事物 <地点> <内容>`：向地点添加可观察物
- `帮助`
- `退出`

---
## Action Set

当前可执行动作（由 AI 自主选择）：

- `move`
- `talk`
- `wait`
- `observe`
- `eat`
- `inspect`
- `reflect`
- `review_plan`
- `rest`
- `query_hours`
- `ask_npc_location`
- `verify_fact`

---

## Planner and Replanning

- `plan_state` 维护当前目标与步骤。
- 每回合根据当前计划与环境决策动作。
- 当关键行动失败（如目标不在、地点不可达）时：
  - 记录高重要性失败记忆
  - 清空旧计划
  - 立刻触发 Planner 重新生成新计划

---

## RAG Memory (LanceDB + Ollama Embedding)

默认启用（可关闭）：

- 向量库：LanceDB
- 嵌入：Ollama embedding 模型（默认 `nomic-embed-text`）

先拉取嵌入模型：

```bash
ollama pull nomic-embed-text
```

### 环境变量

```bash
# PowerShell
$env:OLLAMA_API_URL="http://localhost:11434/api/generate"
$env:OLLAMA_MODEL="qwen3.5:4b"
$env:OLLAMA_EMBED_MODEL="nomic-embed-text"
$env:WORLD_RAG="1"
$env:WORLD_RAG_PATH=".world_data/lance_zero"
```

说明：

- `WORLD_RAG=0` 可关闭 RAG。
- 程序会优先调用 `/api/embed`，并自动兼容旧 `/api/embeddings`。
- 若嵌入不可用，会自动停用本局向量同步，避免持续刷错。

### 会话隔离

每次运行会生成独立 `session_id`，RAG 表名形如：

- `zero_memories_<session_id>`

因此新一局不会自动检索上一局的日志/记忆。

## 🎨 如何为这个世界添加内容

### 添加新角色

在 `npcs/` 目录下创建 JSON 文件，格式如下：

```json
{
  "name": "角色名",
  "role": "角色定位",
  "location": "初始位置",
  "personality": "性格描述",
  "relationship": "与主角的关系",
  "dialogue_topics": ["话题1", "话题2"],
  "initial_memories": ["初始记忆1", "初始记忆2"]
}

示例 - 新角色：
{
  "name": "流浪诗人艾米莉",
  "role": "流浪诗人",
  "location": "城镇广场",
  "personality": "浪漫、感性、喜欢即兴创作诗歌",
  "dialogue_topics": ["诗歌", "旅行见闻", "爱情故事"],
  "initial_memories": ["我走遍了整个大陆", "每个地方都有自己的故事"]
}
添加新地点
编辑 world_state.py 中的 locations 字典。

添加随机事件
编辑 world_state.py 中的随机事件列表。

🤝 如何贡献
Fork 本仓库

创建你的分支 (git checkout -b feature/AmazingNPC)

提交修改 (git commit -m '添加某个NPC')

推送到分支 (git push origin feature/AmazingNPC)

打开 Pull Request
## Known Notes

- 当前模式是观察者模式，不提供“你来控制角色移动/说话”的交互。
- 长时间运行时，模型输出质量与本地模型性能、显存、上下文长度密切相关。
- 若出现 LLM 超时，可先确认 `ollama run <model>` 已在单独终端保持运行。

---
