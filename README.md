🌍 楚门ai的世界 - 可扩展AI生命模拟框架

# Closed Text World AI (Observer Mode)

一个受《楚门的世界》启发的AI实验项目。AI智能体拥有记忆、遗忘机制和独立NPC系统。
**你只能观察，不能操控。**  

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

---

## Tech Stack

- Python 3.9+
- `requests`
- Local Ollama API (`http://localhost:11434`)

安装依赖：

```bash
pip install requests

---

## Project Structure

```text
world/
├─ main.py
├─ memory_system.py
├─ world_state.py
├─ npc_manager.py
├─ README.md
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
- `查看记忆`：查看零的记忆摘要
- `查看状态`：查看当前世界状态
- `添加事实 <内容>`：添加全局事实
- `添加事物 <地点> <内容>`：向地点添加可观察物
- `帮助`
- `退出`

---

## Ollama (Optional but Recommended)

推荐模型（平衡效果与速度）：

- `qwen3.5:4b`
- `qwen3.5:latest`

启动模型示例：

```bash
ollama run qwen3.5:4b
```

可选环境变量：

```bash
# Windows PowerShell
$env:OLLAMA_API_URL="http://localhost:11434/api/generate"
$env:OLLAMA_MODEL="qwen3.5:4b"
```
若 Ollama 未启动，程序会自动使用回退文案继续运行。
---

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
