# Closed Text World AI (Observer Mode)

一个纯文本 AI 小世界实验：你只能观察，不能操控。  
主角「零」会自主思考、行动、对话、学习；NPC 也各自有独立记忆与人格。

---

## Features

- **Observer-only**: 用户只负责观察和添加世界信息，不直接控制角色行为。
- **Autonomous agent**: 零每回合自主选择 `move / talk / observe / wait / eat` 等行为。
- **Memory model**: 基于艾宾浩斯遗忘曲线 `strength = importance * exp(-0.1 * hours_passed)`。
- **Memory reinforcement**: 被回忆的记忆会刷新并轻微增强。
- **Permanent memory**: `importance=999` 的记忆永不衰减（例如「我是人类」）。
- **Independent NPC memory**: 每个 NPC 有自己的记忆系统，互不共享。
- **LLM-driven dialogue**: 对话由人格 + 场景 + 记忆动态生成，不是固定模板。
- **Ollama fallback**: 本地模型不可用时自动回退，程序不中断。
- **Anti-loop behavior**: 包含跨场景对话拦截与“左右横跳”防抖逻辑。

---

## Tech Stack

- Python 3.9+
- `requests`
- Local Ollama API (`http://localhost:11434`)

安装依赖：

```bash
pip install requests
```

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

## Known Notes

- 当前模式是观察者模式，不提供“你来控制角色移动/说话”的交互。
- 长时间运行时，模型输出质量与本地模型性能、显存、上下文长度密切相关。
- 若出现 LLM 超时，可先确认 `ollama run <model>` 已在单独终端保持运行。

---

## Upload to GitHub (Step by Step)

下面是最常用、最稳的上传流程（在项目根目录执行）：

1. 在 GitHub 网页新建一个空仓库（例如 `world-ai-observer`），**不要**勾选自动初始化 README。  
2. 本地执行：

```bash
git init
git add .
git commit -m "Initial commit: autonomous text world with observer mode"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<你的仓库名>.git
git push -u origin main
```

如果你使用 SSH：

```bash
git remote add origin git@github.com:<你的用户名>/<你的仓库名>.git
git push -u origin main
```

后续更新只需：

```bash
git add .
git commit -m "your message"
git push
```

---

## License

你可以按需要添加 `MIT` 或 `Apache-2.0` 许可证文件后再公开发布。
