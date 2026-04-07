"""
艾宾浩斯记忆系统模块
实现了基于时间衰减的记忆存储和召回机制
核心公式：strength = importance * exp(-0.1 * hours_passed)
"""

import json
import math
from datetime import datetime
from typing import List, Dict


class MemoryItem:
    """
    单条记忆项类
    包含记忆的内容、重要性、创建时间和最后访问时间
    """

    def __init__(self, content: str, importance: float = 0.5, memory_id: str = None):
        """
        初始化一条记忆

        参数:
            content: 记忆内容文本
            importance: 重要性(0-1)，永久记忆设置为999
            memory_id: 记忆唯一标识符，不提供则自动生成
        """
        self.content = content
        self.importance = importance
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.id = memory_id or f"mem_{self.created_at.timestamp()}"

    def get_current_strength(self) -> float:
        """
        计算当前记忆强度

        使用艾宾浩斯遗忘曲线公式：strength = importance * exp(-0.1 * hours_passed)
        hours_passed 是从记忆创建到现在经过的小时数

        返回:
            当前记忆强度值
        """
        if self.is_permanent():
            # 永久记忆不随时间衰减
            return self.importance

        # 使用“最后访问时间”作为衰减基准，访问后可重新变清晰
        hours_passed = (datetime.now() - self.last_accessed).total_seconds() / 3600.0
        strength = self.importance * math.exp(-0.1 * hours_passed)
        return max(0.0, min(strength, self.importance))  # 限制在[0, importance]范围内

    def refresh(self):
        """
        刷新记忆（访问记忆时调用）
        更新最后访问时间，相当于"复习"记忆
        """
        self.last_accessed = datetime.now()

        # 非永久记忆在被访问时略微增强，模拟“复习”
        if not self.is_permanent():
            self.importance = min(1.0, self.importance + 0.05)

    def is_permanent(self) -> bool:
        """
        判断是否为永久记忆

        返回:
            True如果是永久记忆(importance>=999)
        """
        return self.importance >= 999

    def to_dict(self) -> dict:
        """
        将记忆项转换为字典，用于序列化保存

        返回:
            包含所有字段的字典
        """
        return {
            'id': self.id,
            'content': self.content,
            'importance': self.importance,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        从字典恢复记忆项对象

        参数:
            data: 包含记忆数据的字典

        返回:
            MemoryItem对象
        """
        memory = cls(data['content'], data['importance'], data['id'])
        memory.created_at = datetime.fromisoformat(data['created_at'])
        memory.last_accessed = datetime.fromisoformat(data['last_accessed'])
        return memory

    def __str__(self) -> str:
        strength = self.get_current_strength()
        perm_mark = "🔒 " if self.is_permanent() else ""
        return f"{perm_mark}[强度:{strength:.2f}] {self.content}"


class MemorySystem:
    """
    记忆系统类
    管理所有记忆项，提供添加、召回、刷新、清理、摘要等核心功能
    """

    def __init__(self, owner_name: str = "Unknown"):
        """
        初始化记忆系统

        参数:
            owner_name: 记忆系统的拥有者名称（主角或NPC）
        """
        self.owner_name = owner_name
        self.memories: Dict[str, MemoryItem] = {}  # 使用字典存储，id -> MemoryItem
        self.permanent_memory_added = False

    def add_memory(self, content: str, importance: float = 0.5) -> str:
        """
        添加新记忆到系统

        参数:
            content: 记忆内容
            importance: 重要性(0-1)，永久记忆用999

        返回:
            新记忆的ID
        """
        # 避免添加重复的空内容
        if not content or len(content.strip()) == 0:
            return ""

        # 检查是否已存在完全相同的记忆（避免重复）
        for mem in self.memories.values():
            if mem.content == content and mem.importance == importance:
                mem.refresh()  # 刷新已存在的记忆
                return mem.id

        memory = MemoryItem(content, importance)
        self.memories[memory.id] = memory
        return memory.id

    def add_permanent_memory(self, content: str) -> str:
        """
        添加永久记忆（importance=999，永不衰减）
        用于主角的"我是人类"等核心设定

        参数:
            content: 永久记忆内容

        返回:
            新记忆的ID
        """
        return self.add_memory(content, importance=999)

    def recall(self, query: str = None, limit: int = 10) -> List[MemoryItem]:
        """
        召回记忆，按当前强度降序排序

        参数:
            query: 可选的查询文本，用于关键词过滤
            limit: 最多返回的记忆数量

        返回:
            记忆项列表，按强度从高到低排序
        """
        if not self.memories:
            return []

        # 计算每条记忆的当前强度
        memory_list = []
        for memory in self.memories.values():
            strength = memory.get_current_strength()

            # 如果有查询关键词，进行简单的匹配过滤
            if query:
                if query.lower() in memory.content.lower():
                    memory_list.append((memory, strength))
            else:
                memory_list.append((memory, strength))

        # 按强度降序排序
        memory_list.sort(key=lambda x: x[1], reverse=True)

        # 返回前limit条记忆，并刷新这些记忆（访问强化）
        recalled = [mem for mem, _ in memory_list[:limit]]
        for mem in recalled:
            mem.refresh()

        return recalled

    def recall_by_keywords(self, keywords: List[str], limit: int = 5) -> List[MemoryItem]:
        """
        根据关键词列表召回相关记忆

        参数:
            keywords: 关键词列表
            limit: 最多返回数量

        返回:
            匹配的记忆项列表
        """
        if not keywords:
            return self.recall(limit=limit)

        matched = []
        for memory in self.memories.values():
            content_lower = memory.content.lower()
            # 检查是否包含任意关键词
            if any(keyword.lower() in content_lower for keyword in keywords):
                strength = memory.get_current_strength()
                matched.append((memory, strength))
                memory.refresh()

        # 按强度排序
        matched.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in matched[:limit]]

    def get_summary(self, limit: int = 15) -> str:
        """
        获取记忆摘要，用于生成AI的上下文提示

        参数:
            limit: 最多包含的记忆数量

        返回:
            格式化的记忆摘要文本
        """
        memories = self.recall(limit=limit)
        if not memories:
            return f"{self.owner_name} 目前没有任何记忆。"

        summary_lines = [f"{self.owner_name} 记住以下信息:"]
        for i, mem in enumerate(memories, 1):
            strength_indicator = ""
            if mem.get_current_strength() < 0.3:
                strength_indicator = " [正在遗忘]"
            elif mem.get_current_strength() > 0.8:
                strength_indicator = " [清晰]"
            summary_lines.append(f"{i}. {mem.content}{strength_indicator}")

        return "\n".join(summary_lines)

    def cleanup_forgotten(self, threshold: float = 0.01) -> int:
        """
        清理强度低于阈值的记忆（模拟自然遗忘）

        参数:
            threshold: 遗忘阈值，强度低于此值的记忆将被删除

        返回:
            清理的记忆数量
        """
        forgotten_ids = []
        for mem_id, memory in self.memories.items():
            if not memory.is_permanent():  # 永久记忆永不清理
                if memory.get_current_strength() < threshold:
                    forgotten_ids.append(mem_id)

        for mem_id in forgotten_ids:
            del self.memories[mem_id]

        return len(forgotten_ids)

    def get_memory_count(self) -> int:
        """获取当前记忆总数"""
        return len(self.memories)

    def save_to_file(self, filepath: str):
        """
        将记忆系统保存到JSON文件

        参数:
            filepath: 保存路径
        """
        data = {
            'owner_name': self.owner_name,
            'memories': [mem.to_dict() for mem in self.memories.values()]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str):
        """
        从JSON文件加载记忆系统

        参数:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.owner_name = data.get('owner_name', self.owner_name)
            self.memories = {}
            for mem_data in data.get('memories', []):
                memory = MemoryItem.from_dict(mem_data)
                self.memories[memory.id] = memory
        except FileNotFoundError:
            print(f"记忆文件 {filepath} 不存在，使用空记忆系统")
        except Exception as e:
            print(f"加载记忆文件失败: {e}")

    def __str__(self) -> str:
        """返回记忆系统的简要描述"""
        return f"MemorySystem({self.owner_name}, 记忆数:{len(self.memories)})"


# 简单的测试代码（仅在直接运行此文件时执行）
if __name__ == "__main__":
    print("=== 记忆系统测试 ===\n")

    # 创建主角的记忆系统
    protagonist_memory = MemorySystem("主角")

    # 添加永久记忆
    protagonist_memory.add_permanent_memory("我是人类")
    print("✓ 添加永久记忆: 我是人类")

    # 添加普通记忆
    protagonist_memory.add_memory("今天是晴天", importance=0.5)
    protagonist_memory.add_memory("书店老板推荐了一本有趣的书", importance=0.7)
    print("✓ 添加了两条普通记忆")

    # 显示当前记忆
    print("\n当前记忆:")
    for mem in protagonist_memory.memories.values():
        print(f"  {mem}")

    # 测试召回
    print("\n召回记忆（按强度排序）:")
    recalled = protagonist_memory.recall(limit=5)
    for mem in recalled:
        print(f"  {mem}")

    # 测试记忆摘要
    print(f"\n记忆摘要:\n{protagonist_memory.get_summary()}")

    # 测试关键词召回
    print("\n关键词'书'相关记忆:")
    keyword_recall = protagonist_memory.recall_by_keywords(["书"])
    for mem in keyword_recall:
        print(f"  {mem}")

    print("\n✓ 记忆系统测试完成")