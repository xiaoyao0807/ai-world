"""
世界状态管理模块
负责维护文本世界中的地点、时间、天气和事件队列。
"""

from datetime import datetime, timedelta
from typing import Dict, List


class WorldState:
    """
    世界状态类
    管理：
    1. 当前地点
    2. 世界时间
    3. 当前天气
    4. 事件队列
    """

    def __init__(self):
        self.available_locations: List[str] = [
            "广场",
            "书店",
            "花店",
            "餐厅",
            "警察局",
            "酒吧",
            "咖啡馆",
            "公园",
            "小巷",
        ]
        self.current_location: str = "广场"
        self.current_time: datetime = datetime(2026, 4, 7, 9, 0, 0)
        self.weather: str = "晴天"
        self.event_queue: List[str] = []
        self.location_objects: Dict[str, List[str]] = {loc: [] for loc in self.available_locations}
        self.global_facts: List[str] = [
            "广场是一个圆形大广场",
            "广场旁边是步行街，两个区域在地图上连在一起",
            "零的家在广场旁边的一栋居民楼里",
        ]
        self.location_objects["广场"].extend(
            [
                "正中心有一座圆形喷泉",
                "喷泉外圈偏远一点的位置有一把大型固定式遮阳伞",
                "遮阳伞下是连体桌椅，形成一块小型公共休息区",
                "连向步行街的拱形入口",
            ]
        )
        self.location_objects["广场"].append("路牌：书店、花店、餐厅、警察局、酒吧、咖啡馆、公园、小巷")
        self.location_objects["花店"].append("花店玻璃橱窗与二楼小阳台")
        self.location_objects["咖啡馆"].extend(["自助点单机器", "后厨咖啡机器人"])
        self.location_objects["餐厅"].append("金姨小馆的木招牌")
        self.location_objects["书店"].append("门口告示：营业时间 09:50-18:00（店主在岗时可进入）")
        self.location_aliases: Dict[str, str] = {
            "金姨小馆": "餐厅",
            "老陈书店": "书店",
            "花店二楼": "花店",
            "警局": "警察局",
            "派出所": "警察局",
        }

    def resolve_location(self, raw_location: str) -> str:
        cleaned = raw_location.strip()
        if cleaned in self.available_locations:
            return cleaned
        if cleaned in self.location_aliases:
            return self.location_aliases[cleaned]
        for alias, canonical in self.location_aliases.items():
            if alias in cleaned:
                return canonical
        for loc in self.available_locations:
            if loc in cleaned or cleaned in loc:
                return loc
        return cleaned

    def move_to(self, location: str, allow_bookstore_entry: bool = True) -> str:
        location = self.resolve_location(location)
        if location not in self.available_locations:
            return f"无法移动：地点 '{location}' 不存在。可选地点：{', '.join(self.available_locations)}"
        if location == "书店" and not allow_bookstore_entry:
            return "书店现在没人，不可以进入。"
        if not self.is_location_open(location):
            return f"{location}当前未开放，暂时无法进入。"
        if location == self.current_location:
            return f"你已经在{location}了。"

        old_location = self.current_location
        self.current_location = location
        self.advance_time(minutes=10)
        event_text = f"你从{old_location}移动到了{location}。"
        self.add_event(event_text)
        return event_text

    def wait(self, hours: float = 1.0) -> str:
        if hours <= 0:
            hours = 1.0
        minutes = int(hours * 60)
        self.advance_time(minutes=minutes)

        hour = self.current_time.hour
        if 6 <= hour < 12:
            self.weather = "晴天"
        elif 12 <= hour < 18:
            self.weather = "多云"
        elif 18 <= hour < 22:
            self.weather = "微风"
        else:
            self.weather = "夜凉"

        event_text = f"你在{self.current_location}等待了{hours:.1f}小时。"
        self.add_event(event_text)
        return event_text

    def advance_time(self, minutes: int) -> None:
        if minutes < 0:
            return
        self.current_time += timedelta(minutes=minutes)

    def add_event(self, event_text: str) -> None:
        if not event_text or not event_text.strip():
            return
        timestamp = self.current_time.strftime("%H:%M")
        self.event_queue.append(f"[{timestamp}] {event_text}")

    def pop_events(self, limit: int = 5) -> List[str]:
        if limit <= 0:
            return []
        events = self.event_queue[:limit]
        self.event_queue = self.event_queue[limit:]
        return events

    def get_time_text(self) -> str:
        return self.current_time.strftime("%Y-%m-%d %H:%M")

    def get_world_description(self, for_actor: bool = False) -> str:
        local_items = self.location_objects.get(self.current_location, [])
        local_text = "、".join(local_items) if local_items else "暂无特别事物"
        facts_text = "；".join(self.global_facts[-3:]) if self.global_facts else "暂无"
        if for_actor:
            # 零不应直接看到系统隐藏规则（如各店营业时间/开放状态）
            return (
                f"你当前在{self.current_location}。\n"
                f"时间：{self.get_time_text()}\n"
                f"天气：{self.weather}\n"
                f"你眼前可见事物：{local_text}\n"
                f"你已知公共信息：{facts_text if facts_text != '暂无' else '信息有限，需要继续探索'}"
            )

        open_text = "、".join(
            [f"{loc}:{'开放' if self.is_location_open(loc) else '关闭'}" for loc in self.available_locations]
        )
        return (
            f"你当前在{self.current_location}。\n"
            f"时间：{self.get_time_text()}\n"
            f"天气：{self.weather}\n"
            f"地点事物：{local_text}\n"
            f"近期世界事实：{facts_text}\n"
            f"地点开放状态：{open_text}\n"
            f"可前往地点：{', '.join(self.available_locations)}"
        )

    def is_location_open(self, location: str) -> bool:
        hour = self.current_time.hour + self.current_time.minute / 60.0
        if location == "酒吧":
            return hour >= 20.0 or hour < 3.0
        if location == "餐厅":
            return 7.0 <= hour < 21.0
        if location == "警察局":
            return True
        if location == "咖啡馆":
            return True
        return True

    def get_location_hours_text(self, location: str) -> str:
        location = self.resolve_location(location)
        hours_map = {
            "书店": "09:50-18:00（店主在岗时可进入）",
            "花店": "08:30-19:00",
            "餐厅": "07:00-21:00",
            "酒吧": "20:00-03:00",
            "警察局": "00:00-24:00",
            "咖啡馆": "00:00-24:00",
            "广场": "00:00-24:00",
            "公园": "00:00-24:00",
            "小巷": "00:00-24:00",
        }
        return hours_map.get(location, "未知")

    def add_world_thing(self, thing: str, location: str = "") -> str:
        cleaned_thing = thing.strip()
        cleaned_location = location.strip()
        if not cleaned_thing:
            return "添加失败：事物内容为空。"

        if cleaned_location:
            if cleaned_location not in self.available_locations:
                return f"添加失败：地点'{cleaned_location}'不存在。"
            self.location_objects[cleaned_location].append(cleaned_thing)
            event_text = f"观察者在{cleaned_location}添加了新事物：{cleaned_thing}"
            self.add_event(event_text)
            return event_text

        self.global_facts.append(cleaned_thing)
        event_text = f"观察者向世界加入了新事实：{cleaned_thing}"
        self.add_event(event_text)
        return event_text

    def __str__(self) -> str:
        return (
            f"WorldState(location={self.current_location}, "
            f"time={self.get_time_text()}, weather={self.weather}, "
            f"events={len(self.event_queue)})"
        )


if __name__ == "__main__":
    world = WorldState()
    print("=== 世界状态测试 ===")
    print(world.get_world_description())
