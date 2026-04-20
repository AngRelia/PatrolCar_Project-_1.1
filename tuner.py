# tuner.py
# 核心功能：提供一个图形化界面（GUI），让用户可以在设备上直接调节不同颜色的阈值并保存。
# 设计亮点：
# 1. 自动校验：严格限制 LAB/Grayscale 的取值范围，并保证 Min <= Max。
# 2. 性能优化：图像二值化预览非常耗时，采用“降帧+降分辨率缓存”策略，保证UI响应不卡顿。
# 3. 交互优化：长按按钮支持“连发”，并且按住越久，调节速度越快。

import json
import os
import time

from maix import image

# 导入视觉算法模块（用于生成二值化预览图）
import vision

# 阈值配置文件的本地存储路径
THRESH_FILE = "/data/my_thresholds.json"

# ==================== 默认阈值配置 ====================
# LAB 颜色空间阈值格式: [L_min, L_max, A_min, A_max, B_min, B_max]
# L (亮度): 0 ~ 100
# A (红绿轴): -128 ~ 127
# B (黄蓝轴): -128 ~ 127
DEFAULT_THRESHOLDS = {
    "red": [0, 80, 40, 80, 10, 80],
    "green": [0, 80, -120, -10, 0, 30],
    "blue": [0, 80, 30, 100, -120, -60],
    "black": [0, 30], # 灰度值格式: [Gray_min, Gray_max]，范围 0 ~ 100
}

COLOR_ORDER = ["red", "green", "blue", "black"]
COLOR_LABELS = {
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "black": "Black",
}

# 全局变量：当前正在使用的阈值字典
current_thresholds = {}

# ==================== UI 状态与缓存变量 ====================
_ui_color_idx = 0        # 当前正在调节的颜色索引（对应 COLOR_ORDER）
_toast_text = ""         # 屏幕弹窗提示文本
_toast_until = 0.0       # 弹窗提示消失的时间戳

# 防误触与长按连发控制
_ignore_until_release = False # 状态切换防误触标记（如从 main 长按切过来时，忽略初次触摸）
_hold_action = None           # 当前被按住的按钮动作
_hold_start_time = 0.0        # 按钮开始按下的时间戳
_hold_last_repeat = 0.0       # 上一次触发“连发”的时间戳

# 预览图缓存优化：防止每帧都进行二值化运算导致 UI 极度卡顿
_preview_cache_data = None    # 缓存的二值化预览图像数据
_preview_cache_color = ""     # 缓存对应的颜色类别
_preview_cache_key = None     # 缓存对应的具体阈值参数（参数一变，缓存失效）
_preview_cache_time = 0.0     # 上次生成缓存的时间

# 调参页采用高刷新率UI + 固定 128x96 低分辨率二值预览的策略。
# 0.03秒 = 约 33 FPS 的预览刷新率。
PREVIEW_REFRESH_INTERVAL = 0.03
PREVIEW_SIZE = (128, 96)
PREVIEW_SIZE_TEXT = "128x96"


def _copy_defaults():
    """深拷贝默认阈值，防止意外修改全局常量"""
    return {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}


def _clamp(v, lo, hi):
    """限幅函数：将数值 v 限制在 [lo, hi] 区间内"""
    return max(lo, min(hi, int(v)))


def _normalize_rgb(values, default_values):
    """
    规范化 LAB 颜色阈值，确保数据合法性。
    为什么要规范化？防止读取损坏的 JSON 文件导致程序崩溃或算法异常。
    """
    if not isinstance(values, list):
        return list(default_values)
        
    merged = list(default_values)
    # 取最多6个元素覆盖默认值
    for i in range(min(6, len(values))):
        merged[i] = int(values[i])

    # 严格限制 L, A, B 的物理取值范围
    merged[0] = _clamp(merged[0], 0, 100)
    merged[1] = _clamp(merged[1], 0, 100)
    merged[2] = _clamp(merged[2], -128, 127)
    merged[3] = _clamp(merged[3], -128, 127)
    merged[4] = _clamp(merged[4], -128, 127)
    merged[5] = _clamp(merged[5], -128, 127)

    # 逻辑校验：最小值决不能大于最大值，否则视觉算法会直接报错
    if merged[0] > merged[1]:
        merged[1] = merged[0]
    if merged[2] > merged[3]:
        merged[3] = merged[2]
    if merged[4] > merged[5]:
        merged[5] = merged[4]
    return merged


def _normalize_black(values, default_values):
    """规范化黑色的灰度阈值，逻辑同 _normalize_rgb"""
    if isinstance(values, list):
        if len(values) >= 2:
            out = [int(values[0]), int(values[1])]
        else:
            out = list(default_values)
    else:
        out = list(default_values)

    out[0] = _clamp(out[0], 0, 100)
    out[1] = _clamp(out[1], 0, 100)
    if out[0] > out[1]:
        out[1] = out[0]
    return out


def _normalize_thresholds(data):
    """规范化所有阈值配置字典"""
    normalized = _copy_defaults()
    if not isinstance(data, dict):
        return normalized

    for color in ["red", "green", "blue"]:
        normalized[color] = _normalize_rgb(data.get(color), DEFAULT_THRESHOLDS[color])

    # 黑色阈值的处理稍有不同，只取前两个值（因为是灰度不是LAB）
    black_raw = data.get("black")
    if isinstance(black_raw, list) and len(black_raw) >= 6:
        black_raw = [black_raw[0], black_raw[1]]
    normalized["black"] = _normalize_black(black_raw, DEFAULT_THRESHOLDS["black"])
    return normalized


def load_thresholds():
    """
    从本地 JSON 文件加载阈值。
    如果文件不存在或内容损坏，静默回退到默认设置。
    """
    global current_thresholds
    if os.path.exists(THRESH_FILE):
        try:
            with open(THRESH_FILE, "r") as f:
                current_thresholds = _normalize_thresholds(json.load(f))
            return
        except Exception:
            pass # 出错不抛异常，直接进入下一步的默认值逻辑
    current_thresholds = _copy_defaults()


def save_thresholds():
    """将当前阈值写入本地 JSON 文件进行持久化保存"""
    dir_name = os.path.dirname(THRESH_FILE)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(THRESH_FILE, "w") as f:
        json.dump(current_thresholds, f)


def enter_tuner():
    """
    进入调参界面的初始化挂钩函数（由 main.py 调用）。
    重置缓存并开启防误触机制。
    """
    global _ignore_until_release, _hold_action
    global _preview_cache_data, _preview_cache_color, _preview_cache_key, _preview_cache_time
    
    # 设为 True，直到用户松开手指才开始响应按钮，防止 main 中的长按直接触发调参界面的某个按钮
    _ignore_until_release = True 
    _hold_action = None
    
    # 清空缓存强制重新渲染
    _preview_cache_data = None
    _preview_cache_color = ""
    _preview_cache_key = None
    _preview_cache_time = 0.0


def _point_in_btn(px, py, btn):
    """碰撞检测：点是否在按钮区域内"""
    return btn["x"] <= px <= btn["x"] + btn["w"] and btn["y"] <= py <= btn["y"] + btn["h"]


def _draw_button(img, btn, text):
    """绘制带黑边、白底、居中黑字的扁平化按钮"""
    img.draw_rect(btn["x"], btn["y"], btn["w"], btn["h"], image.COLOR_BLACK, thickness=3)
    img.draw_rect(btn["x"] + 2, btn["y"] + 2, btn["w"] - 4, btn["h"] - 4, image.COLOR_WHITE, thickness=-1)
    
    # 计算文字的像素宽高以实现居中
    tw, th = image.string_size(text, scale=1.8, thickness=2)
    tx = btn["x"] + (btn["w"] - tw) // 2
    ty = btn["y"] + (btn["h"] - th) // 2
    img.draw_string(tx, ty, text, image.COLOR_BLACK, scale=1.8, thickness=2)


def _layout():
    """
    计算UI的绝对坐标布局。
    设计为：左边一排按键(-)，右边一排按键(+)，中间是画面预览区，底部是状态显示区。
    """
    btn_w = 112
    btn_h = 48
    top = 8
    left_x = 8
    right_x = 640 - 8 - btn_w
    row_y = [top + i * btn_h for i in range(8)] # 垂直方向分配8个按钮的Y坐标

    preview_x = left_x + btn_w + 8
    preview_w = right_x - preview_x - 8

    status_h = 96
    status_y = 480 - status_h - 8
    status_x = preview_x + 8
    status_w = preview_w - 16

    preview_y = top
    preview_h = status_y - preview_y - 8

    return {
        "btn_w": btn_w, "btn_h": btn_h,
        "left_x": left_x, "right_x": right_x,
        "row_y": row_y,
        "preview": (preview_x, preview_y, preview_w, preview_h),
        "status": (status_x, status_y, status_w, status_h),
    }


def _build_rgb_buttons(layout):
    """构建彩色(RGB)调参所需的按钮字典（减小在左，增大在右）"""
    left_x, right_x = layout["left_x"], layout["right_x"]
    btn_w, btn_h = layout["btn_w"], layout["btn_h"]
    row_y = layout["row_y"]

    return {
        "back": {"x": left_x, "y": row_y[0], "w": btn_w, "h": btn_h, "text": "Back"},
        "switch": {"x": right_x, "y": row_y[0], "w": btn_w, "h": btn_h, "text": "Switch"},
        
        "lmin_dec": {"x": left_x, "y": row_y[1], "w": btn_w, "h": btn_h, "text": "Lmin-"},
        "lmax_dec": {"x": left_x, "y": row_y[2], "w": btn_w, "h": btn_h, "text": "Lmax-"},
        "amin_dec": {"x": left_x, "y": row_y[3], "w": btn_w, "h": btn_h, "text": "Amin-"},
        "amax_dec": {"x": left_x, "y": row_y[4], "w": btn_w, "h": btn_h, "text": "Amax-"},
        "bmin_dec": {"x": left_x, "y": row_y[5], "w": btn_w, "h": btn_h, "text": "Bmin-"},
        "bmax_dec": {"x": left_x, "y": row_y[6], "w": btn_w, "h": btn_h, "text": "Bmax-"},
        "reset": {"x": left_x, "y": row_y[7], "w": btn_w, "h": btn_h, "text": "RESET"},
        
        "lmin_inc": {"x": right_x, "y": row_y[1], "w": btn_w, "h": btn_h, "text": "Lmin+"},
        "lmax_inc": {"x": right_x, "y": row_y[2], "w": btn_w, "h": btn_h, "text": "Lmax+"},
        "amin_inc": {"x": right_x, "y": row_y[3], "w": btn_w, "h": btn_h, "text": "Amin+"},
        "amax_inc": {"x": right_x, "y": row_y[4], "w": btn_w, "h": btn_h, "text": "Amax+"},
        "bmin_inc": {"x": right_x, "y": row_y[5], "w": btn_w, "h": btn_h, "text": "Bmin+"},
        "bmax_inc": {"x": right_x, "y": row_y[6], "w": btn_w, "h": btn_h, "text": "Bmax+"},
        "save": {"x": right_x, "y": row_y[7], "w": btn_w, "h": btn_h, "text": "SAVE"},
    }


def _build_black_buttons(layout):
    """构建黑色(灰度)调参所需的按钮字典（只调节 Gmin 和 Gmax）"""
    left_x, right_x = layout["left_x"], layout["right_x"]
    btn_w, btn_h = layout["btn_w"], layout["btn_h"]
    row_y = layout["row_y"]

    return {
        "back": {"x": left_x, "y": row_y[0], "w": btn_w, "h": btn_h, "text": "Back"},
        "switch": {"x": right_x, "y": row_y[0], "w": btn_w, "h": btn_h, "text": "Switch"},
        
        "gmin_dec": {"x": left_x, "y": row_y[1], "w": btn_w, "h": btn_h, "text": "Gmin-"},
        "gmax_dec": {"x": left_x, "y": row_y[2], "w": btn_w, "h": btn_h, "text": "Gmax-"},
        "gmin_inc": {"x": right_x, "y": row_y[1], "w": btn_w, "h": btn_h, "text": "Gmin+"},
        "gmax_inc": {"x": right_x, "y": row_y[2], "w": btn_w, "h": btn_h, "text": "Gmax+"},
        
        "reset": {"x": left_x, "y": row_y[7], "w": btn_w, "h": btn_h, "text": "RESET"},
        "save": {"x": right_x, "y": row_y[7], "w": btn_w, "h": btn_h, "text": "SAVE"},
    }


def _show_toast(text):
    """设置一个短时留存的屏幕提示语（如保存成功）"""
    global _toast_text, _toast_until
    _toast_text = text
    _toast_until = time.time() + 1.0  # 驻留 1 秒钟


def _draw_status_box(img, current_color, layout):
    """绘制底部状态框，显示当前的阈值具体数字"""
    status_x, status_y, status_w, status_h = layout["status"]
    img.draw_rect(status_x, status_y, status_w, status_h, image.COLOR_BLACK, thickness=4)
    img.draw_rect(status_x + 2, status_y + 2, status_w - 4, status_h - 4, image.COLOR_WHITE, thickness=-1)

    if current_color == "black":
        line1 = "L:--- A:--- B:---" # 黑色不使用 LAB，显示占位符
    else:
        vals = current_thresholds[current_color]
        line1 = f"L:{vals[0]:03d}-{vals[1]:03d}  A:{vals[2]:03d}-{vals[3]:03d}  B:{vals[4]:03d}-{vals[5]:03d}"

    g_vals = current_thresholds["black"]
    line2 = f"G:{g_vals[0]:03d}-{g_vals[1]:03d}  C:{COLOR_LABELS[current_color]}  R:{PREVIEW_SIZE_TEXT}"
    line3 = "W:tracked  B:background" # 提示：二值图预览中，白色代表追踪的目标，黑色代表被剔除的背景

    img.draw_string(status_x + 10, status_y + 14, line1, image.COLOR_BLACK, scale=1.5, thickness=2)
    img.draw_string(status_x + 10, status_y + 44, line2, image.COLOR_BLACK, scale=1.35, thickness=2)
    img.draw_string(status_x + 10, status_y + 70, line3, image.COLOR_BLACK, scale=1.25, thickness=2)


def _draw_toast(img):
    """如果有生效的提示语，在屏幕顶端居中绘制"""
    if time.time() <= _toast_until and _toast_text:
        tw, th = image.string_size(_toast_text, scale=2.0, thickness=2)
        x = (640 - tw) // 2
        y = 10
        # 黑色半透明底框
        img.draw_rect(x - 12, y - 6, tw + 24, th + 12, image.COLOR_BLACK, thickness=-1)
        img.draw_string(x, y, _toast_text, image.COLOR_YELLOW, scale=2.0, thickness=2)


def _adjust_rgb(color, idx, delta):
    """
    修改 LAB 阈值的核心逻辑（自带防越界与联动纠错）。
    idx: 要修改的参数索引(0:Lmin, 1:Lmax, 2:Amin...)
    delta: 步长（+1 或 -1）
    """
    vals = current_thresholds[color]
    # 对 L 通道特殊限幅 (0-100)，对 A/B 通道限幅 (-128 到 127)
    if idx in (0, 1):
        vals[idx] = _clamp(vals[idx] + delta, 0, 100)
    else:
        vals[idx] = _clamp(vals[idx] + delta, -128, 127)

    # 联动纠错机制（重要！）：
    # 比如在 L 通道，pair_start 为 0 (Lmin)，pair_start+1 为 1 (Lmax)。
    # 如果用户把 Lmin 调得比 Lmax 还要大，系统自动把 Lmax “推着”往上走，反之亦然。
    pair_start = (idx // 2) * 2
    if vals[pair_start] > vals[pair_start + 1]:
        if idx % 2 == 0: # 正在调节 min 值，把 max 推大
            vals[pair_start + 1] = vals[pair_start]
        else:            # 正在调节 max 值，把 min 推小
            vals[pair_start] = vals[pair_start + 1]


def _adjust_black(idx, delta):
    """黑色(灰度)阈值的步长调节与纠错逻辑"""
    vals = current_thresholds["black"]
    vals[idx] = _clamp(vals[idx] + delta, 0, 100)
    if vals[0] > vals[1]:
        if idx == 0:
            vals[1] = vals[0]
        else:
            vals[0] = vals[1]


def _is_repeatable_action(action, current_color):
    """
    判断动作是否支持“连发”。
    Switch（切换颜色）、Save（保存）、Back（返回）这些按键不支持长按连发。
    只有加减参数的按钮支持。
    """
    if current_color == "black":
        return action in ("gmin_dec", "gmax_dec", "gmin_inc", "gmax_inc")
    return action in (
        "lmin_dec", "lmax_dec", "amin_dec", "amax_dec", "bmin_dec", "bmax_dec",
        "lmin_inc", "lmax_inc", "amin_inc", "amax_inc", "bmin_inc", "bmax_inc",
    )


def _handle_action(action, current_color):
    """根据动作执行具体业务逻辑，返回状态信号"""
    global _ui_color_idx

    # 控制流动作
    if action == "back":
        return "back" # 通知 main.py 退出调参模块
    if action == "switch":
        _ui_color_idx = (_ui_color_idx + 1) % len(COLOR_ORDER) # 轮询切换调参颜色
        return None
    if action == "reset":
        current_thresholds[current_color] = list(DEFAULT_THRESHOLDS[current_color])
        _show_toast(f"({COLOR_LABELS[current_color]} Rested)")
        return None
    if action == "save":
        save_thresholds()
        _show_toast("(Saved)")
        return None

    # 数值调节动作（映射具体的步长和索引）
    if current_color == "black":
        map_black = {
            "gmin_dec": (0, -1), "gmax_dec": (1, -1),
            "gmin_inc": (0, 1),  "gmax_inc": (1, 1),
        }
        if action in map_black:
            idx, delta = map_black[action]
            _adjust_black(idx, delta)
            return None
    else:
        map_rgb = {
            "lmin_dec": (0, -1), "lmax_dec": (1, -1),
            "amin_dec": (2, -1), "amax_dec": (3, -1),
            "bmin_dec": (4, -1), "bmax_dec": (5, -1),
            "lmin_inc": (0, 1),  "lmax_inc": (1, 1),
            "amin_inc": (2, 1),  "amax_inc": (3, 1),
            "bmin_inc": (4, 1),  "bmax_inc": (5, 1),
        }
        if action in map_rgb:
            idx, delta = map_rgb[action]
            _adjust_rgb(current_color, idx, delta)
            return None

    return None


def _find_action_from_touch(tx, ty, buttons):
    """遍历所有按钮，判断触点 (tx, ty) 落在哪个按钮上，返回对应的 action 字符串"""
    for action, btn in buttons.items():
        if _point_in_btn(tx, ty, btn):
            return action
    return None


def run_tuner_ui(img, ts, _disp):
    """
    调参界面的主处理函数。每帧由 main.py 调用。
    负责 UI 渲染、缓存刷新以及复杂的触摸连按交互处理。
    """
    global _ignore_until_release, _hold_action, _hold_start_time, _hold_last_repeat
    global _preview_cache_data, _preview_cache_color, _preview_cache_key, _preview_cache_time

    # 1. 准备布局与按钮数据
    layout = _layout()
    current_color = COLOR_ORDER[_ui_color_idx]
    buttons = _build_black_buttons(layout) if current_color == "black" else _build_rgb_buttons(layout)
    now = time.time()

    # 2. 图像预览缓存优化机制
    # 判断是否需要重新进行视觉算法的二值化处理
    current_key = tuple(current_thresholds[current_color])
    need_refresh = (
        _preview_cache_data is None             # 初始化时
        or _preview_cache_color != current_color # 切换了颜色分类时
        or _preview_cache_key != current_key    # 用户修改了任何一个数值时
        or (now - _preview_cache_time) >= PREVIEW_REFRESH_INTERVAL # 达到固定刷新间隔时
    )
    if need_refresh:
        # 调用算法层执行耗时的寻找并提取二值化图操作
        _preview_cache_data = vision.get_binary_preview_rects(
            img,
            current_color,
            current_thresholds,
            use_mask=True,
            preview_size=PREVIEW_SIZE,
        )
        # 更新缓存印记
        _preview_cache_color = current_color
        _preview_cache_key = current_key
        _preview_cache_time = now

    # 3. 绘制底层UI
    img.draw_rect(0, 0, 640, 480, image.COLOR_GRAY, thickness=-1) # 画灰色背景底板
    vision.draw_binary_preview(img, _preview_cache_data, layout["preview"]) # 绘制中间的缓存预览结果

    # 4. 绘制上层UI组件
    for btn in buttons.values():
        _draw_button(img, btn, btn["text"])
    _draw_status_box(img, current_color, layout)
    _draw_toast(img)

    # 5. 读取并处理触摸事件
    touch_data = ts.read()
    pressed = touch_data and len(touch_data) >= 3 and touch_data[2] == 1

    # 【防误触机制】如果用户松开手指，立刻解除忽略锁定状态，并清空历史长按数据
    if not pressed:
        _ignore_until_release = False
        _hold_action = None
        return None

    # 如果还在强制忽略阶段（比如还没松开从 main 过来的那一指头），则直接跳过处理
    if _ignore_until_release:
        return None

    # 获取坐标点，检测当前按下了哪个按钮
    tx, ty = touch_data[0], touch_data[1]
    action = _find_action_from_touch(tx, ty, buttons)
    
    # 按在空白处，清空长按判定
    if not action:
        _hold_action = None
        return None

    # 【按键初次触发逻辑】与上一次按压的动作不同，说明是刚开始按下的“第一下”
    if action != _hold_action:
        _hold_action = action
        _hold_start_time = now
        _hold_last_repeat = now
        return _handle_action(action, current_color)

    # 【长按连发逻辑】如果用户一直按着相同的按键，并且这个按键支持连发（加减号）
    if _is_repeatable_action(action, current_color):
        hold_elapsed = now - _hold_start_time
        # 按下的前 0.70 秒，每 0.30 秒触发一次（慢速调节）。
        # 一旦按住超过 0.70 秒，每 0.08 秒就会触发一次（快速调节，极大提升使用体验）。
        repeat_interval = 0.30 if hold_elapsed < 0.70 else 0.08
        
        # 满足连发间隔时间，执行调节
        if now - _hold_last_repeat >= repeat_interval:
            _hold_last_repeat = now
            return _handle_action(action, current_color)

    return None