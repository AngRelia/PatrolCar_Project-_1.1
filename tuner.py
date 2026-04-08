import json
import os
import time

from maix import image

import vision

THRESH_FILE = "/data/my_thresholds.json"

DEFAULT_THRESHOLDS = {
    "red": [0, 80, 40, 80, 10, 80],
    "green": [0, 80, -120, -10, 0, 30],
    "blue": [0, 80, 30, 100, -120, -60],
    "black": [0, 30],
}

COLOR_ORDER = ["red", "green", "blue", "black"]
COLOR_LABELS = {
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "black": "Black",
}

current_thresholds = {}

_ui_color_idx = 0
_toast_text = ""
_toast_until = 0.0

_ignore_until_release = False
_hold_action = None
_hold_start_time = 0.0
_hold_last_repeat = 0.0

_preview_cache_data = None
_preview_cache_color = ""
_preview_cache_key = None
_preview_cache_time = 0.0

# 调参页采用高刷新率 + 固定 128x96 低分辨率二值预览。
PREVIEW_REFRESH_INTERVAL = 0.03
PREVIEW_SIZE = (128, 96)
PREVIEW_SIZE_TEXT = "128x96"


def _copy_defaults():
    return {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _normalize_rgb(values, default_values):
    if not isinstance(values, list):
        return list(default_values)
    merged = list(default_values)
    for i in range(min(6, len(values))):
        merged[i] = int(values[i])

    merged[0] = _clamp(merged[0], 0, 100)
    merged[1] = _clamp(merged[1], 0, 100)
    merged[2] = _clamp(merged[2], -128, 127)
    merged[3] = _clamp(merged[3], -128, 127)
    merged[4] = _clamp(merged[4], -128, 127)
    merged[5] = _clamp(merged[5], -128, 127)

    if merged[0] > merged[1]:
        merged[1] = merged[0]
    if merged[2] > merged[3]:
        merged[3] = merged[2]
    if merged[4] > merged[5]:
        merged[5] = merged[4]
    return merged


def _normalize_black(values, default_values):
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
    normalized = _copy_defaults()
    if not isinstance(data, dict):
        return normalized

    for color in ["red", "green", "blue"]:
        normalized[color] = _normalize_rgb(data.get(color), DEFAULT_THRESHOLDS[color])

    black_raw = data.get("black")
    if isinstance(black_raw, list) and len(black_raw) >= 6:
        black_raw = [black_raw[0], black_raw[1]]
    normalized["black"] = _normalize_black(black_raw, DEFAULT_THRESHOLDS["black"])
    return normalized


def load_thresholds():
    global current_thresholds
    if os.path.exists(THRESH_FILE):
        try:
            with open(THRESH_FILE, "r") as f:
                current_thresholds = _normalize_thresholds(json.load(f))
            return
        except Exception:
            pass
    current_thresholds = _copy_defaults()


def save_thresholds():
    dir_name = os.path.dirname(THRESH_FILE)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(THRESH_FILE, "w") as f:
        json.dump(current_thresholds, f)


def enter_tuner():
    global _ignore_until_release, _hold_action
    global _preview_cache_data, _preview_cache_color, _preview_cache_key, _preview_cache_time
    _ignore_until_release = True
    _hold_action = None
    _preview_cache_data = None
    _preview_cache_color = ""
    _preview_cache_key = None
    _preview_cache_time = 0.0


def _point_in_btn(px, py, btn):
    return btn["x"] <= px <= btn["x"] + btn["w"] and btn["y"] <= py <= btn["y"] + btn["h"]


def _draw_button(img, btn, text):
    img.draw_rect(btn["x"], btn["y"], btn["w"], btn["h"], image.COLOR_BLACK, thickness=3)
    img.draw_rect(btn["x"] + 2, btn["y"] + 2, btn["w"] - 4, btn["h"] - 4, image.COLOR_WHITE, thickness=-1)
    tw, th = image.string_size(text, scale=1.8, thickness=2)
    tx = btn["x"] + (btn["w"] - tw) // 2
    ty = btn["y"] + (btn["h"] - th) // 2
    img.draw_string(tx, ty, text, image.COLOR_BLACK, scale=1.8, thickness=2)


def _layout():
    btn_w = 112
    btn_h = 48
    top = 8
    left_x = 8
    right_x = 640 - 8 - btn_w
    row_y = [top + i * btn_h for i in range(8)]

    preview_x = left_x + btn_w + 8
    preview_w = right_x - preview_x - 8

    status_h = 96
    status_y = 480 - status_h - 8
    status_x = preview_x + 8
    status_w = preview_w - 16

    preview_y = top
    preview_h = status_y - preview_y - 8

    return {
        "btn_w": btn_w,
        "btn_h": btn_h,
        "left_x": left_x,
        "right_x": right_x,
        "row_y": row_y,
        "preview": (preview_x, preview_y, preview_w, preview_h),
        "status": (status_x, status_y, status_w, status_h),
    }


def _build_rgb_buttons(layout):
    left_x = layout["left_x"]
    right_x = layout["right_x"]
    btn_w = layout["btn_w"]
    btn_h = layout["btn_h"]
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
    left_x = layout["left_x"]
    right_x = layout["right_x"]
    btn_w = layout["btn_w"]
    btn_h = layout["btn_h"]
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
    global _toast_text, _toast_until
    _toast_text = text
    _toast_until = time.time() + 1.0


def _draw_status_box(img, current_color, layout):
    status_x, status_y, status_w, status_h = layout["status"]
    img.draw_rect(status_x, status_y, status_w, status_h, image.COLOR_BLACK, thickness=4)
    img.draw_rect(status_x + 2, status_y + 2, status_w - 4, status_h - 4, image.COLOR_WHITE, thickness=-1)

    if current_color == "black":
        line1 = "L:--- A:--- B:---"
    else:
        vals = current_thresholds[current_color]
        line1 = f"L:{vals[0]:03d}-{vals[1]:03d}  A:{vals[2]:03d}-{vals[3]:03d}  B:{vals[4]:03d}-{vals[5]:03d}"

    g_vals = current_thresholds["black"]
    line2 = f"G:{g_vals[0]:03d}-{g_vals[1]:03d}  C:{COLOR_LABELS[current_color]}  R:{PREVIEW_SIZE_TEXT}"
    line3 = "W:tracked  B:background"

    img.draw_string(status_x + 10, status_y + 14, line1, image.COLOR_BLACK, scale=1.5, thickness=2)
    img.draw_string(status_x + 10, status_y + 44, line2, image.COLOR_BLACK, scale=1.35, thickness=2)
    img.draw_string(status_x + 10, status_y + 70, line3, image.COLOR_BLACK, scale=1.25, thickness=2)


def _draw_toast(img):
    if time.time() <= _toast_until and _toast_text:
        tw, th = image.string_size(_toast_text, scale=2.0, thickness=2)
        x = (640 - tw) // 2
        y = 10
        img.draw_rect(x - 12, y - 6, tw + 24, th + 12, image.COLOR_BLACK, thickness=-1)
        img.draw_string(x, y, _toast_text, image.COLOR_YELLOW, scale=2.0, thickness=2)


def _adjust_rgb(color, idx, delta):
    vals = current_thresholds[color]
    if idx in (0, 1):
        vals[idx] = _clamp(vals[idx] + delta, 0, 100)
    else:
        vals[idx] = _clamp(vals[idx] + delta, -128, 127)

    pair_start = (idx // 2) * 2
    if vals[pair_start] > vals[pair_start + 1]:
        if idx % 2 == 0:
            vals[pair_start + 1] = vals[pair_start]
        else:
            vals[pair_start] = vals[pair_start + 1]


def _adjust_black(idx, delta):
    vals = current_thresholds["black"]
    vals[idx] = _clamp(vals[idx] + delta, 0, 100)
    if vals[0] > vals[1]:
        if idx == 0:
            vals[1] = vals[0]
        else:
            vals[0] = vals[1]


def _is_repeatable_action(action, current_color):
    if current_color == "black":
        return action in ("gmin_dec", "gmax_dec", "gmin_inc", "gmax_inc")
    return action in (
        "lmin_dec", "lmax_dec", "amin_dec", "amax_dec", "bmin_dec", "bmax_dec",
        "lmin_inc", "lmax_inc", "amin_inc", "amax_inc", "bmin_inc", "bmax_inc",
    )


def _handle_action(action, current_color):
    global _ui_color_idx

    if action == "back":
        return "back"
    if action == "switch":
        _ui_color_idx = (_ui_color_idx + 1) % len(COLOR_ORDER)
        return None
    if action == "reset":
        current_thresholds[current_color] = list(DEFAULT_THRESHOLDS[current_color])
        _show_toast(f"({COLOR_LABELS[current_color]} Rested)")
        return None
    if action == "save":
        save_thresholds()
        _show_toast("(Saved)")
        return None

    if current_color == "black":
        map_black = {
            "gmin_dec": (0, -1),
            "gmax_dec": (1, -1),
            "gmin_inc": (0, 1),
            "gmax_inc": (1, 1),
        }
        if action in map_black:
            idx, delta = map_black[action]
            _adjust_black(idx, delta)
            return None
    else:
        map_rgb = {
            "lmin_dec": (0, -1),
            "lmax_dec": (1, -1),
            "amin_dec": (2, -1),
            "amax_dec": (3, -1),
            "bmin_dec": (4, -1),
            "bmax_dec": (5, -1),
            "lmin_inc": (0, 1),
            "lmax_inc": (1, 1),
            "amin_inc": (2, 1),
            "amax_inc": (3, 1),
            "bmin_inc": (4, 1),
            "bmax_inc": (5, 1),
        }
        if action in map_rgb:
            idx, delta = map_rgb[action]
            _adjust_rgb(current_color, idx, delta)
            return None

    return None


def _find_action_from_touch(tx, ty, buttons):
    for action, btn in buttons.items():
        if _point_in_btn(tx, ty, btn):
            return action
    return None


def run_tuner_ui(img, ts, _disp):
    global _ignore_until_release, _hold_action, _hold_start_time, _hold_last_repeat
    global _preview_cache_data, _preview_cache_color, _preview_cache_key, _preview_cache_time

    layout = _layout()
    current_color = COLOR_ORDER[_ui_color_idx]
    buttons = _build_black_buttons(layout) if current_color == "black" else _build_rgb_buttons(layout)

    now = time.time()

    current_key = tuple(current_thresholds[current_color])
    need_refresh = (
        _preview_cache_data is None
        or _preview_cache_color != current_color
        or _preview_cache_key != current_key
        or (now - _preview_cache_time) >= PREVIEW_REFRESH_INTERVAL
    )
    if need_refresh:
        _preview_cache_data = vision.get_binary_preview_rects(
            img,
            current_color,
            current_thresholds,
            use_mask=True,
            preview_size=PREVIEW_SIZE,
        )
        _preview_cache_color = current_color
        _preview_cache_key = current_key
        _preview_cache_time = now

    img.draw_rect(0, 0, 640, 480, image.COLOR_GRAY, thickness=-1)
    vision.draw_binary_preview(img, _preview_cache_data, layout["preview"])

    for btn in buttons.values():
        _draw_button(img, btn, btn["text"])

    _draw_status_box(img, current_color, layout)
    _draw_toast(img)

    touch_data = ts.read()
    pressed = touch_data and len(touch_data) >= 3 and touch_data[2] == 1

    if not pressed:
        _ignore_until_release = False
        _hold_action = None
        return None

    if _ignore_until_release:
        return None

    tx, ty = touch_data[0], touch_data[1]
    action = _find_action_from_touch(tx, ty, buttons)
    if not action:
        _hold_action = None
        return None

    if action != _hold_action:
        _hold_action = action
        _hold_start_time = now
        _hold_last_repeat = now
        return _handle_action(action, current_color)

    if _is_repeatable_action(action, current_color):
        hold_elapsed = now - _hold_start_time
        repeat_interval = 0.30 if hold_elapsed < 0.70 else 0.08
        if now - _hold_last_repeat >= repeat_interval:
            _hold_last_repeat = now
            return _handle_action(action, current_color)

    return None