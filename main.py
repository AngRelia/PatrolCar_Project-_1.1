# main.py
# 开机默认显示实时画面与 Start 按钮；长按非 Start 区域进入阈值调节界面。

import time
from maix import app, image, camera, display, touchscreen

import tuner
import vision

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
LONG_PRESS_SECONDS = 1.2
HOME_DETECT_INTERVAL = 0.06
HOME_DETECT_SIZE = (320, 240)

_home_markers_cache = []
_home_markers_last_time = 0.0


def init_hardware():
    """初始化摄像头、屏幕和触摸屏。"""
    cam = camera.Camera(TARGET_WIDTH, TARGET_HEIGHT)
    cam.skip_frames(10)
    disp = display.Display()
    ts = touchscreen.TouchScreen()
    return cam, disp, ts


def _start_btn_layout():
    btn_w, btn_h = 180, 64
    btn_x = (TARGET_WIDTH - btn_w) // 2
    btn_y = TARGET_HEIGHT - btn_h - 10
    return {"x": btn_x, "y": btn_y, "w": btn_w, "h": btn_h}


def _is_pressed(touch_data):
    return touch_data and len(touch_data) >= 3 and touch_data[2] == 1


def _point_in_rect(px, py, rect):
    return rect["x"] <= px <= rect["x"] + rect["w"] and rect["y"] <= py <= rect["y"] + rect["h"]


def _draw_start_button(img):
    btn = _start_btn_layout()
    fill = image.COLOR_GREEN
    img.draw_rect(btn["x"], btn["y"], btn["w"], btn["h"], image.COLOR_BLACK, thickness=3)
    img.draw_rect(btn["x"] + 2, btn["y"] + 2, btn["w"] - 4, btn["h"] - 4, fill, thickness=-1)
    img.draw_string(btn["x"] + 48, btn["y"] + 18, "Start", image.COLOR_BLACK, scale=2.2, thickness=2)
    return btn


def _draw_cached_markers(img, markers):
    for m in markers:
        x, y, w, h = m["x"], m["y"], m["w"], m["h"]
        img.draw_rect(x, y, w, h, image.COLOR_WHITE, thickness=2)
        img.draw_string(x, max(0, y - 20), f"{m['color']}-{m['shape']}", image.COLOR_WHITE, scale=1.6, thickness=2)


def _draw_home_overlay(img, thresholds, now):
    global _home_markers_cache, _home_markers_last_time

    if now - _home_markers_last_time >= HOME_DETECT_INTERVAL:
        _home_markers_cache = vision.identify_markers_multi(
            img,
            thresholds,
            draw=False,
            max_results=8,
            detect_size=HOME_DETECT_SIZE,
            fast_mode=True,
        )
        _home_markers_last_time = now

    _draw_cached_markers(img, _home_markers_cache)
    return _draw_start_button(img)


def main():
    cam, disp, ts = init_hardware()
    tuner.load_thresholds()

    state = "HOME"
    touch_hold_start_time = 0.0
    last_frame_time = time.time()
    while not app.need_exit():
        img = cam.read()
        now = time.time()

        if state == "HOME":
            start_btn = _draw_home_overlay(img, tuner.current_thresholds, now)
            touch_data = ts.read()

            if _is_pressed(touch_data):
                tx, ty = touch_data[0], touch_data[1]
                if _point_in_rect(tx, ty, start_btn):
                    touch_hold_start_time = 0.0
                else:
                    if touch_hold_start_time == 0.0:
                        touch_hold_start_time = now
                    elif now - touch_hold_start_time >= LONG_PRESS_SECONDS:
                        state = "TUNER"
                        tuner.enter_tuner()
                        touch_hold_start_time = 0.0
            else:
                touch_hold_start_time = 0.0

        elif state == "TUNER":
            action = tuner.run_tuner_ui(img, ts, disp)
            if action == "back":
                state = "HOME"

        dt = now - last_frame_time
        fps = 1.0 / dt if dt > 0 else 0.0
        last_frame_time = now
        img.draw_rect(6, 6, 126, 34, image.COLOR_BLACK, thickness=-1)
        img.draw_string(12, 12, f"FPS:{fps:.1f}", image.COLOR_GREEN, scale=1.6, thickness=2)

        disp.show(img)


if __name__ == "__main__":
    main()