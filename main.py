# main.py
# 核心功能：开机默认显示实时画面与 Start 按钮；长按非 Start 区域进入颜色/形状阈值调节界面。
# 架构设计：采用简单的状态机（State Machine）模式，分为主界面（HOME）和调参界面（TUNER）。

import time
from maix import app, image, camera, display, touchscreen

# 导入自定义模块（假设已在同级目录下建立）
import tuner   # 负责阈值调节UI与参数管理
import vision  # 负责底层的图像识别与视觉算法

# ==================== 全局常量配置 ====================
TARGET_WIDTH = 640          # 摄像头与屏幕的目标显示宽度
TARGET_HEIGHT = 480         # 摄像头与屏幕的目标显示高度
LONG_PRESS_SECONDS = 1.2    # 触发进入调参界面的长按时间阈值（秒）

# --- 视觉识别性能优化参数 ---
HOME_DETECT_INTERVAL = 0.06 # 主界面视觉识别的执行间隔（秒）。防止每帧都识别导致掉帧，实现类似“节流(Throttling)”的效果
HOME_DETECT_SIZE = (320, 240) # 视觉识别时图像的缩放尺寸。降低分辨率可大幅提高识别帧率(FPS)
HOME_ARROW_DETECT_INTERVAL = 0.12 # 箭头检测频率可更低，进一步释放算力
HOME_ARROW_DETECT_SIZE = (224, 168) # 箭头方向只需粗分辨率即可稳定判定

# ==================== 全局状态变量 ====================
_home_markers_cache = []    # 缓存上一次识别到的目标结果，用于在未执行识别的帧中保持画框显示
_home_markers_last_time = 0.0 # 记录上一次成功执行视觉识别的时间戳
_home_arrow_dir_cache = "UNKNOWN" # 缓存黑色箭头方向，避免每帧重复计算
_home_arrow_last_time = 0.0 # 记录上一次箭头方向判定时间戳


def init_hardware():
    """
    初始化硬件设备：摄像头、LCD显示屏和触摸屏。
    Returns:
        cam, disp, ts: 返回初始化好的摄像头、显示屏和触摸屏对象。
    """
    # 初始化摄像头，指定分辨率
    cam = camera.Camera(TARGET_WIDTH, TARGET_HEIGHT)
    # 跳过前10帧。原因：摄像头刚启动时，自动曝光(AE)和自动白平衡(AWB)尚未稳定，画面可能过曝或偏色
    cam.skip_frames(10)
    
    # 初始化LCD显示屏和触摸屏
    disp = display.Display()
    ts = touchscreen.TouchScreen()
    return cam, disp, ts


def _start_btn_layout():
    """
    计算并返回 Start 按钮在屏幕上的布局坐标和尺寸。
    Returns:
        dict: 包含按钮 x, y, w, h 的字典。按钮水平居中，底部贴边。
    """
    btn_w, btn_h = 180, 64
    btn_x = (TARGET_WIDTH - btn_w) // 2  # 水平居中计算公式
    btn_y = TARGET_HEIGHT - btn_h - 10   # 距离底部 10 像素
    return {"x": btn_x, "y": btn_y, "w": btn_w, "h": btn_h}


def _is_pressed(touch_data):
    """
    判断当前触摸屏是否被按下。
    Args:
        touch_data: touchscreen.read() 返回的数据，通常格式为 [x, y, status]
    Returns:
        bool: 如果被按下返回 True，否则返回 False。
    """
    # touch_data[2] == 1 通常代表按下状态 (按下为1，抬起或无触摸为0)
    return touch_data and len(touch_data) >= 3 and touch_data[2] == 1


def _point_in_rect(px, py, rect):
    """
    碰撞检测：判断一个坐标点 (px, py) 是否在指定的矩形区域内。
    """
    return rect["x"] <= px <= rect["x"] + rect["w"] and rect["y"] <= py <= rect["y"] + rect["h"]


def _draw_start_button(img):
    """
    在图像上绘制 Start 按钮。
    Args:
        img: 当前摄像头帧的图像对象。
    Returns:
        dict: 返回按钮的布局信息，方便后续做点击检测。
    """
    btn = _start_btn_layout()
    fill = image.COLOR_GREEN
    
    # 绘制按钮外边框 (黑色，线宽3)
    img.draw_rect(btn["x"], btn["y"], btn["w"], btn["h"], image.COLOR_BLACK, thickness=3)
    # 绘制按钮内部填充色 (向内缩进2个像素，厚度为-1表示实心填充)
    img.draw_rect(btn["x"] + 2, btn["y"] + 2, btn["w"] - 4, btn["h"] - 4, fill, thickness=-1)
    # 绘制按钮文字 "Start"
    img.draw_string(btn["x"] + 48, btn["y"] + 18, "Start", image.COLOR_BLACK, scale=2.2, thickness=2)
    
    return btn


def _draw_cached_markers(img, markers):
    """
    将识别到的目标（框和文字）绘制到图像上。
    Args:
        img: 图像对象
        markers: 包含识别结果的列表，每个元素应包含 x, y, w, h, color, shape 等信息
    """
    for m in markers:
        x, y, w, h = m["x"], m["y"], m["w"], m["h"]
        # 绘制目标边界框 (白色框)
        img.draw_rect(x, y, w, h, image.COLOR_WHITE, thickness=2)
        # 在框的上方绘制标签信息 (颜色-形状)。max(0, y-20) 防止文字超出版面上边界
        img.draw_string(x, max(0, y - 20), f"{m['color']}-{m['shape']}", image.COLOR_WHITE, scale=1.6, thickness=2)


def _draw_home_overlay(img, thresholds, now):
    """
    主界面的核心绘制和识别逻辑：执行节流的视觉识别，并绘制UI覆盖层（识别框和按钮）。
    Args:
        img: 当前帧图像
        thresholds: 当前使用的颜色阈值字典
        now: 当前时间戳
    Returns:
        dict: 返回 Start 按钮的位置信息供主循环使用
    """
    global _home_markers_cache, _home_markers_last_time, _home_arrow_dir_cache, _home_arrow_last_time

    # 性能优化核心：不要每帧都跑识别算法。
    # 只有当距离上次识别时间超过 HOME_DETECT_INTERVAL 时，才进行真正的图像处理。
    if now - _home_markers_last_time >= HOME_DETECT_INTERVAL:
        # 调用视觉模块进行多目标识别
        _home_markers_cache = vision.identify_markers_multi(
            img,
            thresholds,
            draw=False,                  # 算法内不直接画图，交由外部统一画，避免画面闪烁
            max_results=8,               # 限制最大结果数，节省性能
            detect_size=HOME_DETECT_SIZE,# 降采样后识别，大幅提升速度
            fast_mode=True,              # 启用快速模式（跳过部分精细计算）
        )

        _home_markers_last_time = now    # 更新最后识别时间

    # 箭头方向检测单独节流：不必跟随颜色/形状检测同频运行
    if now - _home_arrow_last_time >= HOME_ARROW_DETECT_INTERVAL:
        arrow_dir = vision.find_triangle_arrow(
            img,
            thresholds,
            draw=False,
            detect_size=HOME_ARROW_DETECT_SIZE,
        )
        _home_arrow_dir_cache = arrow_dir if arrow_dir else "UNKNOWN"
        _home_arrow_last_time = now

    # 无论当前帧有没有跑识别算法，都使用缓存的识别结果画框。这样视觉上是不掉帧的。
    _draw_cached_markers(img, _home_markers_cache)

    # 在主页实时显示箭头方向缓存结果
    arrow_color = image.COLOR_YELLOW
    if _home_arrow_dir_cache == "FORWARD":
        arrow_color = image.COLOR_GREEN
    elif _home_arrow_dir_cache == "BACKWARD":
        arrow_color = image.COLOR_RED
    img.draw_rect(430, 6, 204, 34, image.COLOR_BLACK, thickness=-1)
    img.draw_string(436, 12, f"ARROW:{_home_arrow_dir_cache}", arrow_color, scale=1.35, thickness=2)
    
    # 绘制 Start 按钮
    return _draw_start_button(img)


def main():
    """
    主程序循环。控制硬件初始化、状态机切换、触摸逻辑处理以及帧率计算。
    """
    # 1. 初始化硬件
    cam, disp, ts = init_hardware()
    
    # 2. 从本地存储或默认配置中加载颜色/形状阈值
    tuner.load_thresholds()

    # 3. 初始化状态变量
    state = "HOME"                  # 初始状态为 HOME (主界面)
    touch_hold_start_time = 0.0     # 记录手指按下的初始时间（用于长按检测）
    last_frame_time = time.time()   # 记录上一帧的时间（用于计算FPS）
    
    # 4. 主事件循环 (app.need_exit() 在按下开发板上的退出键或收到退出信号时返回 True)
    while not app.need_exit():
        # 获取摄像头实时画面
        img = cam.read()
        now = time.time() # 获取当前时间戳

        # ================= 状态机：主界面 (HOME) =================
        if state == "HOME":
            # 绘制 UI 并执行识别算法，获取 Start 按钮的位置
            start_btn = _draw_home_overlay(img, tuner.current_thresholds, now)
            # 读取当前触摸屏数据
            touch_data = ts.read()

            # 处理触摸逻辑
            if _is_pressed(touch_data):
                tx, ty = touch_data[0], touch_data[1]
                
                # 情景 1：用户点在了 Start 按钮上
                if _point_in_rect(tx, ty, start_btn):
                    touch_hold_start_time = 0.0 # 清除长按计时，因为按的是按钮
                    # TODO: 这里可以添加点击 Start 按钮后需要执行的业务逻辑
                    # 例如 state = "RUNNING"
                    pass
                
                # 情景 2：用户点在了屏幕空白区域（长按进入调参界面）
                else:
                    # 如果是刚开始按下，记录当前时间
                    if touch_hold_start_time == 0.0:
                        touch_hold_start_time = now
                    # 如果持续按下的时间超过了设定的阈值 (LONG_PRESS_SECONDS)
                    elif now - touch_hold_start_time >= LONG_PRESS_SECONDS:
                        state = "TUNER"         # 切换到调参状态
                        tuner.enter_tuner()     # 通知 tuner 模块准备进入调参 (例如重置 tuner 的内部状态)
                        touch_hold_start_time = 0.0 # 重置长按计时器，防止后续误触发
            
            # 手指抬起，清空长按计时器
            else:
                touch_hold_start_time = 0.0

        # ================= 状态机：调参界面 (TUNER) =================
        elif state == "TUNER":
            # 将画面控制权和触摸控制权交给 tuner 模块处理
            action = tuner.run_tuner_ui(img, ts, disp)
            # 如果 tuner 模块返回 "back" 信号（比如用户按了返回按钮），则退回主界面
            if action == "back":
                state = "HOME"

        # ================= 性能监控与画面刷新 =================
        # 计算两帧之间的时间差 dt，从而求出当前帧率 FPS
        dt = now - last_frame_time
        fps = 1.0 / dt if dt > 0 else 0.0
        last_frame_time = now
        
        # 在左上角绘制 FPS 背景底色框 (避免文字与复杂背景重叠看不清)
        img.draw_rect(6, 6, 126, 34, image.COLOR_BLACK, thickness=-1)
        # 绘制绿色 FPS 文本，保留一位小数
        img.draw_string(12, 12, f"FPS:{fps:.1f}", image.COLOR_GREEN, scale=1.6, thickness=2)

        # 将最终处理好的图像推送至 LCD 屏幕显示
        disp.show(img)


if __name__ == "__main__":
    main()