# vision.py
# 核心功能：基于颜色阈值和连通域（Blob）特征的形状/目标识别。
# 算法设计亮点：
# 1. 启发式多维度评分：不依赖单一特征，而是综合 宽高比(aspect)、填充率(fill)、质心偏移(center_offset)、凸度(convexity) 等打分。
# 2. 动态透视补偿：针对斜视摄像头带来的形变（如圆变椭圆、矩形变梯形）做了动态宽容度调整。
# 3. 性能优先：利用图像降采样、分级步长(stride)查找、以及二值化预览优化，保障在嵌入式设备上的高帧率。

from maix import image

# ==================== 视觉算法全局阈值 ====================
PREVIEW_MIN_W = 64        # 二值化预览图的最小宽度
PREVIEW_MIN_H = 48        # 二值化预览图的最小高度

MIN_PIXELS_MARKER = 220   # 有效目标的最小像素数（滤除噪点）
MIN_BBOX_SIDE = 12        # 边界框（Bounding Box）的最小边长
MIN_FILL_RATIO = 0.16     # 最小填充率（像素数/外接矩形面积）。太低说明可能是线段或中空噪点
MAX_ASPECT_RATIO = 3.4    # 最大长宽比。过滤极度细长的干扰物
EDGE_MARGIN = 3           # 边缘留白：忽略贴着画面边缘的不完整目标


def _threshold_for_color(color_name, current_thresholds):
    """从字典中安全地提取指定颜色的 LAB/灰度阈值列表"""
    vals = current_thresholds.get(color_name)
    if not isinstance(vals, list):
        return None

    if color_name == "black":
        if len(vals) >= 2:
            return [int(vals[0]), int(vals[1])]
        return None

    if len(vals) >= 6:
        return [int(vals[i]) for i in range(6)]
    return None


def find_triangle_arrow(img, current_thresholds):
    """
    [核心业务逻辑]：寻找黑色的三角形箭头，并判断其朝向（TOWARDS/AWAY）。
    物理原理：三角形的“重心（质心）”会偏向底边。
    通过比较【外接矩形的几何中心】和【连通域的真实质心】在 Y 轴上的差值 (dy)，就能推断箭头指向上还是下。
    """
    thresh = _threshold_for_color("black", current_thresholds)
    if not thresh:
        return None

    blobs = img.find_blobs([thresh], pixels_threshold=1000)
    if blobs:
        # 取面积最大的黑色连通域作为箭头主体
        arrow_blob = max(blobs, key=lambda b: b[2] * b[3])

        x, y, w, h = arrow_blob[0], arrow_blob[1], arrow_blob[2], arrow_blob[3]
        pixels = arrow_blob[4]
        area = w * h

        fill_ratio = pixels / area if area > 0 else 0
        # 如果填充率极高(>0.75)，说明是个方形，不是箭头
        if fill_ratio < 0.75:
            mass_cx, mass_cy = arrow_blob[5], arrow_blob[6]  # 真实质心 (通过像素力矩计算)
            box_cx, box_cy = x + w // 2, y + h // 2          # 几何中心 (外接矩形的中点)

            dy = mass_cy - box_cy # 质心相对于几何中心的 Y 轴偏移
            direction = "UNKNOWN"

            # 质心靠下(dy>5)，说明底边在下面，箭头尖端朝上（远离镜头 AWAY）
            if dy > 5:
                direction = "AWAY"
            # 质心靠上(dy<-5)，说明底边在上面，箭头尖端朝下（朝向镜头 TOWARDS）
            elif dy < -5:
                direction = "TOWARDS"

            img.draw_rect(x, y, w, h, image.COLOR_YELLOW, thickness=2)
            img.draw_cross(mass_cx, mass_cy, image.COLOR_RED)
            img.draw_string(x, max(0, y - 30), f"DIR: {direction}", image.COLOR_YELLOW, scale=2)
            return direction
    return None


def _blob_center(blob):
    """安全提取 Blob 质心的兼容写法（适配不同版本的底层固件API）"""
    x, y, w, h = blob[0], blob[1], blob[2], blob[3]
    try:
        cx = blob[5]
    except Exception:
        try:
            cx_getter = getattr(blob, "cx", None)
            cx = cx_getter() if callable(cx_getter) else (x + w // 2)
        except Exception:
            cx = x + w // 2

    try:
        cy = blob[6]
    except Exception:
        try:
            cy_getter = getattr(blob, "cy", None)
            cy = cy_getter() if callable(cy_getter) else (y + h // 2)
        except Exception:
            cy = y + h // 2

    return int(cx), int(cy)


def _shape_band_score(value, lo, hi, peak):
    """
    [核心算法]：带通滤波器式（Band-pass）模糊评分函数。
    原理：给特征值(value)打分。如果 value 等于最佳理想值(peak)，得 1.0 分。
    越偏离 peak 得分越低；一旦超出 [lo, hi] 的容忍区间，直接得 0 分。
    这使得形态学分类比生硬的 `if/else` 判断具有高得多的鲁棒性。
    """
    if value < lo or value > hi:
        return 0.0
    span = max(peak - lo, hi - peak, 1e-6)
    return max(0.0, 1.0 - (abs(value - peak) / span))


def _safe_blob_method(blob, method_name, default_value):
    """安全调用 Blob 对象的底层 C 方法（如 convexity凸度 等），防止固件不支持报错"""
    try:
        method = getattr(blob, method_name, None)
        if callable(method):
            value = method()
            return float(value)
    except Exception:
        pass
    return float(default_value)


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _adaptive_pixels_threshold(img, base_pixels):
    """动态像素阈值：根据当前处理图像的分辨率，自适应缩放必须达到的最小像素数。防止降采样后目标被当做噪点滤除"""
    ref_area = 640 * 480
    area = max(1, img.width() * img.height())
    scaled = int(base_pixels * (area / ref_area))
    return max(60, scaled)


def _blob_valid(blob, img_w, img_h, min_pixels=None):
    """
    [初筛过滤器]：快速剔除明显不符合要求的色块，节省后续复杂的浮点数几何计算。
    """
    x, y, w, h, pixels = blob[0], blob[1], blob[2], blob[3], blob[4]
    if min_pixels is None:
        min_pixels = MIN_PIXELS_MARKER

    # 1. 基础尺寸过滤
    if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE: return False
    if pixels < min_pixels: return False
    
    # 2. 剔除贴边残缺目标
    if y <= EDGE_MARGIN or x <= EDGE_MARGIN: return False
    if x + w >= img_w - EDGE_MARGIN or y + h >= img_h - EDGE_MARGIN: return False

    area = w * h
    if area <= 0: return False
    fill = pixels / area
    if fill < MIN_FILL_RATIO: return False # 太过稀疏（如网格、线框）直接不要

    aspect = w / h if h > 0 else 999
    # 3. 极端长宽比时的【透视宽容度检查】
    if aspect > MAX_ASPECT_RATIO or aspect < (1.0 / MAX_ASPECT_RATIO):
        elong = max(aspect, 1.0 / max(1e-6, aspect))
        convexity = _clamp01(_safe_blob_method(blob, "convexity", 0.0))
        solidity = _clamp01(_safe_blob_method(blob, "solidity", _safe_blob_method(blob, "density", fill)))
        cx, cy = _blob_center(blob)
        center_dx = abs(cx - (x + w / 2.0)) / max(1.0, w)
        center_dy = abs(cy - (y + h / 2.0)) / max(1.0, h)

        # 斜视下目标可出现极端长宽比；若几何特征仍可靠（凸度高、坚实度高），则允许通过（特赦）。
        perspective_ok = (
            elong <= 4.8
            and fill >= 0.12
            and convexity >= 0.46
            and solidity >= 0.34
            and (center_dx + center_dy) <= 0.34
        )
        if not perspective_ok:
            return False
    return True


def _blob_metrics(blob):
    """
    提取连通域的各项核心几何指标字典，用于形状指纹比对。
    包含：填充率(fill), 宽高比(aspect), 质心偏移率(center_offset), 圆度(roundness), 
          凸度(convexity - 轮廓外凸程度), 坚实度(solidity), 透视度(perspective)
    """
    x, y, w, h, pixels = blob[0], blob[1], blob[2], blob[3], blob[4]
    area = max(1, w * h)
    cx, cy = _blob_center(blob)
    
    fill = pixels / area
    aspect = w / h if h > 0 else 999.0
    center_dx = abs(cx - (x + w / 2.0)) / max(1.0, w)
    center_dy = abs(cy - (y + h / 2.0)) / max(1.0, h)
    
    roundness = _clamp01(_safe_blob_method(blob, "roundness", fill * 1.25))
    convexity = _clamp01(_safe_blob_method(blob, "convexity", 0.72))
    solidity = _clamp01(
        _safe_blob_method(
            blob,
            "solidity",
            _safe_blob_method(blob, "density", fill),
        )
    )
    elongation = max(aspect, 1.0 / max(1e-6, aspect))
    # 透视畸变指数：长宽比越极端，质心越偏，认为透视畸变越严重
    perspective = _clamp01(((elongation - 1.0) / 3.0) + ((center_dx + center_dy) * 1.6))
    
    return {
        "x": x, "y": y, "w": w, "h": h, "pixels": pixels,
        "fill": fill, "aspect": aspect, "cx": cx, "cy": cy,
        "center_offset": center_dx + center_dy,
        "roundness": roundness, "convexity": convexity,
        "solidity": solidity, "elongation": elongation,
        "perspective": perspective,
    }


def _apply_perspective_adjustment(m, shape_scores):
    """
    [算法优化：透视补偿]
    当视角极度倾斜时，圆形会变成椭圆，矩形会变成梯形/平行四边形。
    这里检测到高 perspective 畸变时，动态增加各形状在畸变状态下的“投影匹配分数”。
    """
    p = m.get("perspective", 0.0)
    if p <= 0.08:
        return shape_scores # 畸变小，不需要补偿

    # 矩形透视特征：凸度高、坚实度尚可、中心偏移小
    rect_projective = (
        _shape_band_score(m["convexity"], 0.50, 1.00, 0.78)
        * _shape_band_score(m["solidity"], 0.30, 1.00, 0.68)
        * _shape_band_score(m["center_offset"], 0.00, 0.24, 0.06)
        * _shape_band_score(m["fill"], 0.22, 0.92, 0.62)
    )

    # 圆形透视(椭圆)特征：圆度下降，但依然高度外凸，填充率保持稳定
    circle_projective = (
        _shape_band_score(m["roundness"], 0.26, 1.00, 0.68)
        * _shape_band_score(m["convexity"], 0.48, 1.00, 0.76)
        * _shape_band_score(m["center_offset"], 0.00, 0.22, 0.05)
        * _shape_band_score(m["fill"], 0.22, 0.88, 0.70)
        * (1.0 - _shape_band_score(m["solidity"], 0.74, 1.00, 0.90))
    )

    tri_projective = (
        _shape_band_score(m["fill"], 0.12, 0.76, 0.38)
        * _shape_band_score(m["center_offset"], 0.08, 0.56, 0.24)
        * _shape_band_score(m["convexity"], 0.28, 1.00, 0.68)
        * (1.0 - _shape_band_score(m["solidity"], 0.74, 1.00, 0.92))
    )

    # 将畸变补偿加到总分里
    boost = min(0.22, p * 0.24)
    shape_scores["rectangle"] += boost * rect_projective
    shape_scores["circle"] += boost * circle_projective
    shape_scores["triangle"] += boost * tri_projective
    return shape_scores


def _classify_shape_metrics(m):
    """
    [多维混合评分池]
    根据 blob_metrics 输出的各项指标，对矩形、圆形、三角形进行加权打分，最后决出胜者。
    使用了互相惩罚(penalty)的机制防止误判。
    """
    aspect = m["aspect"]
    fill = m["fill"]
    center_offset = m["center_offset"]
    roundness = m["roundness"]
    convexity = m["convexity"]
    solidity = m["solidity"]

    # 高solidity(坚实度)的形状不太可能是圆，penalize circle evidence
    solidity_penalty = max(0.0, min(1.0, (solidity - 0.75) / 0.15)) if solidity > 0.75 else 0.0
    circle_like_evidence = (
        _shape_band_score(fill, 0.58, 0.92, 0.78)
        * _shape_band_score(center_offset, 0.00, 0.16, 0.04)
        * (1.0 - 0.40 * solidity_penalty)  # 高solidity时减弱圆形证据
    )
    triangle_like_evidence = (
        _shape_band_score(fill, 0.12, 0.78, 0.42)
        * _shape_band_score(center_offset, 0.08, 0.56, 0.22)
        * _shape_band_score(convexity, 0.30, 1.00, 0.68)
    )

    # 各形状基础评分权重配比
    score_rect = (
        0.24 * _shape_band_score(aspect, 0.35, 2.90, 1.00)
        + 0.26 * _shape_band_score(fill, 0.40, 0.98, 0.72)
        + 0.16 * _shape_band_score(center_offset, 0.00, 0.28, 0.06)
        + 0.18 * _shape_band_score(solidity, 0.45, 1.00, 0.86)
        + 0.16 * _shape_band_score(roundness, 0.00, 0.78, 0.40)
        - 0.22 * circle_like_evidence   # 如果它太像圆了，就扣矩形分
        - 0.16 * triangle_like_evidence # 如果太像三角了，就扣矩形分
    )

    score_circle = (
        0.20 * _shape_band_score(aspect, 0.28, 3.30, 1.00)
        + 0.34 * _shape_band_score(fill, 0.22, 0.88, 0.76)
        + 0.16 * _shape_band_score(roundness, 0.10, 1.00, 0.70)
        + 0.30 * _shape_band_score(center_offset, 0.00, 0.20, 0.04)
    )

    score_triangle = (
        0.27 * _shape_band_score(aspect, 0.32, 3.10, 1.00)
        + 0.33 * _shape_band_score(fill, 0.12, 0.76, 0.40)
        + 0.25 * _shape_band_score(center_offset, 0.08, 0.48, 0.20)
        + 0.15 * _shape_band_score(convexity, 0.32, 1.00, 0.70)
        + 0.14 * triangle_like_evidence
    )

    shape_scores = {
        "rectangle": score_rect,
        "circle": score_circle,
        "triangle": score_triangle,
    }
    # 追加透视补偿
    shape_scores = _apply_perspective_adjustment(m, shape_scores)

    best_shape = max(shape_scores, key=shape_scores.get)
    sorted_scores = sorted(shape_scores.values(), reverse=True)
    best_score = sorted_scores[0]
    second_score = sorted_scores[1]
    confidence = best_score - second_score # 置信度：第一名比第二名高出多少分

    # 如果最好成绩都太低，或者冠亚军咬得太死无法区分，则标记为 unknown
    if best_score < 0.22 or confidence < 0.03:
        return "unknown", confidence, best_score
    return best_shape, confidence, best_score


def _classify_shape(blob):
    return _classify_shape_metrics(_blob_metrics(blob))[:2]


# ------------- 下面三个 my_find_* 是针对特定形状预过滤分支，逻辑大同小异 -------------
def my_find_triangle(blobs, img_w, img_h, min_pixels=None):
    results = []
    for blob in blobs:
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels): continue
        m = _blob_metrics(blob)
        if not (0.12 <= m["fill"] <= 0.80 and 0.30 <= m["aspect"] <= 3.20): continue
        if m["center_offset"] < 0.08 and m["convexity"] > 0.62:
            # 冲突排解：菱形(转了45度的矩形)在 axis-aligned(平行坐标轴) bbox 下，
            # 填充率偏低(接近0.5)，容易像三角形，但菱形的质心非常居中，凸度高。这里利用质心提前剔除。
            continue
        score = (
            0.32 * _shape_band_score(m["fill"], 0.12, 0.80, 0.40)
            + 0.28 * _shape_band_score(m["aspect"], 0.30, 3.20, 1.00)
            + 0.22 * _shape_band_score(m["center_offset"], 0.08, 0.48, 0.19)
            + 0.18 * _shape_band_score(m["convexity"], 0.32, 1.00, 0.70)
        )
        m["shape"] = "triangle"
        m["score"] = score + (m["perspective"] * 0.05)
        results.append(m)
    return results


def my_find_rectangle(blobs, img_w, img_h, min_pixels=None):
    results = []
    for blob in blobs:
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels): continue
        m = _blob_metrics(blob)
        if not (0.28 <= m["fill"] <= 0.99 and 0.35 <= m["aspect"] <= 2.95): continue
        if (
            m["fill"] <= 0.66
            and m["center_offset"] >= 0.11
            and m["convexity"] >= 0.54
            and m["roundness"] <= 0.74
        ):
            # 典型三角形几何特征，提前过滤避免被本分支强行识别为矩形。
            continue
        score = (
            0.24 * _shape_band_score(m["fill"], 0.28, 0.99, 0.70)
            + 0.22 * _shape_band_score(m["aspect"], 0.35, 2.95, 1.00)
            + 0.18 * _shape_band_score(m["center_offset"], 0.00, 0.28, 0.06)
            + 0.18 * _shape_band_score(m["solidity"], 0.45, 1.00, 0.86)
            + 0.18 * _shape_band_score(m["roundness"], 0.00, 0.78, 0.40)
        )
        m["shape"] = "rectangle"
        m["score"] = score + (m["perspective"] * 0.06)
        results.append(m)
    return results


def my_find_circle(blobs, img_w, img_h, min_pixels=None):
    results = []
    for blob in blobs:
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels): continue
        m = _blob_metrics(blob)
        if not (0.22 <= m["fill"] <= 0.88 and 0.28 <= m["aspect"] <= 3.20): continue
        if m["fill"] >= 0.82 and m["center_offset"] <= 0.12 and m["solidity"] >= 0.76:
            # 高填充且质心极度居中的目标更像完美的正方形，避免被圆形分支吸走。
            continue
        score = (
            0.34 * _shape_band_score(m["fill"], 0.22, 0.88, 0.76)
            + 0.20 * _shape_band_score(m["aspect"], 0.28, 3.20, 1.00)
            + 0.16 * _shape_band_score(m["roundness"], 0.10, 1.00, 0.70)
            + 0.30 * _shape_band_score(m["center_offset"], 0.00, 0.20, 0.04)
        )
        m["shape"] = "circle"
        m["score"] = score + (m["perspective"] * 0.06)
        results.append(m)
    return results


def my_find_perspective_shapes(blobs, img_w, img_h, min_pixels=None):
    """全局后备探测：用全量特征矩阵对所有 blob 再进行一次不偏科的分类检查"""
    results = []
    for blob in blobs:
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels): continue
        m = _blob_metrics(blob)
        shape, confidence, best_score = _classify_shape_metrics(m)
        if shape == "unknown":
            continue

        # 斜视下 bbox 常拉伸，给透视形变留容错 bonus。
        perspective_skew = abs((m["aspect"] - 1.0))
        perspective_bonus = min(0.08, perspective_skew * 0.03)

        m["shape"] = shape
        m["score"] = (best_score * 0.90) + (confidence * 0.60) + perspective_bonus
        results.append(m)
    return results


def _find_color_blobs(img, thresh, min_pixels=None, fast_mode=True):
    """
    [性能优化]：多级容错搜寻连通域。
    先用 fast_mode (大步长 x_stride/y_stride=2, 寻找速度快一倍) 进行粗搜。
    如果没找到，降低条件，使用小步长或放宽合并边缘(margin)再次查找。
    """
    if min_pixels is None:
        min_pixels = _adaptive_pixels_threshold(img, MIN_PIXELS_MARKER)
    relaxed_pixels = max(45, min_pixels // 2) # 放宽的次级像素要求

    if fast_mode:
        primary_stride, relaxed_stride = 2, 2
        primary_margin, relaxed_margin = 2, 3
    else:
        primary_stride, relaxed_stride = 1, 1
        primary_margin, relaxed_margin = 3, 4

    try:
        # 第一阶段：严格且快速搜索
        blobs = img.find_blobs(
            [thresh], pixels_threshold=min_pixels, area_threshold=min_pixels,
            x_stride=primary_stride, y_stride=primary_stride, merge=True, margin=primary_margin,
        )
        if blobs: return blobs
    except Exception:
        pass

    try:
        # 第二阶段：放宽像素要求与融合间距
        blobs = img.find_blobs(
            [thresh], pixels_threshold=relaxed_pixels, area_threshold=relaxed_pixels,
            x_stride=relaxed_stride, y_stride=relaxed_stride, merge=True, margin=relaxed_margin,
        )
        if blobs: return blobs
    except Exception:
        pass

    # 兜底阶段
    try:
        return img.find_blobs([thresh], pixels_threshold=relaxed_pixels)
    except Exception:
        return []


def _box_iou(a, b):
    """计算两个边界框的 IOU (交并比)。用于去除同一目标的重复识别框 (NMS, 非极大值抑制)"""
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0: return 0.0

    inter = iw * ih
    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pick_best_shape_hit(hits):
    """
    [冲突裁决核心]
    同一个色块可能同时被 圆形/矩形/三角形 判定分支抓取。
    此函数的作用是分析这些分支的得分，进行逻辑上的终极甄别和纠偏。
    """
    if not hits: return None

    # 1. 按形状聚合命中分数，降低单分支抖动导致的误判（计算群智平均分）。
    merged_by_shape = {}
    for hit in hits:
        shape = hit.get("shape", "unknown")
        if shape == "unknown": continue
        merged_by_shape.setdefault(shape, []).append(hit)

    aggregated = []
    for shape, shape_hits in merged_by_shape.items():
        base = max(shape_hits, key=lambda h: h.get("score", 0.0))
        avg_score = sum(h.get("score", 0.0) for h in shape_hits) / max(1, len(shape_hits))
        support_bonus = min(0.18, 0.08 * (len(shape_hits) - 1)) # 同形状被触发多次则加分
        h = dict(base)
        h["score"] = base.get("score", 0.0) + (avg_score * 0.12) + support_bonus
        aggregated.append(h)

    if not aggregated: return None

    hits_sorted = sorted(aggregated, key=lambda h: h["score"], reverse=True)
    best = hits_sorted[0]

    circle_hit, rectangle_hit, triangle_hit = None, None, None
    for hit in hits_sorted:
        if hit.get("shape") == "circle": circle_hit = hit
        elif hit.get("shape") == "rectangle": rectangle_hit = hit
        elif hit.get("shape") == "triangle": triangle_hit = hit

    # --- 开始硬核冲突裁决 ---
    
    # 冲突 1：三角形 vs 矩形。
    # 痛点：矩形倾斜变成菱形后，很容易被错认为三角形。
    if triangle_hit and rectangle_hit:
        tri_fill, tri_offset = triangle_hit.get("fill", 0.0), triangle_hit.get("center_offset", 1.0)
        tri_convexity, tri_aspect = triangle_hit.get("convexity", 0.0), triangle_hit.get("aspect", 1.0)

        # 验证这是否是一个非常强烈的三角形特征
        tri_geometry_strong = (0.14 <= tri_fill <= 0.74 and tri_offset >= 0.10 and tri_convexity >= 0.50 and 0.32 <= tri_aspect <= 3.20)
        # 验证这是否是一个存在风险（低填充率+质心偏离）的假矩形
        rect_triangle_risk = (rectangle_hit.get("fill", 0.0) <= 0.70 and rectangle_hit.get("center_offset", 1.0) >= 0.10)
        
        close_score = triangle_hit["score"] >= (rectangle_hit["score"] * 0.83)
        if tri_geometry_strong and rect_triangle_risk and close_score:
            return triangle_hit

    # 冲突 2：矩形 vs 圆形
    if rectangle_hit and circle_hit:
        rect_fill, rect_offset = rectangle_hit.get("fill", 0.0), rectangle_hit.get("center_offset", 1.0)
        rect_solidity, rect_aspect = rectangle_hit.get("solidity", 0.0), rectangle_hit.get("aspect", 1.0)
        
        # 完美的方块特征识别
        rect_like_square = (rect_fill >= 0.84 and rect_offset <= 0.11 and rect_solidity >= 0.78 and 0.42 <= rect_aspect <= 2.40)
        # 倾斜的矩形特征：高 solidity 是强力证据
        rect_like_tilted = (rect_solidity >= 0.76 and 0.40 <= rect_fill <= 0.88 and rect_offset <= 0.18 and 0.42 <= rect_aspect <= 2.50
                            and not (rect_fill <= 0.70 and rect_offset >= 0.10))
        
        close_score = rectangle_hit["score"] >= (circle_hit["score"] * 0.76)
        if (rect_like_square or rect_like_tilted) and close_score:
            return rectangle_hit

    # 冲突 3：圆被透视压扁后会像椭圆，容易丢分，这里手动捞回。
    if circle_hit:
        circle_fill, circle_offset = circle_hit.get("fill", 0.0), circle_hit.get("center_offset", 1.0)
        circle_solidity = circle_hit.get("solidity", 0.0)
        
        if circle_solidity >= 0.76:
            pass  # solidity太高，说明有坚实的棱角，不是圆，跳过强捞
        else:
            # 不依赖 roundness 的兜底：圆/椭圆在 bbox 内通常填充率接近 pi/4 (约0.785)，且质心极其居中。
            geometric_circle = (0.60 <= circle_fill <= 0.85 and circle_offset <= 0.11 and circle_hit.get("roundness", 0.0) >= 0.30 and 0.35 <= circle_hit.get("aspect", 1.0) <= 2.85)
            if geometric_circle and circle_hit["score"] >= (best["score"] * 0.70):
                return circle_hit

        ellipse_like = (circle_hit.get("roundness", 0.0) >= 0.46 and circle_hit.get("convexity", 0.0) >= 0.58 and circle_offset <= 0.18 and 0.30 <= circle_fill <= 0.88 and 0.35 <= circle_hit.get("aspect", 1.0) <= 2.80)
        close_score = circle_hit["score"] >= (best["score"] * 0.74)
        if ellipse_like and close_score:
            return circle_hit

    # 冲突 4：矩形转成菱形时容易被当做三角。
    if rectangle_hit and triangle_hit:
        diamond_like = (rectangle_hit.get("center_offset", 1.0) <= 0.10 and rectangle_hit.get("convexity", 0.0) >= 0.62 and rectangle_hit.get("solidity", 0.0) >= 0.46 and 0.28 <= rectangle_hit.get("fill", 0.0) <= 0.82 and 0.40 <= rectangle_hit.get("aspect", 1.0) <= 2.60)
        close_score = rectangle_hit["score"] >= (triangle_hit["score"] * 0.90)
        if diamond_like and close_score:
            return rectangle_hit

    return best


def _map_hit_to_original(hit, src_w, src_h, dst_w, dst_h):
    """由于算法可能在降采样的低分辨率图上运行，结束后需要将坐标(x,y,w,h)映射回原图大尺度"""
    x = int((hit["x"] * dst_w) / max(1, src_w))
    y = int((hit["y"] * dst_h) / max(1, src_h))
    w = int((hit["w"] * dst_w) / max(1, src_w))
    h = int((hit["h"] * dst_h) / max(1, src_h))

    # 安全限幅
    x = max(0, min(dst_w - 1, x))
    y = max(0, min(dst_h - 1, y))
    w = max(1, w)
    h = max(1, h)
    if x + w > dst_w: w = max(1, dst_w - x)
    if y + h > dst_h: h = max(1, dst_h - y)
    return {"x": x, "y": y, "w": w, "h": h}


def identify_markers_multi(img, current_thresholds, draw=True, max_results=8, detect_size=(320, 240), fast_mode=True):
    """
    [多目标/多颜色综合识别入口]：被 main.py 主界面调用。
    流程：缩放图像 -> 遍历三种颜色 -> 寻找各形状得分最高的 Blob -> IoU 去重过滤 -> 坐标还原 -> 返回结果列表。
    """
    colors_to_check = ["red", "green", "blue"]
    candidates = []
    
    img_w, img_h = img.width(), img.height()

    # [性能优化关键]：如果设定了 detect_size (如 320x240)，先将画面缩小。
    # 面积变为原先的1/4，搜索连通域的计算量直接缩减75%以上，极大提升FPS！
    work_img = img
    work_w, work_h = img_w, img_h
    if isinstance(detect_size, (tuple, list)) and len(detect_size) >= 2:
        target_w = max(80, min(img_w, int(detect_size[0])))
        target_h = max(60, min(img_h, int(detect_size[1])))
        if target_w < img_w or target_h < img_h:
            try:
                work_img = img.resize(target_w, target_h, method=image.ResizeMethod.NEAREST)
                work_w, work_h = work_img.width(), work_img.height()
            except Exception:
                work_img, work_w, work_h = img, img_w, img_h

    detect_min_pixels = _adaptive_pixels_threshold(work_img, MIN_PIXELS_MARKER)

    # 遍历红绿蓝，寻找对应色块并定形
    for color in colors_to_check:
        thresh = _threshold_for_color(color, current_thresholds)
        if not thresh: continue

        blobs = _find_color_blobs(work_img, thresh, min_pixels=detect_min_pixels, fast_mode=fast_mode)
        if not blobs: continue

        shape_hits = []
        shape_hits.extend(my_find_triangle(blobs, work_w, work_h, min_pixels=detect_min_pixels))
        shape_hits.extend(my_find_rectangle(blobs, work_w, work_h, min_pixels=detect_min_pixels))
        shape_hits.extend(my_find_circle(blobs, work_w, work_h, min_pixels=detect_min_pixels))
        shape_hits.extend(my_find_perspective_shapes(blobs, work_w, work_h, min_pixels=detect_min_pixels))

        # 同一个连通域(blob)可能会被多种形状算法捕获，需要按坐标聚合
        hits_per_blob = {}
        for hit in shape_hits:
            key = (hit["x"], hit["y"], hit["w"], hit["h"])
            hits_per_blob.setdefault(key, []).append(hit)

        # 裁定每一个色块到底是什么形状
        for hits in hits_per_blob.values():
            hit = _pick_best_shape_hit(hits)
            if not hit: continue

            mapped = _map_hit_to_original(hit, work_w, work_h, img_w, img_h)
            # 综合置信度得分 (形状打分 + 面积加权)
            score = (hit["score"] * 2.0) + (hit["pixels"] / 3500.0) + (hit["fill"] * 0.5)
            candidates.append({
                "color": color, "shape": hit["shape"],
                "x": mapped["x"], "y": mapped["y"], "w": mapped["w"], "h": mapped["h"],
                "score": score,
            })

    if not candidates: return []

    # [NMS 非极大值抑制处理]：
    # 按照综合得分从高到低排序，如果两个框严重重合（IoU > 0.35），说明识别了同一个物体，保留高分，踢掉低分。
    candidates.sort(key=lambda item: item["score"], reverse=True)
    kept = []
    for item in candidates:
        overlapped = False
        for k in kept:
            if _box_iou(item, k) > 0.35:
                overlapped = True
                break
        if not overlapped:
            kept.append(item)
            if len(kept) >= max_results:
                break

    if draw:
        for m in kept:
            x, y, w, h = m["x"], m["y"], m["w"], m["h"]
            img.draw_rect(x, y, w, h, image.COLOR_WHITE, thickness=2)
            img.draw_string(x, max(0, y - 20), f"{m['color']}-{m['shape']}", image.COLOR_WHITE, scale=1.6, thickness=2)

    return [{"color": m["color"], "shape": m["shape"], "x": m["x"], "y": m["y"], "w": m["w"], "h": m["h"]} for m in kept]


def identify_markers(img, current_thresholds, draw=True):
    """单目标识别向后兼容接口，只返回得分最高的第一个结果"""
    results = identify_markers_multi(img, current_thresholds, draw=draw, max_results=8)
    if results:
        return {"color": results[0]["color"], "shape": results[0]["shape"]}
    return None


def _pixel_is_foreground(pixel):
    """判断一个像素在二值化(或彩色)图像中是否属于前景有效区（白点）"""
    if isinstance(pixel, int): return pixel > 0
    if isinstance(pixel, (tuple, list)):
        if len(pixel) == 0: return False
        if len(pixel) == 1: return int(pixel[0]) > 0
        return (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) > 0
    return False


def _normalize_preview_size(src_w, src_h, preview_size):
    """固定调参界面二值化预览图的分辨率大小，并保持正确的长宽比"""
    if isinstance(preview_size, (tuple, list)) and len(preview_size) >= 2:
        w, h = int(preview_size[0]), int(preview_size[1])
    else:
        w, h = src_w // 4, src_h // 4

    w = max(PREVIEW_MIN_W, min(src_w, w))
    h = max(PREVIEW_MIN_H, min(src_h, h))

    src_ratio = src_w / src_h if src_h > 0 else 1.0
    if src_ratio > 0:
        h_by_w = max(PREVIEW_MIN_H, int(round(w / src_ratio)))
        if h_by_w <= src_h:
            h = h_by_w
        else:
            h = src_h
            w = max(PREVIEW_MIN_W, min(src_w, int(round(h * src_ratio))))
    return w, h


def _draw_mask_to_roi(dst_img, mask_img, roi):
    """
    手动映射并绘制二值化掩码图。
    在某些底层库中，低分辨率二值图无法直接通过 API 缩放贴到高分辨率彩色图上，
    此函数采用游程编码(Run-length encoding)的思想逐行绘制白色块。
    """
    rx, ry, rw, rh = roi
    mw, mh = mask_img.width(), mask_img.height()
    if mw <= 0 or mh <= 0: return

    row_h = max(1, (rh + mh - 1) // mh)
    for sy in range(mh):
        py = ry + (sy * rh) // mh
        sx = 0
        while sx < mw:
            # 找黑色段（背景）
            while sx < mw and not _pixel_is_foreground(mask_img.get_pixel(sx, sy)):
                sx += 1
            if sx >= mw: break

            # 找白色段（前景）
            run_start = sx
            while sx < mw and _pixel_is_foreground(mask_img.get_pixel(sx, sy)):
                sx += 1

            px1 = rx + (run_start * rw) // mw
            px2 = rx + (sx * rw) // mw
            run_w = max(1, px2 - px1)
            # 一次性画满整个白色长条块（性能远高于画一个个点）
            dst_img.draw_rect(px1, py, run_w, row_h, image.COLOR_WHITE, thickness=-1)


def get_binary_preview_rects(src_img, target_name, current_thresholds, use_mask=False, preview_size=None):
    """
    [性能瓶颈攻坚]：生成调参页的黑白二值预览图数据。
    如果全分辨率进行 `.binary()` 操作极其耗时（会让调参UI卡死）。
    优化：将原图强行缩小至 128x96 或更低，再进行 `.binary()`，最后给到 UI 去做拉伸渲染。
    """
    thresh = _threshold_for_color(target_name, current_thresholds)
    if not thresh:
        return {"mode": "empty", "rects": []}

    if not use_mask:
        # 只返回识别框的情况（省去全屏二值化步骤）
        blobs = []
        try:
            blobs = src_img.find_blobs([thresh], pixels_threshold=80)
        except Exception:
            pass
        rects = []
        if blobs:
            blobs_sorted = sorted(blobs, key=lambda b: b[2] * b[3], reverse=True)
            for blob in blobs_sorted[:24]:
                if blob[2] * blob[3] < 90: continue
                rects.append((blob[0], blob[1], blob[2], blob[3]))
        return {"mode": "rects", "rects": rects}

    # [二值化性能优化核心] 先降分辨率再二值化，可显著提升调参界面的按键响应与画面刷新率。
    try:
        src_w, src_h = src_img.width(), src_img.height()
        low_w, low_h = _normalize_preview_size(src_w, src_h, preview_size)
        low_img = src_img.resize(low_w, low_h, method=image.ResizeMethod.NEAREST)
        low_img.binary([thresh]) # 在小图上跑耗时操作
        return {"mode": "mask_lowres", "mask": low_img, "rects": []}
    except Exception:
        pass

    # 兼容回退：如果固件不支持小图resize后运算，退回原尺寸耗时操作。
    try:
        mask_img = src_img.copy()
        mask_img.binary([thresh])
        return {"mode": "mask", "mask": mask_img, "rects": []}
    except Exception:
        pass

    return {"mode": "rects", "rects": []}


def draw_binary_preview(dst_img, preview_data, roi):
    """在调参界面的特定方框 (ROI) 区域内，渲染前面生成的二值化预览数据"""
    img_w, img_h = dst_img.width(), dst_img.height()
    rx, ry, rw, rh = roi
    
    # 画背景黑框
    dst_img.draw_rect(rx, ry, rw, rh, image.COLOR_BLACK, thickness=4)
    dst_img.draw_rect(rx + 2, ry + 2, rw - 4, rh - 4, image.COLOR_BLACK, thickness=-1)

    rects = preview_data.get("rects", []) if isinstance(preview_data, dict) else preview_data

    # 如果有掩码图（二值图）
    if isinstance(preview_data, dict) and preview_data.get("mode") in ("mask", "mask_lowres"):
        mask_img = preview_data.get("mask")
        if mask_img:
            try:
                # 尝试通过硬件加速的方法拉伸回去并贴图
                scaled_mask = mask_img.resize(rw, rh, method=image.ResizeMethod.NEAREST)
                dst_img.draw_image(rx, ry, scaled_mask)
                return
            except Exception:
                try:
                    # 如果不支持，启用手动CPU渲染兜底方案
                    _draw_mask_to_roi(dst_img, mask_img, roi)
                    return
                except Exception:
                    pass

    # 回退模式：如果没有掩码图，只画白色的包围方块（示意大概位置）
    for bx, by, bw, bh in rects:
        px = rx + (bx * rw) // img_w
        py = ry + (by * rh) // img_h
        pw = max(1, (bw * rw) // img_w)
        ph = max(1, (bh * rh) // img_h)
        dst_img.draw_rect(px, py, pw, ph, image.COLOR_WHITE, thickness=-1)


def draw_preview(img, target_name, current_thresholds):
    """简单的辅助工具，在图像上画出某个颜色最大的连通域"""
    thresh = _threshold_for_color(target_name, current_thresholds)
    if not thresh:
        return

    blobs = img.find_blobs([thresh], pixels_threshold=400)
    if blobs:
        b = max(blobs, key=lambda blob: blob[2] * blob[3])
        img.draw_rect(b[0], b[1], b[2], b[3], image.COLOR_WHITE, thickness=2)