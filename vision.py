from maix import image

PREVIEW_MIN_W = 64
PREVIEW_MIN_H = 48

MIN_PIXELS_MARKER = 220
MIN_BBOX_SIDE = 12
MIN_FILL_RATIO = 0.16
MAX_ASPECT_RATIO = 3.4
EDGE_MARGIN = 3


def _threshold_for_color(color_name, current_thresholds):
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



def _blob_center(blob):
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
    if value < lo or value > hi:
        return 0.0
    span = max(peak - lo, hi - peak, 1e-6)
    return max(0.0, 1.0 - (abs(value - peak) / span))


def _safe_blob_method(blob, method_name, default_value):
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
    ref_area = 640 * 480
    area = max(1, img.width() * img.height())
    scaled = int(base_pixels * (area / ref_area))
    return max(60, scaled)


def _blob_valid(blob, img_w, img_h, min_pixels=None):
    x, y, w, h, pixels = blob[0], blob[1], blob[2], blob[3], blob[4]
    if min_pixels is None:
        min_pixels = MIN_PIXELS_MARKER

    if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
        return False
    if pixels < min_pixels:
        return False
    if y <= EDGE_MARGIN or x <= EDGE_MARGIN:
        return False
    if x + w >= img_w - EDGE_MARGIN or y + h >= img_h - EDGE_MARGIN:
        return False

    area = w * h
    if area <= 0:
        return False
    fill = pixels / area
    if fill < MIN_FILL_RATIO:
        return False

    aspect = w / h if h > 0 else 999
    if aspect > MAX_ASPECT_RATIO or aspect < (1.0 / MAX_ASPECT_RATIO):
        elong = max(aspect, 1.0 / max(1e-6, aspect))
        convexity = _clamp01(_safe_blob_method(blob, "convexity", 0.0))
        solidity = _clamp01(_safe_blob_method(blob, "solidity", _safe_blob_method(blob, "density", fill)))
        cx, cy = _blob_center(blob)
        center_dx = abs(cx - (x + w / 2.0)) / max(1.0, w)
        center_dy = abs(cy - (y + h / 2.0)) / max(1.0, h)

        # 斜视下目标可出现极端长宽比；若几何特征仍可靠，则允许通过。
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
    perspective = _clamp01(((elongation - 1.0) / 3.0) + ((center_dx + center_dy) * 1.6))
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "pixels": pixels,
        "fill": fill,
        "aspect": aspect,
        "cx": cx,
        "cy": cy,
        "center_offset": center_dx + center_dy,
        "roundness": roundness,
        "convexity": convexity,
        "solidity": solidity,
        "elongation": elongation,
        "perspective": perspective,
    }


def _apply_perspective_adjustment(m, shape_scores):
    p = m.get("perspective", 0.0)
    if p <= 0.08:
        return shape_scores

    rect_projective = (
        _shape_band_score(m["convexity"], 0.50, 1.00, 0.78)
        * _shape_band_score(m["solidity"], 0.30, 1.00, 0.68)
        * _shape_band_score(m["center_offset"], 0.00, 0.24, 0.06)
        * _shape_band_score(m["fill"], 0.22, 0.92, 0.62)
    )

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

    boost = min(0.22, p * 0.24)
    shape_scores["rectangle"] += boost * rect_projective
    shape_scores["circle"] += boost * circle_projective
    shape_scores["triangle"] += boost * tri_projective
    return shape_scores


def _classify_shape_metrics(m):
    aspect = m["aspect"]
    fill = m["fill"]
    center_offset = m["center_offset"]
    roundness = m["roundness"]
    convexity = m["convexity"]
    solidity = m["solidity"]

    # 高solidity的形状不太可能是圆，penalize circle evidence
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

    score_rect = (
        0.24 * _shape_band_score(aspect, 0.35, 2.90, 1.00)
        + 0.26 * _shape_band_score(fill, 0.40, 0.98, 0.72)
        + 0.16 * _shape_band_score(center_offset, 0.00, 0.28, 0.06)
        + 0.18 * _shape_band_score(solidity, 0.45, 1.00, 0.86)
        + 0.16 * _shape_band_score(roundness, 0.00, 0.78, 0.40)
        - 0.22 * circle_like_evidence
        - 0.16 * triangle_like_evidence
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
    shape_scores = _apply_perspective_adjustment(m, shape_scores)

    best_shape = max(shape_scores, key=shape_scores.get)
    sorted_scores = sorted(shape_scores.values(), reverse=True)
    best_score = sorted_scores[0]
    second_score = sorted_scores[1]
    confidence = best_score - second_score

    if best_score < 0.22 or confidence < 0.03:
        return "unknown", confidence, best_score
    return best_shape, confidence, best_score



def my_find_triangle(blobs, img_w, img_h, min_pixels=None):
    results = []
    for blob in blobs:
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels):
            continue
        m = _blob_metrics(blob)
        if not (0.12 <= m["fill"] <= 0.80 and 0.30 <= m["aspect"] <= 3.20):
            continue
        if m["center_offset"] < 0.08 and m["convexity"] > 0.62:
            # 菱形矩形在 axis-aligned bbox 下填充率偏低，容易像三角形，这里提前剔除。
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
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels):
            continue
        m = _blob_metrics(blob)
        if not (0.28 <= m["fill"] <= 0.99 and 0.35 <= m["aspect"] <= 2.95):
            continue
        if (
            m["fill"] <= 0.66
            and m["center_offset"] >= 0.11
            and m["convexity"] >= 0.54
            and m["roundness"] <= 0.74
        ):
            # 典型三角形几何特征，避免被矩形分支抢走。
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
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels):
            continue
        m = _blob_metrics(blob)
        if not (0.22 <= m["fill"] <= 0.88 and 0.28 <= m["aspect"] <= 3.20):
            continue
        if m["fill"] >= 0.82 and m["center_offset"] <= 0.12 and m["solidity"] >= 0.76:
            # 高填充且质心居中的目标更像矩形/正方形，避免被圆形分支吸走。
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
    results = []
    for blob in blobs:
        if not _blob_valid(blob, img_w, img_h, min_pixels=min_pixels):
            continue
        m = _blob_metrics(blob)
        shape, confidence, best_score = _classify_shape_metrics(m)
        if shape == "unknown":
            continue

        # 斜视下 bbox 常拉伸，适当给透视形变留容错。
        perspective_skew = abs((m["aspect"] - 1.0))
        perspective_bonus = min(0.08, perspective_skew * 0.03)

        m["shape"] = shape
        m["score"] = (best_score * 0.90) + (confidence * 0.60) + perspective_bonus
        results.append(m)
    return results


def _find_color_blobs(img, thresh, min_pixels=None, fast_mode=True):
    if min_pixels is None:
        min_pixels = _adaptive_pixels_threshold(img, MIN_PIXELS_MARKER)
    relaxed_pixels = max(45, min_pixels // 2)

    if fast_mode:
        primary_stride = 2
        relaxed_stride = 2
        primary_margin = 2
        relaxed_margin = 3
    else:
        primary_stride = 1
        relaxed_stride = 1
        primary_margin = 3
        relaxed_margin = 4

    try:
        blobs = img.find_blobs(
            [thresh],
            pixels_threshold=min_pixels,
            area_threshold=min_pixels,
            x_stride=primary_stride,
            y_stride=primary_stride,
            merge=True,
            margin=primary_margin,
        )
        if blobs:
            return blobs
    except Exception:
        pass

    try:
        blobs = img.find_blobs(
            [thresh],
            pixels_threshold=relaxed_pixels,
            area_threshold=relaxed_pixels,
            x_stride=relaxed_stride,
            y_stride=relaxed_stride,
            merge=True,
            margin=relaxed_margin,
        )
        if blobs:
            return blobs
    except Exception:
        pass

    try:
        return img.find_blobs([thresh], pixels_threshold=relaxed_pixels)
    except Exception:
        return []


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _pick_best_shape_hit(hits):
    if not hits:
        return None

    # 按形状聚合命中分数，降低单分支抖动导致的误判。
    merged_by_shape = {}
    for hit in hits:
        shape = hit.get("shape", "unknown")
        if shape == "unknown":
            continue
        merged_by_shape.setdefault(shape, []).append(hit)

    aggregated = []
    for shape, shape_hits in merged_by_shape.items():
        base = max(shape_hits, key=lambda h: h.get("score", 0.0))
        avg_score = sum(h.get("score", 0.0) for h in shape_hits) / max(1, len(shape_hits))
        support_bonus = min(0.18, 0.08 * (len(shape_hits) - 1))
        h = dict(base)
        h["score"] = base.get("score", 0.0) + (avg_score * 0.12) + support_bonus
        aggregated.append(h)

    if not aggregated:
        return None

    hits_sorted = sorted(aggregated, key=lambda h: h["score"], reverse=True)
    best = hits_sorted[0]

    circle_hit = None
    rectangle_hit = None
    triangle_hit = None
    for hit in hits_sorted:
        if hit.get("shape") == "circle":
            circle_hit = hit
        elif hit.get("shape") == "rectangle":
            rectangle_hit = hit
        elif hit.get("shape") == "triangle":
            triangle_hit = hit

    # 三角形与矩形冲突时，优先识别出明显三角形，避免矩形分支过强覆盖。
    if triangle_hit and rectangle_hit:
        tri_fill = triangle_hit.get("fill", 0.0)
        tri_offset = triangle_hit.get("center_offset", 1.0)
        tri_convexity = triangle_hit.get("convexity", 0.0)
        tri_aspect = triangle_hit.get("aspect", 1.0)

        tri_geometry_strong = (
            0.14 <= tri_fill <= 0.74
            and tri_offset >= 0.10
            and tri_convexity >= 0.50
            and 0.32 <= tri_aspect <= 3.20
        )
        rect_triangle_risk = (
            rectangle_hit.get("fill", 0.0) <= 0.70
            and rectangle_hit.get("center_offset", 1.0) >= 0.10
        )
        close_score = triangle_hit["score"] >= (rectangle_hit["score"] * 0.83)
        if tri_geometry_strong and rect_triangle_risk and close_score:
            return triangle_hit

    if rectangle_hit and circle_hit:
        rect_fill = rectangle_hit.get("fill", 0.0)
        rect_offset = rectangle_hit.get("center_offset", 1.0)
        rect_solidity = rectangle_hit.get("solidity", 0.0)
        rect_aspect = rectangle_hit.get("aspect", 1.0)
        
        # Perfect/near-perfect squares
        rect_like_square = (
            rect_fill >= 0.84
            and rect_offset <= 0.11
            and rect_solidity >= 0.78
            and 0.42 <= rect_aspect <= 2.40
        )
        
        # Tilted rectangles: high solidity is strong evidence for rectangle, even if offset/fill lower
        rect_like_tilted = (
            rect_solidity >= 0.76
            and 0.40 <= rect_fill <= 0.88
            and rect_offset <= 0.18
            and 0.42 <= rect_aspect <= 2.50
            and not (rect_fill <= 0.70 and rect_offset >= 0.10)
        )
        
        close_score = rectangle_hit["score"] >= (circle_hit["score"] * 0.76)
        if (rect_like_square or rect_like_tilted) and close_score:
            return rectangle_hit

    # 圆被透视压扁后会像椭圆，优先用 roundness + convexity 纠偏。
    if circle_hit:
        circle_fill = circle_hit.get("fill", 0.0)
        circle_offset = circle_hit.get("center_offset", 1.0)
        circle_solidity = circle_hit.get("solidity", 0.0)
        
        # 高solidity通常意味着矩形/凸形，不是圆
        if circle_solidity >= 0.76:
            # Prefer rectangle if solidity is high, even if other metrics match circle
            pass  # Skip geometric_circle fallback for high-solidity shapes
        else:
            # 不依赖 roundness 的兜底：圆/椭圆在 bbox 内通常填充率接近 pi/4，且质心更居中。
            geometric_circle = (
                0.60 <= circle_fill <= 0.85  # Tightened bounds: 0.60-0.85 instead of 0.56-0.88
                and circle_offset <= 0.11    # Tightened: 0.11 instead of 0.14
                and circle_hit.get("roundness", 0.0) >= 0.30
                and 0.35 <= circle_hit.get("aspect", 1.0) <= 2.85
            )
            if geometric_circle and circle_hit["score"] >= (best["score"] * 0.70):
                return circle_hit

        ellipse_like = (
            circle_hit.get("roundness", 0.0) >= 0.46
            and circle_hit.get("convexity", 0.0) >= 0.58
            and circle_hit.get("center_offset", 1.0) <= 0.18
            and 0.30 <= circle_hit.get("fill", 0.0) <= 0.88
            and 0.35 <= circle_hit.get("aspect", 1.0) <= 2.80
        )
        close_score = circle_hit["score"] >= (best["score"] * 0.74)
        if ellipse_like and close_score:
            return circle_hit

    # 矩形转成菱形时，填充率会下降且容易被误打成三角；用质心偏移和凸性纠偏回矩形。
    if rectangle_hit and triangle_hit:
        diamond_like = (
            rectangle_hit.get("center_offset", 1.0) <= 0.10
            and rectangle_hit.get("convexity", 0.0) >= 0.62
            and rectangle_hit.get("solidity", 0.0) >= 0.46
            and 0.28 <= rectangle_hit.get("fill", 0.0) <= 0.82
            and 0.40 <= rectangle_hit.get("aspect", 1.0) <= 2.60
        )
        close_score = rectangle_hit["score"] >= (triangle_hit["score"] * 0.90)
        if diamond_like and close_score:
            return rectangle_hit

    return best


def _map_hit_to_original(hit, src_w, src_h, dst_w, dst_h):
    x = int((hit["x"] * dst_w) / max(1, src_w))
    y = int((hit["y"] * dst_h) / max(1, src_h))
    w = int((hit["w"] * dst_w) / max(1, src_w))
    h = int((hit["h"] * dst_h) / max(1, src_h))

    x = max(0, min(dst_w - 1, x))
    y = max(0, min(dst_h - 1, y))
    w = max(1, w)
    h = max(1, h)
    if x + w > dst_w:
        w = max(1, dst_w - x)
    if y + h > dst_h:
        h = max(1, dst_h - y)
    return {"x": x, "y": y, "w": w, "h": h}


def identify_markers_multi(img, current_thresholds, draw=True, max_results=8, detect_size=(320, 240), fast_mode=True):
    colors_to_check = ["red", "green", "blue"]
    candidates = []
    
    img_w, img_h = img.width(), img.height()

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
                work_img = img
                work_w, work_h = img_w, img_h

    detect_min_pixels = _adaptive_pixels_threshold(work_img, MIN_PIXELS_MARKER)

    for color in colors_to_check:
        thresh = _threshold_for_color(color, current_thresholds)
        if not thresh:
            continue

        blobs = _find_color_blobs(work_img, thresh, min_pixels=detect_min_pixels, fast_mode=fast_mode)
        if not blobs:
            continue

        shape_hits = []
        shape_hits.extend(my_find_triangle(blobs, work_w, work_h, min_pixels=detect_min_pixels))
        shape_hits.extend(my_find_rectangle(blobs, work_w, work_h, min_pixels=detect_min_pixels))
        shape_hits.extend(my_find_circle(blobs, work_w, work_h, min_pixels=detect_min_pixels))
        shape_hits.extend(my_find_perspective_shapes(blobs, work_w, work_h, min_pixels=detect_min_pixels))

        # 同一 blob 若命中多个形状，只保留分数最高的一个。
        hits_per_blob = {}
        for hit in shape_hits:
            key = (hit["x"], hit["y"], hit["w"], hit["h"])
            hits_per_blob.setdefault(key, []).append(hit)

        for hits in hits_per_blob.values():
            hit = _pick_best_shape_hit(hits)
            if not hit:
                continue

            mapped = _map_hit_to_original(hit, work_w, work_h, img_w, img_h)
            score = (hit["score"] * 2.0) + (hit["pixels"] / 3500.0) + (hit["fill"] * 0.5)
            candidates.append(
                {
                    "color": color,
                    "shape": hit["shape"],
                    "x": mapped["x"],
                    "y": mapped["y"],
                    "w": mapped["w"],
                    "h": mapped["h"],
                    "score": score,
                }
            )

    if not candidates:
        return []

    # 按评分排序并做轻量去重，避免同一目标被重复框选。
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



def _pixel_is_foreground(pixel):
    if isinstance(pixel, int):
        return pixel > 0
    if isinstance(pixel, (tuple, list)):
        if len(pixel) == 0:
            return False
        if len(pixel) == 1:
            return int(pixel[0]) > 0
        return (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) > 0
    return False


def _normalize_preview_size(src_w, src_h, preview_size):
    if isinstance(preview_size, (tuple, list)) and len(preview_size) >= 2:
        w = int(preview_size[0])
        h = int(preview_size[1])
    else:
        w = src_w // 4
        h = src_h // 4

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
    rx, ry, rw, rh = roi
    mw, mh = mask_img.width(), mask_img.height()
    if mw <= 0 or mh <= 0:
        return

    row_h = max(1, (rh + mh - 1) // mh)
    for sy in range(mh):
        py = ry + (sy * rh) // mh
        sx = 0
        while sx < mw:
            while sx < mw and not _pixel_is_foreground(mask_img.get_pixel(sx, sy)):
                sx += 1
            if sx >= mw:
                break

            run_start = sx
            while sx < mw and _pixel_is_foreground(mask_img.get_pixel(sx, sy)):
                sx += 1

            px1 = rx + (run_start * rw) // mw
            px2 = rx + (sx * rw) // mw
            run_w = max(1, px2 - px1)
            dst_img.draw_rect(px1, py, run_w, row_h, image.COLOR_WHITE, thickness=-1)


def get_binary_preview_rects(src_img, target_name, current_thresholds, use_mask=False, preview_size=None):
    thresh = _threshold_for_color(target_name, current_thresholds)
    if not thresh:
        return {"mode": "empty", "rects": []}

    if not use_mask:
        blobs = []
        try:
            blobs = src_img.find_blobs([thresh], pixels_threshold=80)
        except Exception:
            blobs = []
        rects = []
        if blobs:
            blobs_sorted = sorted(blobs, key=lambda b: b[2] * b[3], reverse=True)
            for blob in blobs_sorted[:24]:
                if blob[2] * blob[3] < 90:
                    continue
                rects.append((blob[0], blob[1], blob[2], blob[3]))
        return {"mode": "rects", "rects": rects}

    # 先降分辨率再二值化，可显著提升调参刷新率。
    try:
        src_w, src_h = src_img.width(), src_img.height()
        low_w, low_h = _normalize_preview_size(src_w, src_h, preview_size)
        low_img = src_img.resize(low_w, low_h, method=image.ResizeMethod.NEAREST)
        low_img.binary([thresh])
        return {"mode": "mask_lowres", "mask": low_img, "rects": []}
    except Exception:
        pass

    # 兼容回退：无法缩放时，退回原尺寸二值化。
    try:
        mask_img = src_img.copy()
        mask_img.binary([thresh])
        return {"mode": "mask", "mask": mask_img, "rects": []}
    except Exception:
        pass

    return {"mode": "rects", "rects": []}


def draw_binary_preview(dst_img, preview_data, roi):
    img_w, img_h = dst_img.width(), dst_img.height()
    rx, ry, rw, rh = roi
    dst_img.draw_rect(rx, ry, rw, rh, image.COLOR_BLACK, thickness=4)
    dst_img.draw_rect(rx + 2, ry + 2, rw - 4, rh - 4, image.COLOR_BLACK, thickness=-1)

    rects = preview_data.get("rects", []) if isinstance(preview_data, dict) else preview_data

    if isinstance(preview_data, dict) and preview_data.get("mode") in ("mask", "mask_lowres"):
        mask_img = preview_data.get("mask")
        if mask_img:
            try:
                scaled_mask = mask_img.resize(rw, rh, method=image.ResizeMethod.NEAREST)
                dst_img.draw_image(rx, ry, scaled_mask)
                return
            except Exception:
                try:
                    _draw_mask_to_roi(dst_img, mask_img, roi)
                    return
                except Exception:
                    pass

    # 回退模式：如果没有 mask，直接画矩形
    for bx, by, bw, bh in rects:
        px = rx + (bx * rw) // img_w
        py = ry + (by * rh) // img_h
        pw = max(1, (bw * rw) // img_w)
        ph = max(1, (bh * rh) // img_h)
        dst_img.draw_rect(px, py, pw, ph, image.COLOR_WHITE, thickness=-1)



