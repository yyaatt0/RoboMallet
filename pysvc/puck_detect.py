import cv2
import numpy as np
import time


MIN_R = 30
MAX_R = 60
SCALE = 0.5
INV_SCALE = 1.0 / SCALE
MIN_CONFIDENCE = 0.45
NUM_SAMPLE_PTS = 36


def validate_circle(edges, cx, cy, cr):
    h, w = edges.shape
    angles = np.linspace(0, 2 * np.pi, NUM_SAMPLE_PTS, endpoint=False)
    xs = np.round(cx + cr * np.cos(angles)).astype(int)
    ys = np.round(cy + cr * np.sin(angles)).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() == 0:
        return 0.0
    hits = 0
    for px, py, v in zip(xs, ys, valid):
        if not v:
            continue
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and edges[ny, nx] > 0:
                    hits += 1
                    break
            else:
                continue
            break
    return hits / valid.sum()


def detect_hough(small_blur, edges, sh):
    circles = cv2.HoughCircles(
        small_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=sh // 3,
        param1=80,
        param2=22,
        minRadius=MIN_R,
        maxRadius=MAX_R,
    )
    if circles is not None:
        for i in range(circles.shape[1]):
            x, y, r = circles[0, i]
            conf = validate_circle(edges, int(x), int(y), int(r))
            if conf >= MIN_CONFIDENCE:
                return float(x) * INV_SCALE, float(y) * INV_SCALE, float(r) * INV_SCALE, conf
    return None


def detect_contour(small_gray, edges):
    thresh = cv2.adaptiveThreshold(
        small_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    best_conf = 0.0
    min_area = np.pi * MIN_R * MIN_R
    max_area = np.pi * MAX_R * MAX_R

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = (4.0 * np.pi * area) / (perim * perim)
        if circ < 0.65:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(cnt)
        if rad < MIN_R or rad > MAX_R:
            continue
        conf = validate_circle(edges, int(cx), int(cy), int(rad))
        if conf < MIN_CONFIDENCE:
            continue
        score = circ * area * conf
        if score > best_score:
            best_score = score
            best_conf = conf
            best = (float(cx) * INV_SCALE, float(cy) * INV_SCALE, float(rad) * INV_SCALE, best_conf)

    return best


def draw_puck(frame, cx, cy, cr, detected, conf):
    h, w = frame.shape[:2]

    x1 = max(cx - cr, 0)
    y1 = max(cy - cr, 0)
    x2 = min(cx + cr, w)
    y2 = min(cy + cr, h)

    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2].copy()
        mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(mask_roi, (cx - x1, cy - y1), cr, 255, -1)
        fill = (0, 220, 0) if detected else (0, 160, 220)
        colored = np.full_like(roi, fill, dtype=np.uint8)
        blended = cv2.addWeighted(colored, 0.25, roi, 0.75, 0)
        np.copyto(frame[y1:y2, x1:x2], blended, where=mask_roi[:, :, None] > 0)

    color = (0, 255, 0) if detected else (0, 180, 255)
    cv2.circle(frame, (cx, cy), cr, color, 2, cv2.LINE_AA)

    s, g = 12, 3
    cv2.line(frame, (cx - s, cy), (cx - g, cy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx + g, cy), (cx + s, cy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - s), (cx, cy - g), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + g), (cx, cy + s), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    return color


def draw_hud(frame, fps, cx, cy, cr, tracking, detected, method, conf, fw):
    bar = frame[0:42, :].copy()
    frame[0:42, :] = (bar * 0.45).astype(np.uint8)

    fps_col = (0, 255, 255) if fps >= 25 else (0, 140, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_col, 2, cv2.LINE_AA)

    if tracking:
        tag = "TRACKING" if detected else "COASTING"
        tc = (0, 255, 0) if detected else (0, 180, 255)
        cv2.putText(frame, tag, (fw // 2 - 55, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 2, cv2.LINE_AA)
        info = f"X:{cx}  Y:{cy}  R:{cr}  C:{conf:.0%}  [{method}]"
        cv2.putText(frame, info, (fw - 360, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "NO PUCK DETECTED", (fw // 2 - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hw = int(fw * SCALE)
    hh = int(fh * SCALE)


    small_gray = np.empty((hh, hw), dtype=np.uint8)
    small_blur = np.empty_like(small_gray)

    sx, sy, sr  = 0.0, 0.0, 0.0
    tracking    = False
    lost_frames = 0
    ALPHA       = 0.6
    MAX_LOST    = 12
    method      = ""
    last_conf   = 0.0

    frame_num        = 0
    CONTOUR_INTERVAL = 4

    fc       = 0
    fps_show = 0.0
    t0       = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_num += 1

        small = cv2.resize(frame, (hw, hh), interpolation=cv2.INTER_AREA)
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY, dst=small_gray)
        cv2.GaussianBlur(small_gray, (7, 7), 1.5, dst=small_blur)
        edges = cv2.Canny(small_blur, 50, 150)

        result = detect_hough(small_blur, edges, hh)
        method = "hough"

        if result is None and (frame_num % CONTOUR_INTERVAL == 0):
            result = detect_contour(small_gray, edges)
            method = "contour"

        detected = False
        if result is not None:
            dx, dy, dr, conf = result
            last_conf = conf
            if tracking:
                sx = ALPHA * dx + (1 - ALPHA) * sx
                sy = ALPHA * dy + (1 - ALPHA) * sy
                sr = ALPHA * dr + (1 - ALPHA) * sr
            else:
                sx, sy, sr = dx, dy, dr
            tracking    = True
            lost_frames = 0
            detected    = True
        elif tracking:
            lost_frames += 1
            if lost_frames > MAX_LOST:
                tracking = False

        if tracking:
            cx = int(round(sx))
            cy = int(round(sy))
            cr = max(int(round(sr)), 1)
            draw_puck(frame, cx, cy, cr, detected, last_conf)
        else:
            cx, cy, cr = 0, 0, 0

        fc += 1
        dt = time.perf_counter() - t0
        if dt >= 0.5:
            fps_show = fc / dt
            fc = 0
            t0 = time.perf_counter()

        draw_hud(frame, fps_show, cx, cy, cr, tracking, detected, method, last_conf, fw)

        cv2.imshow("Puck Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()