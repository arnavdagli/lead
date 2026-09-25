#!/usr/bin/env python3
"""lead: webcam object tracker with latency-compensated prediction.

Draws where a tracked object IS (green, solid) and where it WILL BE
after `lead_ms` (magenta, dashed), using a constant-velocity Kalman
filter. Logs prediction error against ground truth once the future
actually arrives, so the whole thesis (prediction quality) is
measurable, not just visual.
"""

import argparse
import csv
import sys
import time
from collections import deque

import cv2
import numpy as np

# ---- Tunables not exposed as CLI flags -------------------------------
DEFAULT_FPS = 30.0          # fallback when source can't report a real fps
PROCESS_NOISE = 1e-2        # Kalman process noise (trust in constant-v model)
MEASUREMENT_NOISE = 1e-1    # Kalman measurement noise (trust in CSRT box)
LIVE_ERROR_WINDOW = 100     # frames of rolling error shown in the overlay
DASH_LEN = 10                # px, for dashed prediction box
FPS_SMOOTHING = 0.9          # EMA factor for the displayed processing fps
HEADLESS_BOX_FRACTION = 0.25 # default target box size (of frame dim) when
                              # there's no display to drag a box on


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lead-ms", type=int, default=300,
                    help="how far ahead to predict, in ms (default 300)")
    p.add_argument("--source", default="0",
                    help="webcam index or path to a video file (default 0)")
    p.add_argument("--record", default=None, metavar="PATH",
                    help="write the annotated output to an mp4")
    p.add_argument("--no-display", action="store_true",
                    help="headless: no windows, no interactive ROI select")
    return p.parse_args()


def open_source(source):
    # numeric-looking strings are webcam indices, everything else a path
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def make_tracker():
    # CSRT moved into cv2.legacy in some OpenCV builds; support both.
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


def select_initial_box(frame, headless):
    if not headless:
        box = cv2.selectROI("Select object (drag box, press ENTER)", frame,
                             fromCenter=False, showCrosshair=False)
        cv2.destroyWindow("Select object (drag box, press ENTER)")
        if box[2] == 0 or box[3] == 0:
            sys.exit("No box selected, exiting.")
        return box
    # Headless: no window to drag a box on, so fall back to a fixed
    # centered box sized relative to the frame. Simpler than running
    # motion detection to auto-find a target, and keeps headless runs
    # deterministic/reproducible on a given video file.
    h, w = frame.shape[:2]
    bw, bh = int(w * HEADLESS_BOX_FRACTION), int(h * HEADLESS_BOX_FRACTION)
    return ((w - bw) // 2, (h - bh) // 2, bw, bh)


def make_kalman(cx, cy, dt):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                     [0, 1, 0, dt],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    return kf


def dashed_line(img, pt1, pt2, color, thickness):
    pt1 = np.array(pt1, dtype=np.float64)
    pt2 = np.array(pt2, dtype=np.float64)
    dist = np.linalg.norm(pt2 - pt1)
    if dist < 1:
        return
    n = max(1, int(dist / DASH_LEN))
    for i in range(n):
        if i % 2 == 0:
            a = pt1 + (pt2 - pt1) * (i / n)
            b = pt1 + (pt2 - pt1) * ((i + 1) / n)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                      color, thickness)


def dashed_rect(img, top_left, bottom_right, color, thickness):
    x1, y1 = top_left
    x2, y2 = bottom_right
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        dashed_line(img, corners[i], corners[(i + 1) % 4], color, thickness)


def main():
    args = parse_args()
    cap = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1 or fps > 240:
        fps = DEFAULT_FPS
    dt = 1.0 / fps
    lead_frames = max(1, round(args.lead_ms / (1000.0 / fps)))

    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read a frame from the source.")

    box = select_initial_box(frame, args.no_display)
    tracker = make_tracker()
    tracker.init(frame, box)

    cx0, cy0 = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    kalman = make_kalman(cx0, cy0, dt)

    writer = None
    if args.record:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, fps, (w, h))

    # CSRT's update() reliably fails if called on the exact same frame
    # passed to init() (it needs a frame to actually track motion in),
    # so the init frame is consumed here and the loop below starts on
    # the next one.
    ok, frame = cap.read()
    if not ok:
        sys.exit("Source ended right after target selection, nothing to track.")

    pending = deque()   # (target_frame_idx, pred_x, pred_y, pred_t_ms)
    recent_errors = deque(maxlen=LIVE_ERROR_WINDOW)
    all_rows = []       # for error_log.csv
    frame_idx = 0
    proc_fps = fps
    last_t = time.time()

    try:
        while True:
            now = time.time()
            iter_dt = now - last_t
            last_t = now
            if iter_dt > 0:
                proc_fps = FPS_SMOOTHING * proc_fps + (1 - FPS_SMOOTHING) * (1.0 / iter_dt)

            success, box = tracker.update(frame)
            if not success:
                print(f"Tracking lost at frame {frame_idx}, stopping.")
                break

            mx, my = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
            t_ms = frame_idx * (1000.0 / fps)

            kalman.predict()
            state = kalman.correct(
                np.array([[np.float32(mx)], [np.float32(my)]])).flatten()
            vx, vy = state[2], state[3]

            lead_s = args.lead_ms / 1000.0
            pred_x = state[0] + vx * lead_s
            pred_y = state[1] + vy * lead_s

            # Score any prediction whose target time has now arrived.
            while pending and pending[0][0] == frame_idx:
                _, px, py, pred_t_ms = pending.popleft()
                err = float(np.hypot(mx - px, my - py))
                recent_errors.append(err)
                all_rows.append([frame_idx, round(t_ms, 1), round(px, 1),
                                  round(py, 1), round(mx, 1), round(my, 1),
                                  round(err, 2)])

            pending.append((frame_idx + lead_frames, pred_x, pred_y, t_ms + args.lead_ms))

            # --- draw ---
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            px1, py1 = int(pred_x - w / 2), int(pred_y - h / 2)
            dashed_rect(frame, (px1, py1), (px1 + w, py1 + h), (255, 0, 255), 2)
            cv2.line(frame, (int(mx), int(my)), (int(pred_x), int(pred_y)),
                      (255, 255, 0), 1)

            mean_err = np.mean(recent_errors) if recent_errors else 0.0
            cv2.putText(frame, f"FPS: {proc_fps:.1f}  lead: {args.lead_ms}ms  "
                                 f"err: {mean_err:.1f}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("lead", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    with open("error_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_ms", "pred_x", "pred_y", "actual_x", "actual_y", "error_px"])
        w.writerows(all_rows)

    if all_rows:
        errors = np.array([r[-1] for r in all_rows])
        print("\n--- lead summary ---")
        print(f"scored predictions : {len(errors)}")
        print(f"mean error (px)    : {errors.mean():.2f}")
        print(f"p95 error (px)     : {np.percentile(errors, 95):.2f}")
        print(f"min / max (px)     : {errors.min():.2f} / {errors.max():.2f}")
        print("error_log.csv written")
    else:
        print("No predictions were scored (source ended before lead_ms elapsed).")


if __name__ == "__main__":
    main()
