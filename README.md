# lead

By the time a real actuator (a motor, a turret, a cursor, a robot arm)
reacts to where an object *was*, the object has already moved somewhere
else — that gap is latency, and it's fixed cost you can't track your way
out of. `lead` tracks a webcam target frame-to-frame with OpenCV's CSRT
tracker, feeds its position into a constant-velocity Kalman filter, and
draws a second box showing where the filter predicts the object will be
`lead_ms` in the future. The prediction is the point: every frame that
prediction is later checked against what actually happened, and the
resulting error (mean and p95, in pixels) is logged and printed, because
a latency-compensation technique that doesn't measure its own error is
just a guess with extra steps.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Note: `requirements.txt` installs
`opencv-contrib-python` rather than plain `opencv-python` — the CSRT
tracker lives in the contrib build, and contrib is a drop-in superset
(same `cv2` import) so nothing else changes.

## Examples

```bash
# Webcam, default 300ms lead: drag a box around the target, then track it
python3 lead.py

# Longer lead time on a specific webcam, saving the annotated output
python3 lead.py --lead-ms 500 --source 1 --record out.mp4

# Headless run on a video file — no window, no webcam needed,
# so results are reproducible without hardware
python3 lead.py --source clip.mp4 --no-display --lead-ms 200 --record out.mp4
```

Every run writes `error_log.csv` (`frame_idx, t_ms, pred_x, pred_y,
actual_x, actual_y, error_px`) and prints a summary (count, mean, p95,
min/max) to stdout on exit.

## What the error numbers mean, and their limits

Each frame, the filter predicts a future position `lead_ms` ahead. That
prediction sits in a queue until a frame actually arrives at that target
time, at which point it's compared against the tracker's real measured
position and logged as a pixel error. Mean error is the average
across the run; p95 is the error your worst-case-but-not-pathological
frame sees — the number that matters more than the mean if you're
deciding how much margin an actuator needs.

Limits worth knowing before trusting these numbers:

- **"Ground truth" is the CSRT box, not the true object position.**
  Tracker jitter and drift show up in the error metric as if it were
  prediction error, even when the Kalman filter is doing fine.
- **Timing is nominal, not measured per-frame.** `t_ms` is derived from
  the source's reported/assumed fps (`frame_idx * 1000/fps`), not a
  wall-clock timestamp per frame. For a video file this is exact; for a
  live webcam with an unstable capture rate, it's an approximation, and
  error will look worse than the filter's true performance during fps
  hiccups.
- **Constant-velocity only.** Sudden turns, stops, or occlusion produce
  large, honest error spikes — the model has no way to anticipate
  acceleration, by design.
- **The predicted box reuses the current measured box size** (the
  Kalman filter only tracks center position/velocity, not scale), so
  the drawn prediction box doesn't grow or shrink even if the real
  object is moving toward or away from the camera.

## License

MIT, see [LICENSE](LICENSE).
