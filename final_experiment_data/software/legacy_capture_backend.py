#!/usr/bin/env python3
"""Guided Raspberry Pi synthetic-light-field DNG capture (MECH5720).

Requires the existing Pi stream.py API via an SSH tunnel, numpy, rawpy.
Original DNGs are saved unmodified. No motor movement is performed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FIELDS = [
    "view", "timestamp_utc", "x_mm", "filename", "bytes", "sha256",
    "exposure_us_requested", "gain_requested", "ae_requested", "raw_height",
    "raw_width", "black_levels_dn", "white_level_dn", "min_dn", "median_dn",
    "p99_dn", "max_dn", "saturated_pct", "near_black_pct", "notes",
]


def api(base, path, payload=None, timeout=20):
    headers = {}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(base + path, data=body, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            content = response.read()
            if response.status != 200:
                raise RuntimeError(f"{path}: HTTP {response.status}")
            return content if path == "/capture_dng" else json.loads(content)
    except HTTPError as exc:
        detail = exc.read(300).decode("utf-8", "replace")
        raise RuntimeError(f"{path}: HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"{path}: connection failed ({exc}); check the Wi-Fi tunnel and Pi service") from exc


def ask(prompt, default=None, *, convert=str, predicate=lambda _: True):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        reply = input(f"{prompt}{suffix}: ").strip()
        if not reply and default is not None:
            reply = str(default)
        try:
            result = convert(reply)
            if predicate(result):
                return result
        except (TypeError, ValueError):
            pass
        print("  Invalid value; please try again.")


def yn(prompt, default=True):
    suffix = "Y/n" if default else "y/N"
    while True:
        x = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not x:
            return default
        if x in ("y", "yes"):
            return True
        if x in ("n", "no"):
            return False
        print("  Please enter y or n.")


def get_setting(controls, name, fallback):
    value = controls.get(name, {}).get("current")
    return fallback if value is None else value


def set_manual(base, exposure, gain):
    answer = api(base, "/controls", {
        "AeEnable": False,
        "ExposureTime": int(exposure),
        "AnalogueGain": float(gain),
    })
    if answer.get("status") != "applied":
        raise RuntimeError(f"Control update not acknowledged: {answer}")
    time.sleep(0.7)
    controls = api(base, "/controls")
    observed = (get_setting(controls, "AeEnable", None),
                get_setting(controls, "ExposureTime", None),
                get_setting(controls, "AnalogueGain", None))
    if observed[0] is not False or observed[1] != int(exposure) or not math.isclose(float(observed[2]), gain, rel_tol=0.001):
        raise RuntimeError(f"Requested controls not confirmed by API: {observed}")
    return observed


def metrics_from_dng(path, expected=(1080, 1920)):
    import numpy as np
    import rawpy
    with rawpy.imread(str(path)) as raw:
        im = raw.raw_image_visible
        if im.shape != expected:
            raise ValueError(f"Unexpected RAW dimensions: {im.shape}; expected {expected}")
        black = [int(x) for x in raw.black_level_per_channel]
        white = int(raw.white_level)
        if white <= max(black):
            raise ValueError("Invalid RAW black/white levels")
        if int(im.max()) == int(im.min()):
            raise ValueError("RAW image is constant")
        # Use the minimum of channel black levels as a conservative diagnostic.
        lower = min(black)
        return {
            "raw_height": int(im.shape[0]), "raw_width": int(im.shape[1]),
            "black_levels_dn": json.dumps(black), "white_level_dn": white,
            "min_dn": int(im.min()), "median_dn": float(np.median(im)),
            "p99_dn": float(np.percentile(im, 99)), "max_dn": int(im.max()),
            "saturated_pct": round(100.0 * float(np.mean(im >= white)), 6),
            "near_black_pct": round(100.0 * float(np.mean(im <= lower + 8)), 6),
        }


def capture(base, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    dng = api(base, "/capture_dng", timeout=90)
    if len(dng) < 100_000:
        raise ValueError(f"Capture returned only {len(dng)} bytes (not a plausible DNG)")
    temp = destination.with_name(destination.name + ".part")
    if temp.exists():
        raise FileExistsError(f"Refusing to overwrite temporary file {temp}")
    try:
        with temp.open("xb") as fp:
            fp.write(dng)
            fp.flush()
            os.fsync(fp.fileno())
        stats = metrics_from_dng(temp)
        os.replace(temp, destination)
    except Exception:
        # Keep failed data for diagnosis; never mislabel it as a valid DNG.
        if temp.exists():
            bad = destination.with_name(destination.name + ".failed")
            if not bad.exists():
                os.replace(temp, bad)
        raise
    return stats, len(dng), hashlib.sha256(dng).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090", help="Existing wireless SSH tunnel address")
    parser.add_argument("--output", required=True, type=Path, help="NEW output session directory (must not exist)")
    parser.add_argument("--views", type=int, default=9, help="Number of physical rail positions (default: 9)")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    if args.views < 2 or args.views > 200:
        parser.error("--views must be between 2 and 200")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}. Choose a NEW session directory.")

    print("\nMECH5720 | Guided synthetic light-field RAW acquisition")
    print("No automatic stage movement. Keep scene, lighting, focus and camera orientation fixed.")
    print("Press Ctrl+C between views to stop safely; completed files and CSV rows remain saved.\n")
    config = api(args.url, "/config")
    controls = api(args.url, "/controls")
    print("Connected. Camera configuration:", config)
    if (config.get("width"), config.get("height")) != (1920, 1080):
        print("Setting 1920 x 1080 RGB888 (DNG RAW will be validated separately)...")
        result = api(args.url, "/config", {"width": 1920, "height": 1080, "format": "RGB888"}, timeout=35)
        if result.get("status") != "reconfigured":
            raise RuntimeError(f"Reconfiguration failed: {result}")
        time.sleep(1)
    config = api(args.url, "/config")
    if (config.get("width"), config.get("height")) != (1920, 1080):
        raise RuntimeError("Camera not configured at 1920x1080")

    print("\nCALIBRATION (before the real run):")
    print("Aim at the ACTUAL targets with final lighting and focus. Existing viewer sliders also work.")
    if yn("Enable AE briefly to obtain a starting exposure", default=True):
        api(args.url, "/controls", {"AeEnable": True})
        print("Wait several seconds for exposure to settle...")
        time.sleep(4)
    controls = api(args.url, "/controls")
    print("Currently reported: exposure", get_setting(controls, "ExposureTime", "?"),
          "us, analogue gain", get_setting(controls, "AnalogueGain", "?"))

    exposure = gain = None
    while True:
        controls = api(args.url, "/controls")
        erange = controls["ExposureTime"]
        grange = controls["AnalogueGain"]
        e_default = get_setting(controls, "ExposureTime", 20000)
        g_default = get_setting(controls, "AnalogueGain", 1.0)
        exposure = ask("Exposure in microseconds (e.g. 20000 = 20 ms)", e_default,
                       convert=int, predicate=lambda v: erange["min"] <= v <= erange["max"])
        gain = ask("Analogue gain (e.g. 1.0, 2.0)", g_default,
                   convert=float, predicate=lambda v: math.isfinite(v) and grange["min"] <= v <= grange["max"])
        set_manual(args.url, exposure, gain)
        print("AE OFF; requested exposure and gain confirmed by the control API.")
        if not yn("Take a calibration test DNG (recommended)", default=True):
            if yn("Accept these settings without RAW quality check", default=False):
                break
            continue
        # Calibration check is kept separately from the actual session; no file is discarded.
        trial_dir = args.output.parent / (args.output.name + "_calibration")
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_file = trial_dir / f"test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.dng"
        print("Capturing calibration frame...")
        try:
            stats, nbytes, digest = capture(args.url, trial_file)
        except Exception as exc:
            print("Calibration capture failed:", exc)
            if not yn("Try again with new settings", default=True):
                raise
            continue
        print("Saved test:", trial_file)
        print(f"RAW median {stats['median_dn']:.1f} DN, p99 {stats['p99_dn']:.1f} DN; "
              f"white {stats['white_level_dn']} DN; saturated {stats['saturated_pct']:.4f}%; "
              f"near-black {stats['near_black_pct']:.2f}%")
        print("Examine the VIEWER for composition/focus; no numerical metric replaces visual judgement.")
        if yn("Accept settings for the entire run", default=True):
            break

    scene = ask("Scene/target depths in mm and notes (free text; use 'unknown' if not measured)",
                "unknown", convert=str, predicate=lambda s: bool(s.strip()))
    operator = ask("Run description", "baseline sweep", convert=str, predicate=lambda s: bool(s.strip()))
    if not yn(f"Start {args.views} views at {exposure} us, {gain:g}x gain, AE OFF", default=True):
        print("Cancelled before creating acquisition directory.")
        return 0

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "manifest.csv"
    info = {
        "session_created_utc": datetime.now(timezone.utc).isoformat(),
        "description": operator, "scene_depths_and_notes": scene,
        "requested_views": args.views, "camera_config": config,
        "controls_requested": {"AeEnable": False, "ExposureTime": exposure, "AnalogueGain": gain},
        "api_url": args.url, "units": {"x": "mm", "exposure": "us", "pixel_values": "raw DN"},
        "note": "RAW DNG original bytes; x_mm is manually measured camera translation. Control API reports requested values, not verified per-frame metadata.",
    }
    (args.output / "session.json").write_text(json.dumps(info, indent=2) + "\n")
    used_positions = []
    with manifest.open("x", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        fp.flush()
        for view in range(1, args.views + 1):
            print(f"\n--- View {view}/{args.views} ---")
            print("Move camera to the required physical rail stop, measure x in mm; wait for vibration to settle.")
            while True:
                x_mm = ask("Measured x (mm, signed relative to chosen origin)", convert=float,
                           predicate=math.isfinite)
                if x_mm not in used_positions:
                    break
                print("  Position already used in this run; enter a new value.")
            notes = input("View notes (optional): ").strip()
            name = f"view_{view:03d}_x{x_mm:+09.3f}mm.dng"
            target = args.output / name
            if not yn(f"Capture at x={x_mm:g} mm now", default=True):
                print("Stopped at user request; existing views are saved.")
                return 0
            # Reassert manual controls in case the viewer was used during the run.
            set_manual(args.url, exposure, gain)
            try:
                stats, nbytes, digest = capture(args.url, target)
            except Exception as exc:
                print("Capture failed; no valid DNG was added to manifest:", exc)
                print("Inspect network/camera and restart with a NEW session directory if necessary.")
                return 1
            row = {
                "view": view, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "x_mm": x_mm, "filename": name, "bytes": nbytes, "sha256": digest,
                "exposure_us_requested": exposure, "gain_requested": gain,
                "ae_requested": False, **stats, "notes": notes,
            }
            writer.writerow(row)
            fp.flush()
            os.fsync(fp.fileno())
            used_positions.append(x_mm)
            print(f"Saved {name} ({nbytes / 1024**2:.2f} MiB). "
                  f"p99={stats['p99_dn']:.1f} DN, saturated={stats['saturated_pct']:.4f}%")
            if stats["saturated_pct"] > 0.1:
                print("WARNING: more than 0.1% RAW pixels at white level. Consider a NEW run with lower exposure/gain.")

    print("\nCOMPLETE! Original DNGs, manifest.csv, and session.json:")
    print(args.output.resolve())
    if len(used_positions) > 1:
        print(f"Total measured baseline: {max(used_positions)-min(used_positions):g} mm")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Any completed images and manifest rows have been preserved.")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
