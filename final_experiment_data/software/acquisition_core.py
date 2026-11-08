#!/usr/bin/env python3
"""MECH5720 v8 acquisition core. Operates on laptop; no remote motion control.

Modes deliberately separate capture protocols from downstream analysis. DNG files
are saved byte-for-byte; original files and historical sessions are never changed.
"""
from __future__ import annotations
import base64
import shutil
import sys
import csv
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from legacy_capture_backend import api, get_setting, set_manual, metrics_from_dng

MODES = {"lightfield", "snr", "ssim_reference", "reposition"}
SESSION_FIELDS = [
    "step_id", "mode", "cycle_index", "position_index", "repeat_index",
    "nominal_x_mm", "x_mm", "timestamp_utc", "filename", "bytes", "sha256",
    "exposure_us_requested", "gain_requested", "ae_requested", "actual_exposure_us",
    "actual_gain", "frame_duration_us", "sensor_timestamp_ns", "capture_metadata_source",
    "raw_height", "raw_width", "black_levels_dn", "white_level_dn", "min_dn",
    "median_dn", "p99_dn", "max_dn", "saturated_pct", "near_black_pct",
    "scene_condition", "reference_kind", "moved_away_and_returned", "notes", "recovered_orphan"
]
TARGET_FIELDS = ["target_id", "description", "depth_mm", "depth_uncertainty_mm", "depth_reference",
                 "material", "surface_finish", "printed_texture", "reflectivity_notes", "plane_notes"]
DEFAULT_ROOT = Path.home() / "Documents/OneDrive/2026/Semester 2/MECH5720/LightField_Data/final_experiment_v8"
METADATA_HEADER = "X-MECH5720-Capture-Metadata"


def stamp(): return datetime.now(timezone.utc).isoformat()


def safe_label(value):
    s = str(value or "").strip()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", s):
        raise ValueError("Use a folder name with letters, numbers, underscores or dashes only (max 64).")
    return s


def local_url(value):
    p = urlsplit(str(value).strip().rstrip("/"))
    if (p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost") or
        p.port not in (8089, 8090) or p.path not in ("", "/") or p.username or
        p.password or p.query or p.fragment):
        raise ValueError("Camera URL must be http://127.0.0.1:8089 (Ethernet) or :8090 (Wi-Fi).")
    return f"http://127.0.0.1:{p.port}"


def finite_number(v, name, min_value=None):
    x = float(v)
    if not math.isfinite(x) or min_value is not None and x < min_value:
        raise ValueError(f"{name} must be a finite number" + (f" >= {min_value}" if min_value is not None else ""))
    return x


def write_json_atomic(path, obj):
    path = Path(path)
    part = path.with_name(path.name + ".writing")
    if part.exists(): raise RuntimeError(f"Temporary metadata file exists: {part}")
    with part.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(part, path)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames or []) != SESSION_FIELDS:
            raise RuntimeError("Session manifest schema mismatch; refusing to resume")
        return list(reader)


def append_row(path, row):
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SESSION_FIELDS, extrasaction="ignore")
        writer.writerow(row)
        f.flush(); os.fsync(f.fileno())


def build_steps(mode, positions, repeats, cycles, scene_condition, reference_kind):
    if mode not in MODES: raise ValueError("Unknown experiment mode")
    if not 1 <= repeats <= 60: raise ValueError("Captures per arrival must be 1–60")
    if not 1 <= cycles <= 50: raise ValueError("Cycle count must be 1–50")
    if not positions or len(positions) > 50: raise ValueError("Provide 1–50 rail positions")
    positions = [finite_number(p, "position") for p in positions]
    if len(set(positions)) != len(positions): raise ValueError("Each listed nominal rail stop must be distinct")
    if mode == "lightfield" and len(positions) < 2: raise ValueError("Light-field mode needs two or more positions")
    if mode in ("snr", "ssim_reference") and len(positions) != 1:
        raise ValueError("SNR and SSIM reference sessions use one stationary position")
    if mode != "reposition" and cycles != 1:
        raise ValueError("Multiple movement cycles are only for reposition mode")
    if mode == "reposition" and cycles < 2:
        raise ValueError("Repositioning test requires at least 2 movement cycles")
    if mode == "reposition" and len(positions) < 2:
        raise ValueError("Use at least 2 stops so each cycle requires physical movement")
    if mode == "snr" and scene_condition not in ("uniform", "dark", "textured"):
        raise ValueError("SNR must be labelled uniform, dark, or textured")
    if mode == "ssim_reference" and reference_kind not in ("single_view_proxy", "independent_reference", "registration_check"):
        raise ValueError("Select a documented SSIM reference type, not an undefined ground truth")
    steps=[]
    for cycle in range(1, cycles + 1):
        for idx, x in enumerate(positions, 1):
            steps.append({"step_id":len(steps)+1,"cycle_index":cycle,
                          "position_index":idx,"nominal_x_mm":x,"repeats":repeats})
    if len(steps)*repeats > 3000: raise ValueError("The session exceeds the 3,000-DNG safety limit")
    return steps


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def capture_dng(base, path):
    """Capture to .part, validate Bayer with rawpy, then atomically rename.

    Optional metadata header only exists with the updated Pi server. Existing
    server is fully compatible; actual exposure fields will then be blank.
    """
    path=Path(path)
    if path.exists() or path.with_name(path.name+".part").exists():
        raise FileExistsError(f"Capture already exists or is incomplete: {path}")
    req=Request(base + "/capture_dng", method="GET")
    with urlopen(req, timeout=100) as resp:
        metadata={}
        encoded=resp.headers.get(METADATA_HEADER)
        if encoded:
            try: metadata=json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")+b"="*((-len(encoded))%4)))
            except Exception: metadata={}
        dng=resp.read()
    if len(dng)<100_000: raise RuntimeError("Pi did not return a plausible DNG file")
    part=path.with_name(path.name+".part")
    try:
        with part.open("xb") as f:
            f.write(dng); f.flush(); os.fsync(f.fileno())
        stats=metrics_from_dng(part)
        os.replace(part,path)
    except Exception:
        # Keep the raw response for diagnosis; never classify as valid data.
        if part.exists():
            failed=path.with_name(path.name+".failed")
            if not failed.exists(): os.replace(part,failed)
        raise
    return stats,len(dng),hashlib.sha256(dng).hexdigest(),metadata


def metadata_columns(meta):
    def val(k):
        x=meta.get(k)
        return x if isinstance(x,(int,float)) and not isinstance(x,bool) else ""
    return {"actual_exposure_us":val("ExposureTime"),"actual_gain":val("AnalogueGain"),
            "frame_duration_us":val("FrameDuration"),"sensor_timestamp_ns":val("SensorTimestamp"),
            "capture_metadata_source":"Pi frame metadata header" if meta else "not available (requested settings only)"}


def preview_dng(dng, png):
    import rawpy
    from PIL import Image
    with rawpy.imread(str(dng)) as raw:
        rgb=raw.postprocess(use_camera_wb=False,no_auto_bright=True,output_bps=8,gamma=(2.222,4.5))
    pic=Image.fromarray(rgb)
    pic=pic.transpose(Image.Transpose.ROTATE_270) # preview-only; RAW stays untouched
    pic.thumbnail((1200,1200),Image.Resampling.LANCZOS)
    if Path(png).exists(): raise FileExistsError(f"Preview exists: {png}")
    pic.save(png,"PNG")


class Acquisition:
    def __init__(self):
        self.lock=threading.RLock()
        self.url="http://127.0.0.1:8089"
        self.connected=False
        self.locked=None
        self.root=None
        self.active=None
        self.log=[]
        self.calibration=None
        self._log("Ready. Start with Camera connect, then set and lock controls.")

    def _log(self, message):
        self.log.append(f"{datetime.now().strftime('%H:%M:%S')}  {message}")
        self.log=self.log[-120:]

    def state(self):
        session=None
        if self.active:
            m=self.active["meta"]
            rows=self.active["rows"]
            next_step=self.next_step()
            session={"folder":str(self.active["folder"]),"mode":m["mode"],
                     "rows":len(rows),"expected":sum(s["repeats"] for s in m["steps"]),
                     "steps":len(m["steps"]),"complete_steps":sum(1 for s in m["steps"] if
                     sum(r["step_id"]==str(s["step_id"]) for r in rows)==s["repeats"]),
                     "next_step":next_step,"complete":next_step is None,
                     "orphan_files":self.orphans()}
        return {"connected":self.connected,"url":self.url,"locked":self.locked,
                "root":str(self.root) if self.root else "","session":session,
                "calibration":self.calibration,"log":list(self.log)}

    def connect(self, url):
        self.url=local_url(url)
        config=api(self.url,"/config",timeout=10)
        if (config.get("width"),config.get("height"))!=(1920,1080):
            res=api(self.url,"/config",{"width":1920,"height":1080,"format":"RGB888"},timeout=45)
            if res.get("status")!="reconfigured": raise RuntimeError("Pi did not accept 1920x1080")
            time.sleep(0.8)
        api(self.url,"/controls",timeout=10)
        self.connected=True
        self.locked=None
        self._log("Connected to Raspberry Pi; configured 1920x1080 RGB888")
        return self.state()

    def lock_camera(self, exposure, gain):
        if self.active: raise ValueError("Finish or pause the active session before changing controls")
        if not self.connected: raise ValueError("Connect to the camera first")
        exp=int(exposure); g=finite_number(gain,"gain",0.1)
        if not 100 <= exp <= 10000000: raise ValueError("Check exposure value in microseconds")
        set_manual(self.url,exp,g)
        self.locked={"exposure_us":exp,"gain":g,"ae":False}
        self._log(f"Manual controls requested and acknowledged: {exp} us, {g:g}x, AE off")
        return self.state()

    def test_capture(self, root):
        if not self.connected or not self.locked: raise ValueError("Connect and lock settings first")
        if self.active: raise ValueError("Cannot calibrate in an active acquisition session")
        folder=Path(root).expanduser().resolve()/"00_calibration"
        folder.mkdir(parents=True,exist_ok=True)
        ident=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dng=folder/f"test_{ident}.dng"
        stats,nbytes,sha,metadata=capture_dng(self.url,dng)
        png=dng.with_suffix(".png")
        try: preview_dng(dng,png)
        except Exception as exc: self._log(f"Preview warning (RAW saved): {exc}")
        report={"timestamp":stamp(),"file":dng.name,"sha256":sha,"bytes":nbytes,
                "settings_requested":self.locked,"stats":stats,"frame_metadata":metadata}
        write_json_atomic(folder/f"test_{ident}.json",report)
        self.root=folder.parent
        self.calibration={"png":str(png) if png.exists() else "", "dng":str(dng),"stats":stats}
        self._log(f"Test RAW saved: {dng.name}, median={stats['median_dn']} DN")
        return self.state()

    def start(self, spec):
        if self.active: raise ValueError("Close/pause the current session first")
        if not self.connected or not self.locked: raise ValueError("Connect camera and lock controls first")
        root=Path(str(spec["root"])).expanduser().resolve()
        name=safe_label(spec["name"])
        mode=str(spec["mode"])
        position_text=str(spec.get("positions", ""))
        try: positions=[float(p.strip()) for p in position_text.split(",") if p.strip()]
        except ValueError: raise ValueError("Positions must be comma-separated numbers (millimetres)")
        repeats=int(spec.get("repeats",1)); cycles=int(spec.get("cycles",1))
        condition=str(spec.get("scene_condition","textured"))
        reference=str(spec.get("reference_kind","single_view_proxy"))
        steps=build_steps(mode,positions,repeats,cycles,condition,reference)
        description=str(spec.get("description","")).strip()
        if not description: raise ValueError("Describe the purpose and physical setup of this session")
        lighting=str(spec.get("lighting","")).strip()
        if not lighting: raise ValueError("Record the illumination configuration")
        depth_reference=str(spec.get("depth_reference","")).strip()
        if not depth_reference: raise ValueError("Record the common target-depth reference plane")
        position_uncertainty=finite_number(spec.get("position_uncertainty_mm",""),"position uncertainty (mm)",0)
        roi=str(spec.get("roi_definition","")).strip()
        if mode=="snr" and not roi:
            raise ValueError("SNR mode requires a repeatable ROI definition (describe the region or pixel bounds)")
        reference_method=str(spec.get("reference_method","")).strip()
        if mode=="ssim_reference" and not reference_method:
            raise ValueError("SSIM mode requires the planned reference and image-registration method")
        session=root/"sessions"/mode/name
        if session.exists(): raise FileExistsError(f"Existing session will NEVER be overwritten: {session}")
        if not root.exists(): root.mkdir(parents=True,exist_ok=True)
        initialise_campaign(root)
        session.mkdir(parents=True,exist_ok=False)
        (session/"raw").mkdir()
        (session/"previews").mkdir()
        meta={"schema_version":"8.0","session_created_utc":stamp(),"mode":mode,"name":name,
              "description":description,"steps":steps,"scene_condition":condition,
              "geometry":{"axis":"horizontal camera translation","nominal_baseline_mm":max(positions)-min(positions),
                          "position_uncertainty_mm":position_uncertainty,
                          "depth_reference":depth_reference,"depth_uncertainty_documented_in":"targets.csv"},
              "lighting":lighting,"focus":str(spec.get("focus_notes","fixed focus, unchanged during run")),
              "roi_definition":roi if mode=="snr" else "",
              "reference_method":reference_method if mode=="ssim_reference" else "",
              "reference_kind":reference if mode=="ssim_reference" else "",
              "controls_requested":dict(self.locked),"api_url":self.url,
              "camera_config":{"width":1920,"height":1080,"format":"RGB888"},
              "notes":str(spec.get("notes","")),
              "protocol_notes": {
                  "snr":"Temporal RAW SNR needs fixed camera/lighting and a documented ROI; use separate uniform/dark sessions if needed.",
                  "ssim_reference":"A captured reference is NOT automatically optical ground truth; agree on reference/registration with processing owners.",
                  "reposition":"Physically leave and return to each stop between cycles; log actual measured x and mark motion confirmation.",
                  "lightfield":"Full static scene captured at multiple measured x with unchanged exposure, gain, orientation, and focus."}[mode]}
        write_json_atomic(session/"session.json",meta)
        with (session/"manifest.csv").open("x",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,SESSION_FIELDS).writeheader();f.flush();os.fsync(f.fileno())
        self.root=root
        self.active={"folder":session,"meta":meta,"rows":[]}
        self._log(f"Started {mode}/{name}: {len(steps)} arrivals, {sum(s['repeats'] for s in steps)} RAW captures")
        return self.state()

    def resume(self, folder):
        if self.active: raise ValueError("Close/pause active session before resuming another")
        folder=Path(str(folder)).expanduser().resolve()
        meta=json.loads((folder/"session.json").read_text(encoding="utf-8"))
        if meta.get("schema_version")!="8.0" or meta.get("mode") not in MODES:
            raise ValueError("Not a recognised v8 experiment session")
        rows=load_csv(folder/"manifest.csv")
        self.active={"folder":folder,"meta":meta,"rows":rows}
        self.root=folder.parents[2] # <root>/sessions/<mode>/<name>
        # Resume must reassert exactly the settings used by the original run.
        wanted=meta["controls_requested"]
        self.locked=dict(wanted)
        self._log(f"Loaded {len(rows)} records; camera connection must be checked before continuing")
        if self.connected:
            set_manual(self.url,wanted["exposure_us"],wanted["gain"])
        return self.state()

    def orphans(self):
        if not self.active: return []
        folder=self.active["folder"]
        indexed={r["filename"] for r in self.active["rows"]}
        return sorted(str(p.relative_to(folder)) for p in (folder/"raw").glob("*.dng") if str(p.relative_to(folder)) not in indexed)

    def next_step(self):
        if not self.active: return None
        counts=Counter(int(r["step_id"]) for r in self.active["rows"])
        for step in self.active["meta"]["steps"]:
            if counts[step["step_id"]] < step["repeats"]:
                return {**step,"already_captured":counts[step["step_id"]]}
        return None

    def _check_current(self):
        if not self.connected: raise ValueError("Reconnect the Pi first")
        cfg=api(self.url,"/config",timeout=15)
        if (cfg.get("width"),cfg.get("height")) != (1920,1080):
            raise RuntimeError("Camera resolution changed; no captures written")

    def capture_step(self, details):
        if not self.active: raise ValueError("Create or resume a session first")
        if self.orphans(): raise RuntimeError("Unindexed DNGs detected! Use Recover orphans before continuing.")
        step=self.next_step()
        if step is None: raise ValueError("This session has all its expected RAW captures")
        x=finite_number(details.get("actual_x_mm"),"measured x_mm")
        tolerance=finite_number(details.get("tolerance_mm",3),"position tolerance",0)
        if abs(x-step["nominal_x_mm"])>tolerance:
            raise ValueError(f"Measured x={x:g} mm is far from planned {step['nominal_x_mm']:g} mm. Check your stop/tolerance.")
        moved=details.get("moved_away_and_returned") is True
        if self.active["meta"]["mode"]=="reposition" and step["cycle_index"]>1 and not moved:
            raise ValueError("For cycle 2+, tick 'I physically moved away and returned'.")
        notes=str(details.get("notes", "")).strip()
        self._check_current()
        meta=self.active["meta"]
        exp=meta["controls_requested"]["exposure_us"]
        gain=meta["controls_requested"]["gain"]
        set_manual(self.url,exp,gain)
        time.sleep(0.3)
        folder=self.active["folder"]
        # Durable physical-position declaration supports orphan recovery.
        inflight={"step_id":step["step_id"],"actual_x_mm":x,"notes":notes,
                  "moved_away_and_returned":moved,"timestamp":stamp()}
        write_json_atomic(folder/"inflight.json",inflight)
        prior=[r for r in self.active["rows"] if int(r["step_id"])==step["step_id"]]
        n_existing=len(prior)
        prior_reps=sorted(int(r["repeat_index"]) for r in prior)
        if prior_reps != list(range(1,n_existing+1)):
            raise RuntimeError("Partial step has non-contiguous/duplicate repeat indices; inspect manifest before resuming")
        if prior:
            recorded_xs={float(r["x_mm"]) for r in prior}
            if recorded_xs != {x}:
                raise ValueError("Partial arrival was captured at a different measured x. Do not mix positions in one step.")
        for rep in range(n_existing+1,step["repeats"]+1):
            file=f"raw/step_{step['step_id']:03d}_x{step['nominal_x_mm']:+09.3f}mm_cycle{step['cycle_index']:02d}_rep{rep:03d}.dng"
            dng=folder/file
            stats,nbytes,sha,frame_meta=capture_dng(self.url,dng)
            row={"step_id":step["step_id"],"mode":meta["mode"],"cycle_index":step["cycle_index"],
                 "position_index":step["position_index"],"repeat_index":rep,
                 "nominal_x_mm":step["nominal_x_mm"],"x_mm":x,"timestamp_utc":stamp(),
                 "filename":file,"bytes":nbytes,"sha256":sha,"exposure_us_requested":exp,
                 "gain_requested":gain,"ae_requested":False,**metadata_columns(frame_meta),**stats,
                 "scene_condition":meta["scene_condition"],"reference_kind":meta["reference_kind"],
                 "moved_away_and_returned":moved,"notes":notes,"recovered_orphan":False}
            append_row(folder/"manifest.csv",row)
            self.active["rows"].append({k:str(v) for k,v in row.items()})
            self._log(f"Saved step {step['step_id']} repeat {rep}/{step['repeats']}: {dng.name}")
            if rep==1:
                png=folder/"previews"/(dng.stem+".png")
                try: preview_dng(dng,png)
                except Exception as exc: self._log(f"Preview warning (DNG is safe): {exc}")
        inflight_path=folder/"inflight.json"
        if inflight_path.exists(): inflight_path.unlink()
        self._log(f"Arrival complete: {step['step_id']} / {len(meta['steps'])}")
        return self.state()

    def recover_orphans(self):
        if not self.active: raise ValueError("Load an acquisition first")
        folder=self.active["folder"]
        missing=self.orphans()
        if not missing: return self.state()
        fp=folder/"inflight.json"
        if not fp.exists(): raise RuntimeError("No physical position checkpoint to safely recover orphan data")
        info=json.loads(fp.read_text())
        step=next((s for s in self.active["meta"]["steps"] if s["step_id"]==info["step_id"]),None)
        if not step: raise RuntimeError("Checkpoint does not identify a planned step")
        patt=re.compile(r"^raw/step_(\d+)_.*_cycle(\d+)_rep(\d+)\.dng$")
        for name in missing:
            match=patt.fullmatch(name)
            if not match or (int(match[1]),int(match[2])) != (step["step_id"],step["cycle_index"]):
                raise RuntimeError(f"Cannot safely attribute orphan: {name}")
            rep=int(match[3])
            if not 1<=rep<=step["repeats"]: raise RuntimeError("Orphan repeat outside plan")
            p=folder/name
            stats=metrics_from_dng(p)
            st=self.active["meta"]
            row={"step_id":step["step_id"],"mode":st["mode"],"cycle_index":step["cycle_index"],
                 "position_index":step["position_index"],"repeat_index":rep,
                 "nominal_x_mm":step["nominal_x_mm"],"x_mm":info["actual_x_mm"],
                 "timestamp_utc":stamp(),"filename":name,"bytes":p.stat().st_size,"sha256":file_hash(p),
                 "exposure_us_requested":st["controls_requested"]["exposure_us"],
                 "gain_requested":st["controls_requested"]["gain"],"ae_requested":False,
                 **metadata_columns({}),**stats,"scene_condition":st["scene_condition"],
                 "reference_kind":st["reference_kind"],"moved_away_and_returned":info["moved_away_and_returned"],
                 "notes":f"RECOVERED: {info['notes']}","recovered_orphan":True}
            append_row(folder/"manifest.csv",row)
            self.active["rows"].append({k:str(v) for k,v in row.items()})
            self._log(f"Recovered DNG not indexed before interruption: {name}")
        if self.next_step() is None or self.next_step()["step_id"] != step["step_id"]:
            fp.unlink(missing_ok=True)
        return self.state()

    def pause(self):
        self._log("Session paused; directory remains resumable by the Resume action")
        self.active=None
        self.locked=None
        return self.state()


def verify_root(root, verify_hashes=True):
    root=Path(root).expanduser().resolve()
    issues=[]; warns=[]; totals={"sessions":0,"expected_dng":0,"recorded_dng":0,"verified_dng":0}
    rows_all=[]; session_status=[]
    if not root.exists(): raise FileNotFoundError(root)
    sessions=sorted(root.glob("sessions/*/*/session.json"))
    if not sessions: issues.append("No v8 sessions under sessions/<mode>/<name>")
    for path in sessions:
        folder=path.parent
        try:
            spec=json.loads(path.read_text(encoding="utf-8"))
            rows=load_csv(folder/"manifest.csv")
            if spec["schema_version"]!="8.0": raise ValueError("Wrong schema version")
            expected=sum(int(s["repeats"]) for s in spec["steps"])
            totals["sessions"]+=1;totals["expected_dng"]+=expected; totals["recorded_dng"]+=len(rows)
            if len(rows)!=expected: issues.append(f"Incomplete session {folder}: {len(rows)}/{expected} DNG rows")
            seen=set(); counts=Counter();
            for row in rows:
                f=row["filename"]
                if f in seen: issues.append(f"Duplicate manifest filename: {folder}/{f}")
                seen.add(f)
                counts[int(row["step_id"])]+=1
                target=(folder/f).resolve()
                if not target.is_relative_to(folder.resolve()):
                    issues.append(f"Path outside session: {f}");continue
                if not target.is_file(): issues.append(f"Missing DNG: {target}");continue
                if target.stat().st_size!=int(row["bytes"]):issues.append(f"File size mismatch: {target}");continue
                if verify_hashes and file_hash(target)!=row["sha256"]: issues.append(f"SHA-256 mismatch: {target}");continue
                totals["verified_dng"]+=1
                rows_all.append({"session":str(folder.relative_to(root)),**row})
            for s in spec["steps"]:
                if counts[s["step_id"]]!=s["repeats"]:
                    issues.append(f"{folder}: step {s['step_id']} has {counts[s['step_id']]}/{s['repeats']} captures")
            for orphan in (folder/"raw").glob("*.dng"):
                if str(orphan.relative_to(folder)) not in seen:
                    issues.append(f"Unindexed/orphan DNG: {orphan}")
            session_status.append({"session":str(folder.relative_to(root)),"mode":spec["mode"],
                                   "expected":expected,"recorded":len(rows)})
        except Exception as exc:
            issues.append(f"Session unreadable: {folder}: {exc}")
    for mode in ("lightfield","snr","reposition"):
        if not any(x["mode"]==mode for x in session_status):
            warns.append(f"No {mode} capture session; coordinate with evaluation owner before claiming completion")
    if not any(x["mode"]=="ssim_reference" for x in session_status):
        warns.append("No optional SSIM reference session; SSIM may use existing views if scientifically justified")
    if not (root/"targets.csv").exists(): warns.append("Missing targets.csv with measured depth uncertainties and material notes")
    return {"generated_utc":stamp(),"root":str(root),"pass":not issues,"issues":issues,
            "warnings":warns,"totals":totals,"sessions":session_status,"master_rows":rows_all}


def package_root(root, dest=None):
    """Verify every DNG then zip; refuse incomplete datasets and preexisting ZIP."""
    from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
    root=Path(root).expanduser().resolve()
    report=verify_root(root,verify_hashes=True)
    qc=root/"quality_control";qc.mkdir(parents=True,exist_ok=True)
    document={k:v for k,v in report.items() if k!="master_rows"}
    write_json_atomic(qc/"integrity_report.json",document)
    if not report["pass"]:
        raise RuntimeError("Packaging BLOCKED: " + "; ".join(report["issues"][:8]))
    master=qc/"master_manifest.csv"
    fields=["session"]+SESSION_FIELDS
    # Only rewrite derived reports, never original capture data. Avoid clobbering existing reports.
    if master.exists(): master.unlink()
    with master.open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(report["master_rows"])
    output=Path(dest).expanduser().resolve() if dest else root.parent/(root.name+"_handover.zip")
    if output.exists(): raise FileExistsError(f"Refusing to overwrite export ZIP: {output}")
    if output.is_relative_to(root): raise ValueError("Export ZIP must be outside the acquisition root")
    partial=output.with_name(output.name+".partial")
    if partial.exists(): raise FileExistsError(f"Partial ZIP exists: {partial}")
    try:
        with ZipFile(partial,"w",allowZip64=True) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    typ=ZIP_STORED if p.suffix.lower() in (".dng",".png",".zip") else ZIP_DEFLATED
                    z.write(p,arcname=f"{root.name}/{p.relative_to(root)}",compress_type=typ)
            bad=z.testzip()
            if bad: raise RuntimeError(f"ZIP integrity problem: {bad}")
        os.replace(partial,output)
    except Exception:
        # Preserve partial for diagnostics rather than risk deleting unique data.
        raise
    return {"zip":str(output),"size_bytes":output.stat().st_size,"verified":report["totals"],
            "warnings":report["warnings"]}


def save_targets(root, items):
    """Write targets with revisioned history; never silently destroy earlier measurements."""
    root=Path(root).expanduser().resolve()
    if not root.exists(): root.mkdir(parents=True,exist_ok=True)
    if not isinstance(items,list) or not items or len(items)>30:
        raise ValueError("Provide 1–30 target records")
    rows=[]; names=set()
    for obj in items:
        name=safe_label(obj.get("target_id",""))
        if name in names: raise ValueError(f"Duplicate target ID: {name}")
        names.add(name)
        depth=finite_number(obj.get("depth_mm"),"depth_mm",1)
        uncertainty=finite_number(obj.get("depth_uncertainty_mm"),"depth uncertainty",0)
        row={c:str(obj.get(c,"")).strip() for c in TARGET_FIELDS}
        row.update(target_id=name,depth_mm=depth,depth_uncertainty_mm=uncertainty)
        if not row["depth_reference"]: raise ValueError("Explain the shared depth reference (e.g. front lens plane)")
        if not row["material"] or not row["printed_texture"]:
            raise ValueError("Each target needs material and printed texture information")
        rows.append(row)
    folder=root/"targets_history";folder.mkdir(exist_ok=True)
    timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup=folder/f"targets_{timestamp}.csv"
    with backup.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=TARGET_FIELDS);writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    dest=root/"targets.csv"
    temp=dest.with_name(dest.name+".writing")
    if temp.exists(): raise FileExistsError(f"Temporary targets file exists: {temp}")
    with temp.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=TARGET_FIELDS);writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,dest)
    return {"path":str(dest),"revision":str(backup),"count":len(rows)}


CAMPAIGN_README = """MECH5720 experiment — data handover

Each sessions/<mode>/<session>/raw/*.dng is an ORIGINAL 1920x1080 Bayer image.
Do NOT edit the DNGs. The companion manifest.csv records physical x_mm, planned
stop, cycle, repeat, requested exposure/gain, optionally actual per-frame
metadata, RAW brightness diagnostics, bytes and SHA-256 hash. Filename paths
in manifest.csv are RELATIVE TO THAT SESSION directory.

Session session.json contains the protocol and planned rail arrivals.
targets.csv contains measured forward depths, uncertainty, material and texture.
previews/*.png are display-only, rotated for the viewer, never analysis sources.
quality_control/master_manifest.csv combines all sessions; inspect
quality_control/integrity_report.json before using the dataset.

Modes: lightfield (refocus/depth), snr (stationary temporal noise),
ssim_reference (reference capture whose validity must be stated),
reposition (physical repeat movements and measured errors).

Use the light-field processing owner for refocus/sharpness, the depth analysis
owner for RMSE, and evaluation owner for robustness/SNR/SSIM interpretation.
The fixed-focus camera and nominal physical stop positions can limit accuracy.
"""


def initialise_campaign(root):
    """Snapshot acquisition software the FIRST time this v8 campaign is made."""
    root=Path(root)
    readme=root/'README_DATA.txt'
    if not readme.exists():
        with readme.open('x',encoding='utf-8') as f:
            f.write(CAMPAIGN_README);f.flush();os.fsync(f.fileno())
    dest=root/'software'
    dest.mkdir(exist_ok=True)
    own=Path(__file__).resolve().parent
    for name in ('acquisition_core.py','panel_v8.py','legacy_capture_backend.py','verify_and_export.py'):
        source=own/name
        if source.is_file() and not (dest/name).exists():
            shutil.copy2(source,dest/name)
    version=dest/'software_snapshot.json'
    if not version.exists():
        files={p.name:file_hash(p) for p in sorted(dest.glob('*.py'))}
        write_json_atomic(version,{'snapshot_created_utc':stamp(),'python_version':sys.version,
                                   'software_sha256':files,
                                   'note':'Snapshot of laptop code when campaign was first created; no Pi service change implied.'})



def list_sessions(root):
    root=Path(root).expanduser().resolve()
    output=[]
    for path in sorted(root.glob('sessions/*/*/session.json')):
        try:
            meta=json.loads(path.read_text(encoding='utf-8'))
            rows=load_csv(path.parent/'manifest.csv')
            expected=sum(int(st['repeats']) for st in meta['steps'])
            output.append({'folder':str(path.parent),'mode':meta['mode'],
                           'saved':len(rows),'expected':expected,'complete':len(rows)==expected})
        except Exception as exc:
            output.append({'folder':str(path.parent),'error':str(exc)})
    return output
