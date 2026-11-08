#!/usr/bin/env python3
"""MECH5720 acquisition v8: desktop-local browser UI (Firefox), camera via SSH tunnel.

It does not move the rail automatically. All DNGs are unmodified. Programmatic
analysis remains separate from the duties of the camera/acquisition operator.
"""
from __future__ import annotations
import argparse
import hmac
import json
import secrets
import threading
import webbrowser
from pathlib import Path
from html import escape
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, parse_qs
from acquisition_core import Acquisition, DEFAULT_ROOT, verify_root, package_root, save_targets, list_sessions

WEB = r'''<!doctype html>
<html lang="en-AU"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MECH5720 Acquisition v8</title><style>
:root{font-family:system-ui,'Segoe UI',Arial,sans-serif;font-size:19px;color-scheme:dark}
*{box-sizing:border-box} body{margin:0;background:#0d1522;color:#e9f2ff;line-height:1.45}
main{max-width:1430px;margin:auto;padding:20px 28px 70px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:1.65rem;margin:8px 0}h2{font-size:1.16rem;margin:0 0 13px;color:#c6dfff}
p{margin:.3rem 0 .85rem;color:#b9cade}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.card{background:#17283e;border:1px solid #36516f;border-radius:12px;padding:18px;margin-top:17px;min-width:0}
label{font-size:.85rem;display:block;margin:8px 0 4px;font-weight:600}
input,select,textarea{width:100%;padding:11px;color:#172238;background:#f6faff;border:1px solid #879cba;border-radius:7px;font:inherit}
textarea{min-height:100px}button{background:#267cc0;color:white;border:1px solid #69abe2;border-radius:7px;padding:11px 15px;font:inherit;cursor:pointer}
button.secondary{background:#304860}button:disabled{opacity:.45;cursor:not-allowed}
.flex{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.small{font-size:.82rem;color:#afc7df}.error{color:#ffe9ec;background:#622333;padding:13px;border-radius:8px;white-space:pre-wrap}
.info{background:#17344a;padding:12px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere}
pre{font-size:.78rem;color:#d4e9ff;background:#0a182a;border:1px solid #304b6b;border-radius:9px;white-space:pre-wrap;max-height:300px;overflow:auto;padding:13px}
.critical{border-left:5px solid #efba60;padding-left:12px;color:#f1d9ab}
.target{border:1px solid #627895;border-radius:8px;padding:12px;margin-top:10px}
.target .three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:9px}
img.preview{max-width:100%;max-height:350px;display:block}
@media(max-width:820px){.grid,.two,.target .three{grid-template-columns:1fr}main{padding:14px}}
</style></head><body><main>
<header><div><div class="small">MECH5720 · SYNTHETIC LIGHT-FIELD PROJECT</div><h1>RAW acquisition + experiment QC — v8</h1><p>Local Firefox controller · manual rail movement · originals preserved · resumable</p></div><div class="flex"><button class="secondary" onclick="zoom(-2)">A−</button><button class="secondary" onclick="zoom(2)">A+</button></div></header>
<div id="error" class="error" hidden></div><div id="summary" class="info">Loading application...</div>
<div class="grid">
<section class="card"><h2>1. Camera and calibration</h2>
<label>Camera tunnel</label><select id="url"><option value="http://127.0.0.1:8089">Ethernet SSH tunnel — 8089</option><option value="http://127.0.0.1:8090">Wi-Fi SSH tunnel — 8090</option></select>
<div class="flex"><button id="connect" onclick="action('connect',{url:get('url')})">Connect / reconnect Pi</button></div>
<div class="two"><div><label>Exposure (microseconds)</label><input id="exposure" type="number" value="32000" min="100"></div><div><label>Gain (×)</label><input id="gain" type="number" step="0.1" value="4" min="0.1"></div></div>
<div class="flex"><button id="lock" onclick="action('lock',{exposure:get('exposure'),gain:get('gain')})">Apply and lock controls</button><button class="secondary" id="test" onclick="action('test',{root:get('root')})">Capture test DNG + PNG</button></div>
<p class="small">The test creates a RAW and preview under 00_calibration. Pi control acknowledgement is NOT a measurement of the per-frame exposure.</p>
<div id="calibration"></div></section>
<section class="card"><h2>2. Experiment and measured targets</h2>
<label>Campaign data root (new or existing v8 campaign)</label><input id="root" value="__ROOT__">
<p class="small">All sessions and categories are stored below this path. This app does not touch existing v7 folders.</p>
<div class="flex"><button class="secondary" onclick="addTarget()">Add measured target</button><button class="secondary" onclick="saveTargets()">Save target measurements</button></div>
<div id="targetRows"></div>
<p class="small">For credible depth RMSE, measure depths forward from the SAME reference plane, record uncertainties, and distinguish material/texture differences.</p>
</section>
<section class="card"><h2>3. Select acquisition protocol</h2>
<div class="two"><div><label>Mode</label><select id="mode" onchange="modeChanged()"><option value="lightfield">Main light-field sweep</option><option value="snr">SNR stationary sequence</option><option value="ssim_reference">SSIM reference / registration</option><option value="reposition">Repositioning cycles</option></select></div>
<div><label>New session folder name</label><input id="name" value="lightfield_01"></div></div>
<label>Planned positions in mm (comma separated)</label><input id="positions" value="0,20,40,60,80,100,120,140,160,180">
<div class="two"><div><label>RAW images per arrival</label><input id="repeats" type="number" min="1" max="60" value="3"></div><div><label>Full movement cycles (reposition only)</label><input id="cycles" type="number" min="1" max="50" value="1"></div></div>
<div class="two"><div><label>SNR scene condition</label><select id="scene_condition"><option value="textured">Textured static scene</option><option value="uniform">Uniform illuminated reference</option><option value="dark">Dark / covered-lens reference</option></select></div>
<div><label>SSIM reference type</label><select id="reference_kind"><option value="single_view_proxy">Aligned single-view proxy (NOT ground truth)</option><option value="independent_reference">Independent reference image (document how obtained)</option><option value="registration_check">Registration consistency test</option></select></div></div>
<label>Purpose / description (required)</label><input id="description" value="Main multi-view RAW acquisition for refocus and depth validation">
<div class="two"><div><label>Lighting setup (required)</label><input id="lighting" value="Overhead room lights only; fixed" placeholder="Stable room LEDs; no rear lamp"></div>
<div><label>Estimated camera-position uncertainty (mm; required)</label><input id="position_uncertainty" type="number" step="any" min="0" value="0.5"></div></div>
<label>Depth measurement reference plane (required)</label><input id="depth_reference" value="Front face of camera lens at nominal central rail stop">
<label>Lens focus and orientation notes</label><input id="focus_notes" value="Fixed-focus stock Pi camera, not adjusted during run; orientation fixed">
<div class="two"><div><label>SNR analysis ROI — only SNR mode</label><input id="roi" placeholder="Target ID and centre rectangular region, e.g. x=600..900, y=350..600; channel samples"></div>
<div><label>SSIM reference/registration method — only SSIM mode</label><input id="reference_method" placeholder="Define reference acquisition, image registration, crop and exposure agreement"></div></div>
<label>Scene conditions, reference limitations, measurement uncertainty and notes</label><textarea id="sessionnotes" placeholder="Lighting, camera orientation, positions measured with ruler, target IDs, reference capture method..."></textarea>
<p id="modeHelp" class="critical"></p>
<button id="start" onclick="start()">Create NEW session (never overwrite)</button>
</section>
<section class="card"><h2>4. Capture the next physical arrival</h2>
<div id="next" class="info">No session active.</div>
<div class="two"><div><label>Actually measured camera x (mm)</label><input id="actualx" type="number" step="any" value="0"></div><div><label>Allowed difference from planned x (mm)</label><input id="tolerance" type="number" step="any" value="3"></div></div>
<label>Notes on this arrival (optional)</label><input id="notes" placeholder="How position was read, vibration, unexpected change...">
<label style="display:flex;align-items:center;gap:9px"><input id="moved" type="checkbox" style="width:23px;height:23px"> I physically moved away and returned to the stop (repositioning only)</label>
<div class="flex"><button id="capture" onclick="captureStep()">Capture all repeats at this arrival</button><button class="secondary" id="recover" onclick="action('recover',{})">Recover interrupted/unindexed DNG</button><button class="secondary" onclick="action('pause',{})">Pause session</button></div>
<p class="small">Every DNG and manifest row is flushed to disk; sessions can be resumed after a program restart. An interrupted recording never overwrites an earlier file.</p>
</section>
<section class="card"><h2>5. Reopen a saved session</h2>
<label>Saved sessions under campaign root</label>
<div class="flex"><button class="secondary" onclick="findSessions()">Find saved sessions</button></div>
<select id="recent" onchange="$('resume').value=get('recent')"><option value="">Click Find saved sessions</option></select>
<label>Session folder (can also paste a path)</label><input id="resume" placeholder=".../final_experiment_v8/sessions/lightfield/lightfield_01">
<button class="secondary" onclick="action('resume',{folder:get('resume')})">Resume saved session</button>
<p class="small">Reconnect to Pi before resuming capture. The original exposure and gain will be reapplied automatically.</p></section>
<section class="card"><h2>6. Verify and hand over data</h2>
<p class="critical">Export checks all file sizes and SHA-256 checksums. It refuses to package sessions with missing, incomplete or unindexed DNGs.</p>
<div class="flex"><button class="secondary" onclick="action('verify',{root:get('root')})">Verify dataset</button><button onclick="makeZip()">Verify and build final ZIP</button></div>
<div id="report" class="info">No verification run yet.</div><p class="small">The ZIP is written on your laptop beside the campaign folder. Old v7 data are NOT included automatically; they can be archived separately.</p></section>
</div><section class="card"><h2>Activity and connection recovery</h2><pre id="log"></pre></section>
</main><script>
'use strict';const TOKEN='__TOKEN__';let running=false,st={};const $=id=>document.getElementById(id);const get=id=>$(id).value.trim();
function zoom(delta){let n=parseInt(localStorage.getItem('v8zoom')||'19',10);n=Math.max(15,Math.min(34,n+delta));document.documentElement.style.fontSize=n+'px';localStorage.setItem('v8zoom',n);}
document.documentElement.style.fontSize=(localStorage.getItem('v8zoom')||'19')+'px';
function message(s){$('error').textContent=s;$('error').hidden=!s;}
async function action(kind,payload={}){if(running)return;running=true;message('');
 try{const res=await fetch('/api/'+kind,{method:'POST',headers:{'Content-Type':'application/json','X-MECH5720-Token':TOKEN},body:JSON.stringify(payload)});const body=await res.json();
 if(!res.ok||!body.ok)throw new Error(body.error||'Request failed');if(body.state)refresh(body.state);if(kind==='verify'||kind==='export')$('report').textContent=JSON.stringify(body.result,null,2);return body.result||{};
 }catch(e){message(String(e.message||e));try{await loadState()}catch(_){}}finally{running=false;refresh(st)}}
async function loadState(){const r=await fetch('/api/state');const b=await r.json();refresh(b.state)}
function refresh(s){st=s||{};const act=s.session;const has=Boolean(act);const step=act?.next_step;
 $('summary').textContent=`Camera: ${s.connected?'connected':'disconnected'} at ${s.url||'—'}  |  Settings: ${s.locked?`${s.locked.exposure_us} us / ${s.locked.gain}x, AE off`:'not locked'}\n`+(has?`Session: ${act.mode}, ${act.rows}/${act.expected} DNGs, ${act.complete_steps}/${act.steps} arrivals`:'No open session');
 $('next').textContent=step?`Arrival ${step.step_id}/${act.steps} | cycle ${step.cycle_index} | planned x=${step.nominal_x_mm} mm | already saved ${step.already_captured}/${step.repeats} DNGs`:(has?'Session complete — pause or verify.':'No active session.');
 if(step && $('actualx').dataset.step!==String(step.step_id)){$('actualx').value=step.nominal_x_mm;$('actualx').dataset.step=String(step.step_id);$('moved').checked=false;}
 $('capture').disabled=running||!s.connected||!step||Boolean(act?.orphan_files?.length);$('lock').disabled=running||!s.connected||has;
 $('test').disabled=running||!s.locked||has;$('start').disabled=running||!s.locked||has;
 $('recover').disabled=running||!has||!act.orphan_files.length;
 $('log').textContent=(s.log||[]).join('\n');$('log').scrollTop=$('log').scrollHeight;
 if(s.calibration){$('calibration').textContent=`Latest test DNG: ${s.calibration.dng}\nRAW median: ${s.calibration.stats?.median_dn} DN, clipped: ${s.calibration.stats?.saturated_pct}%`}
}
function modeChanged(){const m=get('mode');let options={lightfield:['lightfield_01','0,20,40,60,80,100,120,140,160,180',3,1,'Capture all viewpoints with fixed controls and static targets. Subsets give baseline/count comparisons; no extra hardware configs required.','Main multi-view RAW acquisition'],snr:['snr_uniform_01','90',16,1,'Do not move the camera! Capture repeated RAWs over the exact same illuminated target/ROI. Take separate dark reference if agreed with analyst.','Stationary temporal RAW-noise measurement'],ssim_reference:['ssim_reference_01','90',3,1,'SSIM needs a justified aligned reference defined by your processing/evaluation colleagues. A single camera image is NOT automatically ground truth for computational refocus.','SSIM/reference acquisition with defined reference type'],reposition:['reposition_01','0,90,180',1,5,'Move physically between stops each cycle and record actual measured x. Ten captures without physical movement do not measure reposition accuracy.','Repeated rail traversal, movement and return']}[m];
 $('name').value=options[0];$('positions').value=options[1];$('repeats').value=options[2];$('cycles').value=options[3];$('modeHelp').textContent=options[4];$('description').value=options[5];}
function start(){const data={root:get('root'),name:get('name'),mode:get('mode'),positions:get('positions'),repeats:get('repeats'),cycles:get('cycles'),scene_condition:get('scene_condition'),reference_kind:get('reference_kind'),description:get('description'),notes:get('sessionnotes'),lighting:get('lighting'),position_uncertainty_mm:get('position_uncertainty'),depth_reference:get('depth_reference'),focus_notes:get('focus_notes'),roi_definition:get('roi'),reference_method:get('reference_method')};
 if(!confirm(`Create NEW ${data.mode} session?\n${data.name}\n\nThe rail MUST be moved manually, and measured positions entered at each arrival.`))return;action('start',data)}
function captureStep(){const step=st.session?.next_step;if(!step)return;const data={actual_x_mm:get('actualx'),tolerance_mm:get('tolerance'),notes:get('notes'),moved_away_and_returned:$('moved').checked};
 if(!confirm(`Is camera actually at ${data.actual_x_mm} mm?\nCapture remaining DNGs for arrival ${step.step_id}, keeping everything stationary?`))return;action('capture',data)}
async function findSessions(){const result=await action('sessions',{root:get('root')});if(!result)return;
 const sel=$('recent');sel.replaceChildren();for(const item of result){const opt=document.createElement('option');opt.value=item.folder;opt.textContent=`${item.mode||'Unknown'} | ${item.saved||0}/${item.expected||'?'} RAWs | ${item.folder}`;sel.append(opt)}
 if(result.length){sel.value=result.find(x=>!x.complete)?.folder||result[result.length-1].folder;$('resume').value=sel.value;}}
function makeZip(){if(confirm('Hash-check every RAW and create handover ZIP on this laptop? This may take some time.'))action('export',{root:get('root')})}
const T=['target_id','description','depth_mm','depth_uncertainty_mm','depth_reference','material','surface_finish','printed_texture','reflectivity_notes','plane_notes'];
function addTarget(){const wrap=document.createElement('div');wrap.className='target';wrap.innerHTML='<div class="three">'+T.map((k,i)=>`<div><label>${k.replaceAll('_',' ')}</label><input data-k="${k}" placeholder="${['target_01','Printed checkerboard','1000','2','Front lens surface','Matte paper','Matte','High contrast random dots','Low glare','Frontoparallel'][i]}"></div>`).join('')+'</div>'; $('targetRows').append(wrap)}
function saveTargets(){const rows=[...$('targetRows').children].map(c=>Object.fromEntries([...c.querySelectorAll('input')].map(x=>[x.dataset.k,x.value.trim()])));action('targets',{root:get('root'),items:rows})}
addTarget();modeChanged();loadState().catch(e=>message(e.message));
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def jsonout(self, code, data):
        body=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)))
        self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path=="/":
            data=WEB.replace("__TOKEN__",escape(self.server.secret,quote=True)).replace("__ROOT__",escape(str(DEFAULT_ROOT),quote=True)).encode()
            self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(data)))
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(data)
        elif self.path=="/api/state":
            with self.server.controller.lock: self.jsonout(200,{"ok":True,"state":self.server.controller.state()})
        elif self.path=="/favicon.ico":self.send_response(204);self.end_headers()
        else:self.send_error(404)
    def do_POST(self):
        if not self.path.startswith('/api/'):
            self.send_error(404);return
        if not hmac.compare_digest(self.headers.get('X-MECH5720-Token',''),self.server.secret):
            self.jsonout(403,{"ok":False,"error":"Incorrect local session token"});return
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':
            self.jsonout(415,{"ok":False,"error":"JSON required"});return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<200000:raise ValueError('Request too large')
            obj=json.loads(self.rfile.read(n))
            if not isinstance(obj,dict):raise ValueError('JSON object required')
            controller=self.server.controller
            with controller.lock:
                action=self.path[5:]
                if action=='connect':res=controller.connect(obj['url'])
                elif action=='lock':res=controller.lock_camera(obj['exposure'],obj['gain'])
                elif action=='test':res=controller.test_capture(obj['root'])
                elif action=='start':res=controller.start(obj)
                elif action=='resume':res=controller.resume(obj['folder'])
                elif action=='capture':res=controller.capture_step(obj)
                elif action=='recover':res=controller.recover_orphans()
                elif action=='pause':res=controller.pause()
                elif action=='verify':
                    report=verify_root(obj['root']);res={k:v for k,v in report.items() if k!='master_rows'}
                elif action=='export':res=package_root(obj['root'])
                elif action=='targets':res=save_targets(obj['root'],obj['items'])
                elif action=='sessions':res=list_sessions(obj['root'])
                else:raise ValueError('Unknown action')
                self.jsonout(200,{"ok":True,"result":res,"state":controller.state()})
        except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
            self.jsonout(400,{"ok":False,"error":str(exc)})
        except Exception as exc:
            self.jsonout(500,{"ok":False,"error":f"Unexpected failure: {exc}"})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--no-browser',action='store_true')
    args=p.parse_args()
    secret=secrets.token_urlsafe(32)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.controller=Acquisition();server.secret=secret
    print(f'MECH5720 v8 experiment acquisition: http://127.0.0.1:{args.port}')
    print('Keep this terminal open. Ctrl+C ends web app, never overwrites acquisition files.')
    if not args.no_browser: threading.Timer(.7,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
