#!/usr/bin/env python3
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
files=[]
processes=[]
commands=[(['iostat','-t','-dxm','1','nvme3n1'],'iostat-timestamped.log'),
          (['mpstat','-P',','.join(str(c) for c in list(range(48))+list(range(96,112))),'5'],'mpstat-timestamped.log')]
try:
    marker=ROOT/'analysis/telemetry-start.json'
    entry=dict(
        started_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        note='Device/CPU telemetry started during the first formal curve. Per-PID pidstat covers each timed process from launch.')
    if marker.exists():
        previous=json.loads(marker.read_text())
        previous.setdefault('restarts',[]).append(dict(entry,reason='Enumerate CPU IDs for the installed mpstat; its range syntax is unsupported.'))
        entry=previous
    marker.write_text(json.dumps(entry,indent=2)+'\n')
    for cmd,name in commands:
        f=open(str(ROOT/'logs'/name),'a');files.append(f)
        processes.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT))
    while True:
        current=json.loads((ROOT/'status.json').read_text())
        if current.get('status') in ['complete','failed']:break
        time.sleep(15)
finally:
    for p in processes:p.terminate();p.wait()
    for f in files:f.close()
