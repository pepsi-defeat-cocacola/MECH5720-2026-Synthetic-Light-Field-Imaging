#!/usr/bin/env python3
"""CLI dataset audit/export. Does not require the Raspberry Pi."""
import argparse
import json
from pathlib import Path
from acquisition_core import verify_root,package_root
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--export',action='store_true',help='Build final ZIP after all  SHA-256 checks pass')
a=p.parse_args()
try:
    result=package_root(a.root) if a.export else verify_root(a.root)
    result={k:v for k,v in result.items() if k!='master_rows'}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if not a.export and not result['pass']:raise SystemExit(2)
except Exception as ex:
    print(f'ERROR: {ex}');raise SystemExit(1)
