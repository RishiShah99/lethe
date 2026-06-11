"""Print a sample of 'other'-class references from the audit manifest."""

import gzip
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "scratch/audit_manifest_drkernel.jsonl.gz"
n = 0
for line in gzip.open(path, "rt", encoding="utf-8"):
    row = json.loads(line)
    if row["op_class"] == "other":
        print(f"=== id {row['id']}")
        print(row["ref"][:500])
        n += 1
        if n == 6:
            break
