#!/usr/bin/env bash
# nuggetizer-verify: Validate nuggetizer batch output artifacts.
#
# Usage:
#   bash verify.sh <artifact-path> [artifact-type]
#
# artifact-type: create-output | assign-output-answers | assign-output-retrieval | metrics-output
# If omitted, auto-detected via `nuggetizer view`.

set -euo pipefail

ARTIFACT_PATH="${1:?Usage: verify.sh <artifact-path> [artifact-type]}"
ARTIFACT_TYPE="${2:-}"

# Colors (respect NO_COLOR)
if [[ -z "${NO_COLOR:-}" ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; NC=''
fi

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

FAILURES=0

# --- Basic file checks ---
echo "=== File Integrity ==="

if [[ ! -f "$ARTIFACT_PATH" ]]; then
  fail "File not found: $ARTIFACT_PATH"
  exit 1
fi
pass "File exists: $ARTIFACT_PATH"

LINE_COUNT=$(wc -l < "$ARTIFACT_PATH" | tr -d ' ')
if [[ "$LINE_COUNT" -eq 0 ]]; then
  fail "File is empty"
  exit 1
fi
pass "File has $LINE_COUNT records"

# Check every line is valid JSON
BAD_LINES=$(python3 -c "
import json, sys
bad = 0
for i, line in enumerate(open('$ARTIFACT_PATH'), 1):
    line = line.strip()
    if not line:
        continue
    try:
        json.loads(line)
    except json.JSONDecodeError:
        print(f'  Line {i}: invalid JSON', file=sys.stderr)
        bad += 1
print(bad)
")
if [[ "$BAD_LINES" -eq 0 ]]; then
  pass "All lines are valid JSON"
else
  fail "$BAD_LINES lines have invalid JSON"
fi

# --- Auto-detect artifact type if not provided ---
if [[ -z "$ARTIFACT_TYPE" ]]; then
  ARTIFACT_TYPE=$(nuggetizer view "$ARTIFACT_PATH" --output json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('resolved',{}).get('artifact_type','unknown'))" 2>/dev/null || echo "unknown")
  if [[ "$ARTIFACT_TYPE" == "unknown" ]]; then
    warn "Could not auto-detect artifact type. Skipping type-specific checks."
  else
    pass "Auto-detected artifact type: $ARTIFACT_TYPE"
  fi
fi

# --- Type-specific checks ---
echo ""
echo "=== Content Validation ($ARTIFACT_TYPE) ==="

python3 -c "
import json, sys

artifact_type = '$ARTIFACT_TYPE'
path = '$ARTIFACT_PATH'
failures = 0

records = []
for line in open(path):
    line = line.strip()
    if line:
        records.append(json.loads(line))

# Check for duplicate qids
qids = [r.get('qid') or r.get('topic_id') for r in records]
qids_clean = [q for q in qids if q is not None]
dupes = len(qids_clean) - len(set(qids_clean))
if dupes > 0:
    print(f'✗ {dupes} duplicate qid(s) found')
    failures += 1
else:
    print(f'✓ No duplicate qids ({len(qids_clean)} unique)')

if artifact_type == 'create-output':
    for i, r in enumerate(records):
        if 'nuggets' not in r:
            print(f'✗ Record {i+1}: missing nuggets array')
            failures += 1
            continue
        if len(r['nuggets']) == 0:
            print(f'✗ Record {i+1} (qid={r.get(\"qid\",\"?\")}): empty nuggets array')
            failures += 1
        for j, n in enumerate(r['nuggets']):
            if not n.get('text', '').strip():
                print(f'✗ Record {i+1}, nugget {j+1}: empty text')
                failures += 1
            imp = n.get('importance', '')
            if imp not in ('vital', 'okay', 'failed'):
                print(f'✗ Record {i+1}, nugget {j+1}: invalid importance \"{imp}\"')
                failures += 1
    if failures == 0:
        print('✓ All create-output records are well-formed')

elif artifact_type in ('assign-output-answers', 'assign-output-retrieval'):
    valid_labels_3 = {'support', 'partial_support', 'not_support'}
    valid_labels_2 = {'support', 'not_support'}
    all_labels = set()
    for i, r in enumerate(records):
        if 'nuggets' not in r:
            print(f'✗ Record {i+1}: missing nuggets array')
            failures += 1
            continue
        for j, n in enumerate(r['nuggets']):
            label = n.get('assignment', '')
            all_labels.add(label)
            if label not in valid_labels_3:
                print(f'✗ Record {i+1}, nugget {j+1}: invalid assignment \"{label}\"')
                failures += 1
    # Check mode consistency
    if all_labels <= valid_labels_2:
        print('✓ Assignment mode: 2-grade (support/not_support)')
    elif all_labels <= valid_labels_3:
        print('✓ Assignment mode: 3-grade (support/partial_support/not_support)')
    elif failures == 0:
        print(f'⚠ Mixed assignment labels: {all_labels}')
    if failures == 0:
        print('✓ All assign-output records are well-formed')

elif artifact_type == 'metrics-output':
    score_fields = ['strict_vital_score', 'strict_all_score', 'vital_score', 'all_score']
    for i, r in enumerate(records):
        for f in score_fields:
            val = r.get(f)
            if val is None:
                print(f'✗ Record {i+1}: missing {f}')
                failures += 1
            elif not (0.0 <= val <= 1.0):
                print(f'✗ Record {i+1}: {f}={val} out of [0,1] range')
                failures += 1
        # Strict ≤ non-strict
        sv = r.get('strict_vital_score', 0)
        v = r.get('vital_score', 0)
        sa = r.get('strict_all_score', 0)
        a = r.get('all_score', 0)
        if sv > v + 1e-9:
            print(f'✗ Record {i+1}: strict_vital_score ({sv}) > vital_score ({v})')
            failures += 1
        if sa > a + 1e-9:
            print(f'✗ Record {i+1}: strict_all_score ({sa}) > all_score ({a})')
            failures += 1
    if failures == 0:
        print('✓ All metrics-output records are well-formed')

sys.exit(1 if failures > 0 else 0)
" 2>&1 || FAILURES=$((FAILURES + 1))

# --- Summary ---
echo ""
echo "=== Summary ==="
if [[ "$FAILURES" -eq 0 ]]; then
  pass "All checks passed"
  exit 0
else
  fail "$FAILURES check(s) failed"
  exit 1
fi
