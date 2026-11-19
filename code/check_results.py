#!/usr/bin/env python3
"""Validate a reproduction against fixed manuscript results."""
from pathlib import Path
import json, math
root=Path(__file__).resolve().parent.parent
expected=json.loads((root/'results'/'UNSW_Benchmark_Verified_Results.json').read_text())
actual=json.loads((root/'UNSW_Benchmark_Metrics.json').read_text())
assert actual['sourceRows']==expected['source_records']
assert actual['cleanedRows']==expected['unique_consistently_labeled_records']
assert math.isclose(actual['threshold'],expected['calibrated_operating_threshold'],abs_tol=1e-12)
assert actual['partitions']['test']['n']==expected['split_counts']['test']['records']
for key,target in [('logistic','logistic'),('boostedStumps','boosted_stumps'),('calibratedFrozen','calibrated_frozen')]:
 m=actual['models'][key];e=expected['metrics'][target]
 for k in ('tn','fp','fn','tp','acc','precision','recall','f1','fpr','auroc','brier'):
  v=e['accuracy'] if k=='acc' else e[k]
  assert math.isclose(m[k],v,rel_tol=1e-9,abs_tol=1e-9),(key,k,m[k],v)
for k,other in [('rawECE','raw'),('calibratedECE','after_isotonic')]:
 assert math.isclose(actual['calibration'][k],expected['expected_calibration_error'][other],rel_tol=1e-9)
assert len(actual['responsePolicies'])==4
for p,e in zip(actual['responsePolicies'],expected['response_policy_simulation']):
 assert math.isclose(p['attackCoveragePct'],e['attack_intervention_pct'],abs_tol=1e-9)
 assert math.isclose(p['benignActionPct'],e['benign_intervention_pct'],abs_tol=1e-9)
 assert math.isclose(p['cost'],e['cost_per_record'],abs_tol=1e-9)
print('PASS: exact reproducibility of class counts, detection, calibration, and policy results')
