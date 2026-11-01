"""Additional global-threshold comparators; never changes the adaptive policy.

Added after original high-recall results to challenge the comparison. Thresholds
are still computed exclusively on policy_validation, not on test outcomes.
"""
import json
import lightgbm as lgb
import numpy as np
from fdh_context_experiment import OUT,TMP,read,save,zscore,metrics
from decision_layer_v2 import threshold_at_targets

def main():
    x=np.load(TMP/'features.npy',mmap_mode='r');meta=dict(np.load(TMP/'metadata.npz'))
    day=meta['day'];y=meta['y'];amount=meta['amount'];pv=(day>=137)&(day<=149);te=day>=157
    model=lgb.Booster(model_file=str(OUT/'context_model.txt'));cal=read('context_calibration.json')
    s=zscore(model,x[pv],cal);st=zscore(model,x[te],cal)
    original=read('test_results.json');base=read('selected_policies.json')['baseline_validation']
    results={};preds={}
    conditions=[('same_targets',.85,.95),('same_validation_outcomes',(base['tp']-1e-9)/y[pv].sum(),
        1-base['unflagged_fraud_amount']/base['fraud_amount'])]
    for name,c,v in conditions:
        threshold,validation=threshold_at_targets(s,y[pv],amount[pv],c,v)
        action=st>=threshold;actual=metrics(y[te],amount[te],action);ad=original['policies']['adaptive']
        results[name]=dict(threshold=threshold,validation=validation,test=actual,adaptive_vs_this=dict(
            fp_reduction=1-ad['fp']/actual['fp'],tp_difference=ad['tp']-actual['tp'],
            unflagged_amount_difference=ad['unflagged_fraud_amount']-actual['unflagged_fraud_amount'],
            strict_hypothesis_confirmed=bool(ad['fp']<actual['fp'] and ad['tp']>=actual['tp'] and ad['unflagged_fraud_amount']<=actual['unflagged_fraud_amount'])))
        preds[name]=action
    save('additional_fair_comparators.json',dict(note='Fairness audit after first test result; adaptive policy unchanged; comparator thresholds computed only on policy_validation.',comparators=results))
    np.savez_compressed(OUT/'additional_fair_predictions.npz',**preds)
    print('Additional comparators saved; adaptive policy unchanged.')
if __name__=='__main__':main()
