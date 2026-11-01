"""Additional frozen-score comparison under predeclared false-intervention caps."""
import itertools,json
import lightgbm as lgb
import numpy as np
from decision_layer_v3 import expert_priority,apply_policy
from fdh_context_experiment import ROOT,OUT,TMP,read,save,zscore,metrics

def budget_threshold(priority,y,cap):
    max_fp=int(np.floor((~y).sum()*cap))
    # Strictly above the (budget+1)-st highest legitimate score handles ties.
    legitimate=np.sort(priority[~y])[::-1]
    if max_fp>=len(legitimate):threshold=float(np.min(priority)-1)
    else:threshold=float(np.nextafter(legitimate[max_fp],np.inf))
    action=priority>=threshold
    assert int((~y&action).sum())<=max_fp
    return threshold,action

def main():
    study=read('low_friction_protocol.json');protocol=read('protocol.json')
    x=np.load(TMP/'features.npy',mmap_mode='r');meta=dict(np.load(TMP/'metadata.npz'));schema=read('feature_manifest.json')
    indices=[schema['columns'].index(c) for c in schema['base_columns']]
    pv=(meta['day']>=137)&(meta['day']<=149);te=meta['day']>=157
    base=lgb.Booster(model_file=str(OUT/'baseline_model.txt'));context=lgb.Booster(model_file=str(OUT/'context_model.txt'))
    b=zscore(base,x[pv][:,indices],read('baseline_calibration.json'));e=zscore(context,x[pv],read('context_calibration.json'))
    yy=meta['y'][pv];amount=meta['amount'][pv];ref=read('selected_policies.json')['adaptive']['reference_amount']
    candidates={};selected={}
    for cap in study['caps']:
        bt,ba=budget_threshold(b,yy,cap);bm=metrics(yy,amount,ba)
        ct,ca=budget_threshold(e,yy,cap)
        trials=[]
        for w,alpha,margin in itertools.product(*protocol['policy_grid'].values()):
            params=dict(expert_weight=w,amount_alpha=alpha,guard_logit_margin=margin)
            priority=expert_priority(b,e,amount,params,ref)
            guard=np.zeros(len(yy),bool) if margin is None else b>=bt+margin
            guard_fp=int((~yy&guard).sum());budget=int(np.floor((~yy).sum()*cap))
            if guard_fp>budget:continue
            normal_free=(~yy)&(~guard)
            available=budget-guard_fp
            values=np.sort(priority[normal_free])[::-1]
            threshold=float(np.nextafter(values[available],np.inf)) if available<len(values) else float(priority.min()-1)
            action=guard|(priority>=threshold);m=metrics(yy,amount,action)
            assert m['fp']<=budget
            if m['tp']<bm['tp'] or m['unflagged_fraud_amount']>bm['unflagged_fraud_amount']+1e-6:continue
            trials.append(dict(parameters=params,threshold=threshold,reference_amount=ref,validation=m))
        assert trials
        trials.sort(key=lambda r:(-r['validation']['tp'],r['validation']['unflagged_fraud_amount'],r['validation']['fp']))
        selected[str(cap)]=dict(baseline_threshold=bt,context_threshold=ct,adaptive=trials[0],baseline_validation=bm)
    # Persist selections before consulting test scores in these operating modes.
    save('low_friction_selection_lock.json',dict(test_evaluated_yet=False,selections=selected))
    bt=zscore(base,x[te][:,indices],read('baseline_calibration.json'));et=zscore(context,x[te],read('context_calibration.json'))
    y=meta['y'][te];a=meta['amount'][te];results={};predictions={}
    for cap,key in [(v,str(v)) for v in study['caps']]:
        s=selected[key];baseline=bt>=s['baseline_threshold'];rich=et>=s['context_threshold']
        adaptive=apply_policy(bt,et,a,s['adaptive'],s['baseline_threshold'])
        ms=dict(baseline=metrics(y,a,baseline),context_model=metrics(y,a,rich),adaptive=metrics(y,a,adaptive))
        comparisons={}
        for name in ['baseline','context_model']:
            ours=ms['adaptive'];other=ms[name]
            comparisons[name]=dict(fp_reduction=1-ours['fp']/max(1,other['fp']),tp_difference=ours['tp']-other['tp'],
                unflagged_amount_difference=ours['unflagged_fraud_amount']-other['unflagged_fraud_amount'],
                strict_hypothesis_confirmed=bool(ours['fp']<other['fp'] and ours['tp']>=other['tp'] and ours['unflagged_fraud_amount']<=other['unflagged_fraud_amount']))
        results[key]=dict(policies=ms,comparisons=comparisons)
        for name,arr in [('baseline',baseline),('context_model',rich),('adaptive',adaptive)]:
            predictions[f'{key}_{name}']=arr
    save('low_friction_results.json',dict(designated_primary_cap=study['designated_primary_cap'],results=results))
    np.savez_compressed(OUT/'low_friction_predictions.npz',**predictions)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
