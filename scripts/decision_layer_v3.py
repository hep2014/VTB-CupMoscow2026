"""Train type-specific verifier experts over the frozen v2 history baseline.

Policy selection preserves baseline count and amount in two validation windows.
All intervention outcomes count as friction; no fictional confirmation success.
The final time interval is explicitly a reused development holdout.
"""
from __future__ import annotations
import gc,hashlib,itertools,json,pathlib,sys,time
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from decision_layer_v2 import ROOT,CACHE,metrics

OUT=ROOT/'research'/'paysim'/'v3'
V2=ROOT/'models'/'paysim_v2'
def default(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v))
def save(n,v):
    (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf8')
def read(folder,n):return json.loads((folder/n).read_text(encoding='utf8'))
def say(*args):print(*args,flush=True)
def raw_logit(model,x):return logit(np.clip(model.predict(x),1e-8,1-1e-8))
def cal_logit(model,x,cal):return cal['intercept']+cal['slope']*raw_logit(model,x)

def threshold_for_window(priority,guard,y,amount,baseline):
    """Largest score cutoff retaining baseline TP/amount in this window."""
    target_n=int((y&baseline).sum());target_amount=float(amount[y&baseline].sum())
    guard_n=int((y&guard).sum());guard_amount=float(amount[y&guard].sum())
    need_n=max(0,target_n-guard_n);need_amount=max(0.,target_amount-guard_amount)
    if not need_n and need_amount<=1e-6:return 1e6
    available=np.flatnonzero(~guard)
    order=available[np.argsort(-priority[available],kind='stable')]
    cum_n=np.cumsum(y[order]);cum_amount=np.cumsum(amount[order]*y[order])
    ok=(cum_n>=need_n)&(cum_amount+1e-6>=need_amount)
    if not ok.any():raise ValueError('Cannot retain baseline outcomes even when all candidates restricted')
    return float(priority[order[np.flatnonzero(ok)[0]]])

def expert_priority(base,expert,amount,params,reference):
    w=params['expert_weight']
    return (1-w)*base+w*expert+params['amount_alpha']*np.log1p(amount/reference)

def select_policy(base,expert,amount,y,steps,baseline,baseline_threshold,reference,protocol):
    records=[]
    for w,alpha,margin in itertools.product(*protocol['policy_grid'].values()):
        params=dict(expert_weight=w,amount_alpha=alpha,guard_logit_margin=margin)
        priority=expert_priority(base,expert,amount,params,reference)
        guard=np.zeros(len(y),bool) if margin is None else base>=baseline_threshold+margin
        thresholds=[]
        for first,last in protocol['policy_subwindows']:
            m=(steps>=first)&(steps<=last)
            thresholds.append(threshold_for_window(priority[m],guard[m],y[m],amount[m],baseline[m]))
        threshold=min(thresholds)
        action=guard|(priority>=threshold)
        sub=[]
        for first,last in protocol['policy_subwindows']:
            m=(steps>=first)&(steps<=last)
            a=metrics(y[m],amount[m],action[m]);b=metrics(y[m],amount[m],baseline[m])
            assert a['tp']>=b['tp'] and a['unflagged_fraud_amount']<=b['unflagged_fraud_amount']+1e-5
            sub.append(dict(period=[first,last],candidate=a,baseline=b))
        records.append(dict(parameters=params,threshold=threshold,reference_amount=reference,
            validation=metrics(y,amount,action),subwindows=sub))
    records.sort(key=lambda r:(r['validation']['fp'],r['validation']['unflagged_fraud_amount'],
        r['validation']['fn'],r['parameters']['expert_weight'],r['parameters']['amount_alpha']))
    return records[0],records

def apply_policy(base,expert,amount,policy,baseline_threshold):
    p=policy['parameters'];margin=p['guard_logit_margin']
    guard=np.zeros(len(base),bool) if margin is None else base>=baseline_threshold+margin
    return guard|(expert_priority(base,expert,amount,p,policy['reference_amount'])>=policy['threshold'])

def main():
    t=time.time();OUT.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT,'protocol.json');protocol_hash=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    x=np.load(CACHE/'features.npy',mmap_mode='r');meta=dict(np.load(CACHE/'metadata.npz'))
    columns=json.loads((CACHE/'columns.json').read_text())
    y=meta['y'];steps=meta['step'];amount=meta['amount']
    masks={name:(steps>=lo)&(steps<=hi) for name,(lo,hi) in protocol['splits'].items()}
    baseline_model=lgb.Booster(model_file=str(V2/'history_model.txt'))
    assert hashlib.sha256((V2/'history_model.txt').read_bytes()).hexdigest()==protocol['baseline_model_sha256']
    baseline_cal=read(V2,'history_calibration.json')
    baseline_threshold=read(V2,'selected_policies.json')['history_threshold']['threshold']
    pv=masks['policy_validation'];te=masks['test']
    base_pv=cal_logit(baseline_model,x[pv],baseline_cal)
    base_action_pv=base_pv>=baseline_threshold
    experts={};policies={};calibrations={};choices={}
    for typ in protocol['expert_types']:
        typ_mask=x[:,columns.index('type_'+typ)]>.5
        tr=masks['train']&typ_mask;mv=masks['model_validation']&typ_mask;ca=masks['calibration']&typ_mask
        yp=y[tr];trials=[];best_ap=-1;best=None
        for i,params in enumerate(protocol['expert_grid']):
            say('Training verifier',typ,i,'rows',int(tr.sum()))
            model=lgb.LGBMClassifier(**protocol['model_common'],**params,objective='binary',metric='average_precision',
                verbosity=-1,deterministic=True,force_col_wise=True)
            model.fit(x[tr],yp,eval_set=[(x[mv],y[mv])],eval_metric='average_precision',
                callbacks=[lgb.early_stopping(protocol['early_stopping_rounds'],first_metric_only=True,verbose=False)])
            booster=model.booster_;score=booster.predict(x[mv]);ap=average_precision_score(y[mv],score)
            trial=dict(index=i,params=params,validation_ap=ap,validation_auc=roc_auc_score(y[mv],score),
                best_iteration=booster.best_iteration,train_n=int(tr.sum()),validation_n=int(mv.sum()))
            trials.append(trial);say(typ,i,'AP',round(ap,5),'trees',booster.best_iteration)
            if ap>best_ap:best_ap=ap;best=booster;choice=trial
            del model;gc.collect()
        save(typ+'_trials.json',trials);best.save_model(str(OUT/(typ+'_expert.txt')))
        pd.DataFrame({'feature':columns,'gain':best.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(OUT/(typ+'_importance.csv'),index=False,encoding='utf-8-sig')
        lr=LogisticRegression(C=10,max_iter=1000,random_state=20261101)
        lr.fit(raw_logit(best,x[ca]).reshape(-1,1),y[ca])
        cal=dict(slope=float(lr.coef_[0,0]),intercept=float(lr.intercept_[0]),n=int(ca.sum()),fraud=int(y[ca].sum()))
        assert cal['slope']>0
        save(typ+'_calibration.json',cal)
        pm=typ_mask[pv]
        expert=cal_logit(best,x[pv][pm],cal)
        reference=float(np.median(amount[tr]))
        policy,trials_policy=select_policy(base_pv[pm],expert,amount[pv][pm],y[pv][pm],steps[pv][pm],
            base_action_pv[pm],baseline_threshold,reference,protocol)
        policies[typ]=policy;calibrations[typ]=cal;choices[typ]=choice;experts[typ]=best
        save(typ+'_policy_trials.json',trials_policy)
        say('Selected',typ,policy['parameters'],'FP validation',policy['validation']['fp'])
    save('selected_policies.json',dict(baseline_threshold=baseline_threshold,types=policies))
    save('feature_manifest.json',read(V2,'feature_manifest.json'))
    save('selection_lock.json',dict(protocol_sha256=protocol_hash,test_evaluated_yet=False,
        expert_choices=choices,policies=policies,baseline_threshold=baseline_threshold))
    say('Verifier selections frozen; evaluating reused test.')
    base_test=cal_logit(baseline_model,x[te],baseline_cal);base_action=base_test>=baseline_threshold
    action=base_action.copy();expert_scores=np.full(int(te.sum()),np.nan)
    by_type={}
    for typ in protocol['expert_types']:
        m=x[te,columns.index('type_'+typ)]>.5
        e=cal_logit(experts[typ],x[te][m],calibrations[typ]);expert_scores[m]=e
        action[m]=apply_policy(base_test[m],e,amount[te][m],policies[typ],baseline_threshold)
        by_type[typ]=dict(baseline=metrics(y[te][m],amount[te][m],base_action[m]),
            verifier=metrics(y[te][m],amount[te][m],action[m]),expert_ap=average_precision_score(y[te][m],expit(e)))
    bm=metrics(y[te],amount[te],base_action);am=metrics(y[te],amount[te],action)
    comparison=dict(fp_reduction=1-am['fp']/bm['fp'],tp_difference=am['tp']-bm['tp'],
        unflagged_amount_difference=am['unflagged_fraud_amount']-bm['unflagged_fraud_amount'],
        strict_hypothesis_confirmed=bool(am['fp']<bm['fp'] and am['tp']>=bm['tp'] and am['unflagged_fraud_amount']<=bm['unflagged_fraud_amount']))
    save('test_results.json',dict(baseline=bm,verifier=am,comparison=comparison,by_type=by_type))
    np.savez_compressed(OUT/'test_predictions.npz',row_id=meta['row_id'][te],step=steps[te],amount=amount[te],y=y[te],
        baseline_score=base_test,expert_score=expert_scores,baseline=base_action,verifier=action)
    save('run_manifest.json',dict(date='2026-11-01',seconds=time.time()-t,protocol_sha256=protocol_hash,
        source_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        baseline_frozen=True,selection_before_test=True,test_reused=True,balances_used=False,
        original_deleted=not (ROOT/'hello_world.py').exists()))
    say('V3 complete',json.dumps(comparison,default=default))

if __name__=='__main__':main()
