"""Stable, balance-free PaySim models and validation-selected decision layer.

Protocol is saved before tuning. Test is evaluated only after all selections.
This is a reused temporal holdout, not an independent new simulation.
No delayed action is credited as a successfully prevented attack.
"""
from __future__ import annotations
import gc, hashlib, itertools, json, pathlib, sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'models'/'paysim_v2'
CACHE=ROOT/'tmp'/'decision_layer_v2_clean'
COLS=['snapshot_id','source_row_number','ingestion_run_id','step','transaction_type','amount','name_orig','old_balance_orig','new_balance_orig','name_dest','old_balance_dest','new_balance_dest','is_fraud','is_flagged_fraud']

def default(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v))
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf8')
def say(*args):print(*args,flush=True)

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'features.npy').exists() and (CACHE/'metadata.npz').exists() and (CACHE/'columns.json').exists():
        say('Loading previously prepared causal feature cache')
        return np.load(CACHE/'features.npy',mmap_mode='r'),dict(np.load(CACHE/'metadata.npz')),json.loads((CACHE/'columns.json').read_text())
    p=ROOT/'tmp'/'paysim_audit'/'paysim_data.sql'
    if not p.exists():raise FileNotFoundError('Extract paysim_transactions from the original dump first.')
    with p.open(encoding='utf8') as f:
        for i,line in enumerate(f):
            if line.startswith('COPY ml.paysim_transactions '):skip=i+1;break
        else:raise ValueError('COPY header not found')
    df=pd.read_csv(p,sep='\t',skiprows=skip,nrows=6362620,names=COLS,header=None,
        usecols=['source_row_number','step','transaction_type','amount','name_dest','is_fraud'],
        dtype={'source_row_number':'int64','step':'int32','transaction_type':'category','amount':'float64','name_dest':'str','is_fraud':'str'})
    assert len(df)==6362620 and df.source_row_number.nunique()==len(df)
    df=df.sort_values(['step','source_row_number'],kind='stable').reset_index(drop=True)
    y=(df.is_fraud=='t').to_numpy();assert y.sum()==8213
    df['dest_code']=pd.factorize(df.name_dest,sort=False)[0].astype('int32')
    df=df.drop(columns=['name_dest','is_fraud'])
    df['square_amount']=df.amount**2
    df['transfer']=(df.transaction_type=='TRANSFER').astype('int8')
    df['cashout']=(df.transaction_type=='CASH_OUT').astype('int8')
    say('Computing recipient history using strictly earlier hours')
    g=df.groupby(['dest_code','step'],sort=False,observed=True).agg(
        n=('amount','size'),total=('amount','sum'),square=('square_amount','sum'),
        high=('amount','max'),transfer=('transfer','sum'),cashout=('cashout','sum')).reset_index()
    g=g.sort_values('step',kind='stable').reset_index(drop=True)
    by=g.groupby('dest_code',sort=False,observed=True)
    for src,name in [('n','count'),('total','amount'),('square','square'),('transfer','transfers'),('cashout','cashouts')]:
        g['inclusive_'+name]=by[src].cumsum()
        g['prior_'+name]=by['inclusive_'+name].shift(1).fillna(0)
    g['prior_max']=by['high'].cummax().groupby(g.dest_code,sort=False).shift(1).fillna(0)
    g['last_step']=by['step'].shift(1)
    g['age']=g.step-by['step'].transform('min')
    left=g[['dest_code','step']].copy();left['query_step']=left.step-24
    right=g[['dest_code','step','inclusive_count','inclusive_amount']]
    old=pd.merge_asof(left,right,left_on='query_step',right_on='step',by='dest_code',
        direction='backward',allow_exact_matches=False)
    g['recent_count']=g.prior_count-old.inclusive_count.fillna(0).to_numpy()
    g['recent_amount']=(g.prior_amount-old.inclusive_amount.fillna(0).to_numpy()).clip(lower=0)
    assert (g.recent_count>=0).all() and (g.recent_count<=g.prior_count).all()
    del by,left,right,old;gc.collect()
    keep=['dest_code','step','prior_count','prior_amount','prior_square','prior_max',
        'prior_transfers','prior_cashouts','last_step','age','recent_count','recent_amount']
    df['_row']=np.arange(len(df))
    h=df[['dest_code','step','_row']].merge(g[keep],on=['dest_code','step'],how='left',sort=False,validate='many_to_one')
    assert np.array_equal(h._row.to_numpy(),np.arange(len(df)))
    del g;gc.collect()
    x=pd.DataFrame(index=df.index)
    x['log_amount']=np.log1p(df.amount)
    x['hour_sin']=np.sin(2*np.pi*(df.step%24)/24)
    x['hour_cos']=np.cos(2*np.pi*(df.step%24)/24)
    x['is_round_100']=np.isclose(np.remainder(df.amount,100),0,atol=.001).astype('int8')
    for typ in ['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']:
        x['type_'+typ]=(df.transaction_type==typ).astype('int8')
    base_columns=list(x.columns)
    count=h.prior_count.to_numpy();mean=np.divide(h.prior_amount,count,out=np.zeros(len(df)),where=count>0)
    square_mean=np.divide(h.prior_square,count,out=np.zeros(len(df)),where=count>0)
    std=np.where(count>=2,np.sqrt(np.maximum(square_mean-mean**2,0)),0)
    x['dest_seen_before']=(count>0).astype('int8')
    x['dest_log_count']=np.log1p(count)
    x['dest_log_mean']=np.log1p(mean)
    x['dest_log_std']=np.log1p(std)
    x['dest_log_max']=np.log1p(h.prior_max)
    x['log_amount_over_mean']=np.log1p(df.amount)-np.log1p(mean)
    x['log_amount_over_max']=np.log1p(df.amount)-np.log1p(h.prior_max)
    x['dest_transfer_fraction']=np.divide(h.prior_transfers,count,out=np.zeros(len(df)),where=count>0)
    x['dest_cashout_fraction']=np.divide(h.prior_cashouts,count,out=np.zeros(len(df)),where=count>0)
    x['dest_log_age']=np.log1p(h.age)
    x['dest_log_gap']=np.log1p((df.step-h.last_step).fillna(744))
    x['dest_log_recent_count']=np.log1p(h.recent_count)
    x['dest_log_recent_amount']=np.log1p(h.recent_amount)
    meta=dict(row_id=df.source_row_number.to_numpy(),step=df.step.to_numpy(),y=y,
        amount=df.amount.to_numpy(),history_count=count.astype('int32'),transfer=df.transfer.to_numpy())
    columns=list(x.columns)
    values=x.to_numpy(dtype='float32')
    assert np.isfinite(values).all()
    np.save(CACHE/'features.npy',values)
    np.savez_compressed(CACHE/'metadata.npz',**meta)
    (CACHE/'columns.json').write_text(json.dumps(columns),encoding='utf8')
    save('feature_manifest.json',dict(columns=columns,base_columns=base_columns,rows=len(df),
        balance_fields_used=False,target_in_history=False,raw_identity_in_model=False,
        history_lag='strictly less than current step',recent_window='[step-24, step)',
        full_timestamp_order_assumed=False))
    say('Prepared',values.shape)
    return values,meta,columns

def metrics(y,amount,restricted):
    y=np.asarray(y,dtype=bool);r=np.asarray(restricted,dtype=bool);a=np.asarray(amount)
    tp=int((y&r).sum());fp=int((~y&r).sum());fn=int((y&~r).sum());tn=int((~y&~r).sum())
    return dict(n=len(y),fraud=int(y.sum()),legitimate=int((~y).sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        count_recall=float(tp/y.sum()),precision=float(tp/max(tp+fp,1)),
        false_intervention_rate=float(fp/(~y).sum()) if (~y).sum() else None,interventions=int(r.sum()),
        fraud_amount=float(a[y].sum()),unflagged_fraud_amount=float(a[y&~r].sum()),
        amount_recall=float(a[y&r].sum()/a[y].sum()))

def threshold_at_targets(priority,y,amount,target_count=.9,target_amount=.95):
    order=np.argsort(-priority,kind='stable');s=priority[order];yy=y[order];aa=amount[order]
    tp=np.cumsum(yy);captured=np.cumsum(aa*yy)
    ok=(tp>=np.ceil(target_count*y.sum()))&(captured>=target_amount*amount[y].sum())
    first=int(np.flatnonzero(ok)[0]);threshold=float(s[first])
    # Include all ties. This is a frozen numeric threshold, not a capacity claim.
    restricted=priority>=threshold
    return threshold,metrics(y,amount,restricted)

def calibrate(model,x,mask,y,name):
    raw=model.predict(x[mask],num_iteration=model.best_iteration)
    z=logit(np.clip(raw,1e-8,1-1e-8)).reshape(-1,1)
    lr=LogisticRegression(C=100,max_iter=1000,random_state=20261101)
    lr.fit(z,y[mask])
    result=dict(slope=float(lr.coef_[0,0]),intercept=float(lr.intercept_[0]),
        calibration_rows=int(mask.sum()),calibration_fraud=int(y[mask].sum()))
    assert result['slope']>0,'Calibration reversed ranking; investigate model.'
    save(name+'_calibration.json',result)
    return result

def calibrated_logit(model,x,mask,calibration):
    raw=model.predict(x[mask],num_iteration=model.best_iteration)
    return calibration['intercept']+calibration['slope']*logit(np.clip(raw,1e-8,1-1e-8))

def train_family(x,y,tr,mv,columns,prefix,protocol):
    grid=protocol['model_grid'];common=dict(protocol['model_common']);early=common.pop('early_stopping_rounds')
    trials=[];best=None;best_ap=-1
    for i,params in enumerate(grid):
        say('Training',prefix,i,params)
        model=lgb.LGBMClassifier(**common,**params,objective='binary',verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(x[tr],y[tr],eval_set=[(x[mv],y[mv])],eval_metric='average_precision',
            callbacks=[lgb.early_stopping(early,first_metric_only=True,verbose=False)])
        booster=model.booster_;s=booster.predict(x[mv],num_iteration=booster.best_iteration)
        ap=average_precision_score(y[mv],s)
        trial=dict(index=i,params=params,best_iteration=booster.best_iteration,validation_ap=ap,
            validation_auc=roc_auc_score(y[mv],s),finite_scores=bool(np.isfinite(s).all()))
        trials.append(trial);say(prefix,i,'AP',round(ap,5),'trees',booster.best_iteration)
        if ap>best_ap:best=booster;best_ap=ap;best_index=i
        del model;gc.collect()
    save(prefix+'_trials.json',trials)
    best.save_model(str(OUT/(prefix+'_model.txt')))
    pd.DataFrame({'feature':columns,'gain':best.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(OUT/(prefix+'_importance.csv'),index=False,encoding='utf-8-sig')
    return best,dict(selected_index=best_index,selected=trials[best_index])

def priority_score(coarse,expert,amount,history,transfer,params,reference_amount):
    w=params['expert_weight']
    return ((1-w)*coarse+w*expert+params['amount_alpha']*np.log1p(amount/reference_amount)
        +params['history_beta']*np.log1p(history)+params['transfer_gamma']*transfer)

def apply_layer(coarse_logit,expert_logit,amount,history_count,is_transfer,policy,mandatory_hold=None):
    score=priority_score(coarse_logit,expert_logit,amount,history_count,is_transfer,policy['parameters'],policy['reference_amount'])
    result=score>=policy['threshold']
    if mandatory_hold is not None:result=result|np.asarray(mandatory_hold,dtype=bool)
    return result

def main():
    t=time.time();OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    protocol_hash=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    x,meta,columns=prepare();y=meta['y'];steps=meta['step'];amount=meta['amount']
    masks={name:(steps>=bounds[0])&(steps<=bounds[1]) for name,bounds in protocol['splits'].items()}
    assert np.all(sum(mask.astype('int8') for mask in masks.values())==1)
    save('split_counts.json',{name:dict(n=int(mask.sum()),fraud=int(y[mask].sum()),min_step=int(steps[mask].min()),max_step=int(steps[mask].max())) for name,mask in masks.items()})
    base_cols=[i for i,c in enumerate(columns) if not c.startswith('dest_') and c not in ['log_amount_over_mean','log_amount_over_max']]
    coarse_x=x[:,base_cols];coarse_names=[columns[i] for i in base_cols]
    coarse,coarse_choice=train_family(coarse_x,y,masks['train'],masks['model_validation'],coarse_names,'transaction',protocol)
    expert,expert_choice=train_family(x,y,masks['train'],masks['model_validation'],columns,'history',protocol)
    coarse_cal=calibrate(coarse,coarse_x,masks['calibration'],y,'transaction')
    expert_cal=calibrate(expert,x,masks['calibration'],y,'history')
    pv=masks['policy_validation']
    c=calibrated_logit(coarse,coarse_x,pv,coarse_cal);e=calibrated_logit(expert,x,pv,expert_cal)
    yy=y[pv];aa=amount[pv];hh=meta['history_count'][pv];tt=meta['transfer'][pv]
    reference=float(np.median(amount[masks['train']]))
    targets=protocol['policy_validation_targets']
    choose=lambda score:threshold_at_targets(score,yy,aa,targets['fraud_count_recall'],targets['fraud_amount_recall'])
    ct,cm=choose(c);et,em=choose(e)
    policies=dict(transaction_threshold=dict(threshold=ct,validation=cm),history_threshold=dict(threshold=et,validation=em))
    grid=protocol['policy_search'];trials=[]
    for values in itertools.product(*grid.values()):
        params=dict(zip(grid,values));p=priority_score(c,e,aa,hh,tt,params,reference)
        threshold,measure=choose(p)
        trials.append(dict(parameters=params,threshold=threshold,validation=measure))
    # All choices use policy_validation; no test labels or test distributions.
    trials.sort(key=lambda row:(row['validation']['fp'],row['validation']['unflagged_fraud_amount'],
        row['validation']['fn'],sum(abs(v) for k,v in row['parameters'].items() if k!='expert_weight')))
    winner=trials[0];policy=dict(parameters=winner['parameters'],threshold=winner['threshold'],reference_amount=reference)
    policies['adaptive']=dict(**policy,validation=winner['validation'])
    # Simple transparent rule: restrict riskable operation types above amount t.
    riskable=meta['transfer'][pv]>0
    type_cashout=coarse_x[pv,coarse_names.index('type_CASH_OUT')]>0
    riskable=riskable|type_cashout
    rt,rm=choose(np.where(riskable,np.log1p(aa),-1.))
    policies['simple_rule']=dict(threshold=rt,validation=rm)
    save('policy_trials.json',trials);save('selected_policies.json',policies)
    save('selection_lock.json',dict(protocol_sha256=protocol_hash,model_choices=dict(transaction=coarse_choice,history=expert_choice),
        policy=policy,test_evaluated_yet=False,selection_inputs=['train','model_validation','calibration','policy_validation']))
    say('Selections frozen before test evaluation.',policy)
    # Locked comparison on a reused temporal holdout. No further tuning below.
    te=masks['test'];yt=y[te];at=amount[te]
    ctest=calibrated_logit(coarse,coarse_x,te,coarse_cal);etest=calibrated_logit(expert,x,te,expert_cal)
    restricted={
        'transaction_threshold':ctest>=ct,
        'history_threshold':etest>=et,
        'adaptive':apply_layer(ctest,etest,at,meta['history_count'][te],meta['transfer'][te],policy),
        'simple_rule':np.where((meta['transfer'][te]>0)|(coarse_x[te,coarse_names.index('type_CASH_OUT')]>0),np.log1p(at),-1.)>=rt,
    }
    results={name:metrics(yt,at,r) for name,r in restricted.items()}
    ranking=dict(transaction=dict(ap=average_precision_score(yt,expit(ctest)),auc=roc_auc_score(yt,ctest),brier=brier_score_loss(yt,expit(ctest))),
        history=dict(ap=average_precision_score(yt,expit(etest)),auc=roc_auc_score(yt,etest),brier=brier_score_loss(yt,expit(etest))))
    comparisons={}
    for ref in ['transaction_threshold','history_threshold','simple_rule']:
        a=results['adaptive'];b=results[ref]
        comparisons[ref]=dict(fp_relative_change=float(a['fp']/max(1,b['fp'])-1),
            tp_difference=a['tp']-b['tp'],unflagged_fraud_amount_difference=a['unflagged_fraud_amount']-b['unflagged_fraud_amount'],
            strict_hypothesis_confirmed=bool(a['fp']<b['fp'] and a['tp']>=b['tp'] and a['unflagged_fraud_amount']<=b['unflagged_fraud_amount']))
    save('test_results.json',dict(policies=results,ranking=ranking,comparisons=comparisons))
    np.savez_compressed(OUT/'test_predictions.npz',row_id=meta['row_id'][te],step=steps[te],y=yt,amount=at,
        transaction_score=ctest,history_score=etest,**restricted)
    # Day/block summaries give visibility into drift without reselecting policies.
    records=[]
    for day in np.unique(steps[te]//24):
        m=steps[te]//24==day
        for name,r in restricted.items():
            if yt[m].sum():records.append(dict(day=int(day),policy=name,**metrics(yt[m],at[m],r[m])))
    pd.DataFrame(records).to_csv(OUT/'test_by_simulation_day.csv',index=False,encoding='utf-8-sig')
    # Interface checks for mandatory restriction: no learned policy can bypass it.
    assert apply_layer(np.array([-100.]),np.array([-100.]),np.array([1.]),np.array([0]),np.array([0]),policy,np.array([True]))[0]
    save('run_manifest.json',dict(date='2026-11-01',seconds=time.time()-t,protocol_sha256=protocol_hash,
        source_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__,sklearn=sklearn.__version__,
        test_reused=True,balances_used=False,all_interventions_counted_as_friction=True,
        mandatory_hold_override_test_passed=True,hello_world_deleted=not (ROOT/'hello_world.py').exists()))
    say('Completed.',json.dumps(comparisons,default=default))

if __name__=='__main__':main()
