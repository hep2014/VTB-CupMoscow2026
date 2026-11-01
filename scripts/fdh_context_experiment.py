"""Independent FDH-method experiment with matured labels and customer context.

Source is the supplied independently generated FDH dataset, not authors' data.
Generator/oracle fields never enter X. Days excluded between splits ensure the
assumed seven-full-day label delay is respected when selecting each stage.
"""
from __future__ import annotations
import gc,hashlib,json,pathlib,sys,time
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from decision_layer_v2 import ROOT,metrics,threshold_at_targets
from decision_layer_v3 import select_policy,apply_policy

OUT=ROOT/'models'/'context_layer'
TMP=ROOT/'tmp'/'fdh_context'
COLS=['simulation_run_id','transaction_id','tx_datetime','customer_id','terminal_id','tx_amount','tx_time_seconds','tx_time_days','tx_fraud','tx_fraud_scenario','base_tx_amount','scenario_mask','scenario3_hits','ingestion_run_id']
def default(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v))
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf8')
def say(*a):print(*a,flush=True)
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def zscore(model,x,cal):return cal['intercept']+cal['slope']*logit(np.clip(model.predict(x),1e-8,1-1e-8))

def prepare():
    TMP.mkdir(parents=True,exist_ok=True)
    if (TMP/'features.npy').exists() and (TMP/'metadata.npz').exists():
        return np.load(TMP/'features.npy',mmap_mode='r'),dict(np.load(TMP/'metadata.npz')),read('feature_manifest.json')
    p=TMP/'fdh_data.sql'
    with p.open(encoding='utf8') as f:
        for i,line in enumerate(f):
            if line.startswith('COPY ml.fdh_transactions '):skip=i+1;break
        else:raise ValueError('Missing FDH COPY')
    use=['transaction_id','customer_id','terminal_id','tx_amount','tx_time_seconds','tx_time_days','tx_fraud']
    df=pd.read_csv(p,sep='\t',skiprows=skip,nrows=1781887,names=COLS,header=None,usecols=use,
        dtype={'transaction_id':'int64','customer_id':'int32','terminal_id':'int32','tx_amount':'float64',
        'tx_time_seconds':'int64','tx_time_days':'int32','tx_fraud':'str'})
    df=df.sort_values(['tx_time_seconds','transaction_id'],kind='stable').reset_index(drop=True)
    y=(df.tx_fraud=='t').to_numpy();assert len(df)==1781887 and y.sum()==14753
    day=df.tx_time_days.to_numpy();amount=df.tx_amount.to_numpy();T=int(day.max())+1
    x=pd.DataFrame(index=df.index)
    x['amount']=amount;x['log_amount']=np.log1p(amount)
    hour=np.remainder(df.tx_time_seconds.to_numpy(),86400)/3600
    x['hour_sin']=np.sin(2*np.pi*hour/24);x['hour_cos']=np.cos(2*np.pi*hour/24)
    x['weekend']=((day+6)%7>=5).astype('int8')
    def cumulative(entity,weight=None):
        n=int(entity.max())+1
        daily=np.bincount(entity*T+day,weights=weight,minlength=n*T).reshape(n,T)
        return np.pad(np.cumsum(daily,axis=1),((0,0),(1,0)))
    def window(prefix,entity,span,lag):
        end=np.maximum(day-lag,0);start=np.maximum(end-span,0)
        return prefix[entity,end]-prefix[entity,start]
    terminal=df.terminal_id.to_numpy();customer=df.customer_id.to_numpy()
    tc=cumulative(terminal);tf=cumulative(terminal,y.astype(float))
    for w in [1,7,30]:
        seen=window(tc,terminal,w,1)
        known=window(tc,terminal,w,7);frauds=window(tf,terminal,w,7)
        x[f'terminal_log_count_{w}']=np.log1p(seen)
        x[f'terminal_known_risk_{w}']=(frauds+1)/(known+200)
        x[f'terminal_known_support_{w}']=np.log1p(known)
    base_columns=list(x.columns)
    cc=cumulative(customer);cs=cumulative(customer,amount);cq=cumulative(customer,amount**2)
    for w in [1,7,30]:
        count=window(cc,customer,w,1);total=window(cs,customer,w,1)
        mean=np.divide(total,count,out=np.zeros(len(df)),where=count>0)
        x[f'customer_log_count_{w}']=np.log1p(count);x[f'customer_log_mean_{w}']=np.log1p(mean)
        x[f'log_amount_vs_customer_{w}']=np.log1p(amount)-np.log1p(mean)
        if w==30:
            second=np.divide(window(cq,customer,w,1),count,out=np.zeros(len(df)),where=count>0)
            std=np.where(count>=2,np.sqrt(np.maximum(second-mean**2,0)),0.)
            x['customer_log_std_30']=np.log1p(std)
            x['customer_amount_z_30']=np.clip((amount-mean)/(std+1),-20,20)
    # New/known customer-terminal relation from events available before day D-1.
    pair=customer.astype('int64')*(int(terminal.max())+1)+terminal
    relation=pd.DataFrame({'pair':pair,'day':day})
    g=relation.groupby(['pair','day'],sort=False).size().reset_index(name='n').sort_values('day',kind='stable')
    g['inclusive_n']=g.groupby('pair',sort=False).n.cumsum()
    query=relation.copy();query['query_day']=query.day-1
    prior=pd.merge_asof(query,g[['pair','day','inclusive_n']],left_on='query_day',right_on='day',by='pair',
        direction='backward',allow_exact_matches=False)
    count=prior.inclusive_n.fillna(0).to_numpy()
    x['customer_terminal_log_count']=np.log1p(count);x['customer_terminal_new']=(count==0).astype('int8')
    schema=dict(columns=list(x.columns),base_columns=base_columns,rows=len(df),fraud=int(y.sum()),
        oracle_columns_excluded=True,raw_identifiers_excluded=True,history_full_day_delay=1,label_full_day_delay=7,
        rule='For day D: transaction history days < D-1; confirmed labels days < D-7.')
    values=x.to_numpy(dtype='float32');assert np.isfinite(values).all()
    meta=dict(row_id=df.transaction_id.to_numpy(),day=day,amount=amount,y=y,customer=customer,terminal=terminal)
    np.save(TMP/'features.npy',values);np.savez_compressed(TMP/'metadata.npz',**meta);save('feature_manifest.json',schema)
    save('data_inventory.json',dict(rows=len(df),fraud=int(y.sum()),customers=df.customer_id.nunique(),terminals=df.terminal_id.nunique(),
        min_day=int(day.min()),max_day=int(day.max()),min_events_customer=int(df.customer_id.value_counts().min()),
        median_events_customer=float(df.customer_id.value_counts().median())))
    say('FDH features prepared',values.shape)
    return values,meta,schema

def fit_family(x,y,masks,name,columns,protocol):
    best=None;best_ap=-1;trials=[]
    for i,params in enumerate(protocol['model_grid']):
        say('Training FDH',name,i)
        model=lgb.LGBMClassifier(**protocol['common'],**params,objective='binary',metric='average_precision',
            deterministic=True,force_col_wise=True,verbosity=-1)
        tr=masks['train'];mv=masks['model_validation']
        model.fit(x[tr],y[tr],eval_set=[(x[mv],y[mv])],eval_metric='average_precision',
            callbacks=[lgb.early_stopping(protocol['early_stopping'],first_metric_only=True,verbose=False)])
        s=model.booster_.predict(x[mv]);ap=average_precision_score(y[mv],s)
        record=dict(index=i,params=params,ap=ap,auc=roc_auc_score(y[mv],s),best_iteration=model.booster_.best_iteration)
        trials.append(record);say(name,'AP',round(ap,5),'trees',record['best_iteration'])
        if ap>best_ap:best=model.booster_;best_ap=ap;choice=record
        del model;gc.collect()
    best.save_model(str(OUT/(name+'_model.txt')));save(name+'_trials.json',trials)
    pd.DataFrame({'feature':columns,'gain':best.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(OUT/(name+'_importance.csv'),index=False,encoding='utf-8-sig')
    ca=masks['calibration'];lr=LogisticRegression(C=50,max_iter=1000,random_state=20261101)
    lr.fit(logit(np.clip(best.predict(x[ca]),1e-8,1-1e-8)).reshape(-1,1),y[ca])
    cal=dict(slope=float(lr.coef_[0,0]),intercept=float(lr.intercept_[0]));assert cal['slope']>0
    save(name+'_calibration.json',cal)
    return best,cal,choice

def main():
    t=time.time();protocol=read('protocol.json');ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    x,meta,schema=prepare();day=meta['day'];y=meta['y'];amount=meta['amount']
    masks={name:(day>=lo)&(day<=hi) for name,(lo,hi) in protocol['splits'].items()}
    assert np.max(sum(m.astype(int) for m in masks.values()))==1
    assert protocol['splits']['train'][1]+8<=protocol['splits']['model_validation'][0]
    for a,b in [('model_validation','calibration'),('calibration','policy_validation'),('policy_validation','test')]:
        assert protocol['splits'][a][1]+8<=protocol['splits'][b][0]
    save('split_counts.json',{k:dict(n=int(m.sum()),fraud=int(y[m].sum()),min_day=int(day[m].min()),max_day=int(day[m].max())) for k,m in masks.items()})
    indices=[schema['columns'].index(c) for c in schema['base_columns']];base_x=x[:,indices]
    baseline,bc,bchoice=fit_family(base_x,y,masks,'baseline',schema['base_columns'],protocol)
    context,ec,echoice=fit_family(x,y,masks,'context',schema['columns'],protocol)
    pv=masks['policy_validation'];b=zscore(baseline,base_x[pv],bc);e=zscore(context,x[pv],ec)
    threshold,bm=threshold_at_targets(b,y[pv],amount[pv],**{
        'target_count':protocol['baseline_validation_targets']['count_recall'],
        'target_amount':protocol['baseline_validation_targets']['amount_recall']})
    base_action=b>=threshold;reference=float(np.median(amount[masks['train']]))
    policy,trials=select_policy(b,e,amount[pv],y[pv],day[pv],base_action,threshold,reference,protocol)
    # Rich-model global threshold, subject to the same per-window preservation.
    rich=next(row for row in trials if row['parameters']==dict(expert_weight=1,amount_alpha=0,guard_logit_margin=None))
    save('policy_trials.json',trials);save('selected_policies.json',dict(baseline_threshold=threshold,baseline_validation=bm,
        context_threshold=rich,adaptive=policy))
    save('selection_lock.json',dict(protocol_sha256=ph,test_evaluated_yet=False,model_choices=dict(baseline=bchoice,context=echoice),
        policy=policy,baseline_threshold=threshold))
    say('FDH choices frozen before first scoring of days 157-182',policy['parameters'])
    te=masks['test'];bt=zscore(baseline,base_x[te],bc);et=zscore(context,x[te],ec)
    baseline_action=bt>=threshold;context_action=et>=rich['threshold'];adaptive=apply_policy(bt,et,amount[te],policy,threshold)
    result=dict(baseline=metrics(y[te],amount[te],baseline_action),context_model=metrics(y[te],amount[te],context_action),
        adaptive=metrics(y[te],amount[te],adaptive))
    comparisons={}
    for ref in ['baseline','context_model']:
        a=result['adaptive'];b=result[ref]
        comparisons[ref]=dict(fp_reduction=1-a['fp']/max(b['fp'],1),tp_difference=a['tp']-b['tp'],
            unflagged_amount_difference=a['unflagged_fraud_amount']-b['unflagged_fraud_amount'],
            strict_hypothesis_confirmed=bool(a['fp']<b['fp'] and a['tp']>=b['tp'] and a['unflagged_fraud_amount']<=b['unflagged_fraud_amount']))
    save('test_results.json',dict(policies=result,comparisons=comparisons,
        ranking=dict(baseline_ap=average_precision_score(y[te],expit(bt)),context_ap=average_precision_score(y[te],expit(et)),
            baseline_auc=roc_auc_score(y[te],bt),context_auc=roc_auc_score(y[te],et))))
    np.savez_compressed(OUT/'test_predictions.npz',row_id=meta['row_id'][te],day=day[te],amount=amount[te],y=y[te],
        baseline_score=bt,context_score=et,baseline=baseline_action,context_model=context_action,adaptive=adaptive)
    save('run_manifest.json',dict(date='2026-11-01',protocol_sha256=ph,seconds=time.time()-t,
        source_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),python=sys.version,
        label_embargo_days=7,test_first_scored_after_selection=True,test_data_not_used_for_selection=True,
        independent_simulation_from_paysim=True,not_an_official_authors_dataset=True))
    say('FDH complete',json.dumps(comparisons,default=default))

if __name__=='__main__':main()
