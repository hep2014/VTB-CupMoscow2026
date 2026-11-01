"""Independent history sums, delayed-label checks and inference replay."""
import hashlib,json
import numpy as np
import pandas as pd
from fdh_context_experiment import ROOT,OUT,TMP,COLS,read,save,metrics
from context_payment_layer import ContextPaymentLayer

def main():
    x=np.load(TMP/'features.npy',mmap_mode='r');meta=dict(np.load(TMP/'metadata.npz'));schema=read('feature_manifest.json')
    cols=schema['columns'];pred=dict(np.load(OUT/'test_predictions.npz'));low=dict(np.load(OUT/'low_friction_predictions.npz'))
    te=meta['day']>=157;rows=np.flatnonzero(te)
    assert np.array_equal(meta['row_id'][te],pred['row_id']) and len(rows)==252935
    assert np.array_equal(meta['y'][te],pred['y']) and np.array_equal(meta['amount'][te],pred['amount'])
    assert len(np.unique(pred['row_id']))==len(rows)
    assert not any(c in cols for c in ['base_tx_amount','scenario_mask','tx_fraud_scenario','scenario3_hits','customer_id','terminal_id','transaction_id'])
    sample=np.linspace(0,len(rows)-1,256,dtype=int);idx=rows[sample]
    for mode in ['high_recall','low_friction']:
        layer=ContextPaymentLayer(OUT,mode)
        actual=layer.predict_batch(x[idx],meta['amount'][idx])
        expected=pred['adaptive'][sample] if mode=='high_recall' else low['0.005_adaptive'][sample]
        assert np.array_equal(actual['intervene'],expected)
        holds=layer.predict_batch(x[idx],meta['amount'][idx],np.ones(len(idx),bool))
        assert holds['intervene'].all() and (holds['reason']=='MANDATORY_HOLD').all()
        try:layer.predict_batch(x[idx],meta['amount'][idx][::-1])
        except ValueError:pass
        else:raise AssertionError('Misaligned amounts accepted')
    # Read the raw COPY independently; only observable fields and delayed labels.
    raw=pd.read_csv(TMP/'fdh_data.sql',sep='\t',skiprows=24,nrows=1781887,header=None,names=COLS,
        usecols=['transaction_id','customer_id','terminal_id','tx_amount','tx_time_days','tx_fraud'],
        dtype={'transaction_id':'int64','customer_id':'int32','terminal_id':'int32','tx_amount':'float64','tx_time_days':'int32','tx_fraud':'str'})
    raw['y']=raw.tx_fraud=='t'
    assert raw.y.sum()==14753
    chosen=np.linspace(0,len(meta['row_id'])-1,20,dtype=int)
    selected=raw[raw.transaction_id.isin(meta['row_id'][chosen])].set_index('transaction_id')
    checks=0
    for i in chosen:
        r=selected.loc[meta['row_id'][i]];day=int(r.tx_time_days)
        customer=raw[raw.customer_id==r.customer_id];terminal=raw[raw.terminal_id==r.terminal_id]
        expected={}
        for w in [1,7,30]:
            observed=terminal[(terminal.tx_time_days<day-1)&(terminal.tx_time_days>=day-1-w)]
            labelled=terminal[(terminal.tx_time_days<day-7)&(terminal.tx_time_days>=day-7-w)]
            expected[f'terminal_log_count_{w}']=np.log1p(len(observed))
            expected[f'terminal_known_risk_{w}']=(labelled.y.sum()+1)/(len(labelled)+200)
            expected[f'terminal_known_support_{w}']=np.log1p(len(labelled))
            past=customer[(customer.tx_time_days<day-1)&(customer.tx_time_days>=day-1-w)]
            a=past.tx_amount.to_numpy();mean=float(a.mean()) if len(a) else 0.
            expected[f'customer_log_count_{w}']=np.log1p(len(a))
            expected[f'customer_log_mean_{w}']=np.log1p(mean)
            expected[f'log_amount_vs_customer_{w}']=np.log1p(r.tx_amount)-np.log1p(mean)
            if w==30:
                std=float(a.std()) if len(a)>=2 else 0.
                expected['customer_log_std_30']=np.log1p(std)
                expected['customer_amount_z_30']=np.clip((r.tx_amount-mean)/(std+1),-20,20)
        pair=customer[(customer.terminal_id==r.terminal_id)&(customer.tx_time_days<day-1)]
        expected['customer_terminal_log_count']=np.log1p(len(pair));expected['customer_terminal_new']=int(len(pair)==0)
        for column,v in expected.items():
            assert np.isclose(x[i,cols.index(column)],v,rtol=1e-4,atol=1e-3),(int(r.name),column,x[i,cols.index(column)],v)
            checks+=1
    protocol=read('protocol.json');splits=protocol['splits']
    for a,b in [('train','model_validation'),('model_validation','calibration'),('calibration','policy_validation'),('policy_validation','test')]:
        assert splits[a][1]+8<=splits[b][0]
    ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    assert read('selection_lock.json')['protocol_sha256']==read('run_manifest.json')['protocol_sha256']==ph
    # Outcomes across time blocks, with no policy re-selection.
    days=pred['day'];unique=np.unique(days);rng=np.random.default_rng(20261101)
    draws=rng.integers(0,len(unique),size=(2000,len(unique)))
    summaries={}
    references={'high_vs_baseline':(pred['adaptive'],pred['baseline']),
        'low_primary_vs_baseline':(low['0.005_adaptive'],low['0.005_baseline']),
        'low_primary_vs_context':(low['0.005_adaptive'],low['0.005_context_model'])}
    for name,(ours,base) in references.items():
        groups=[]
        for day in unique:
            m=days==day;y=pred['y'][m];a=pred['amount'][m];u=ours[m];b=base[m]
            groups.append([int((~y&u).sum()-(~y&b).sum()),int((y&u).sum()-(y&b).sum()),
                float(a[y&~u].sum()-a[y&~b].sum())])
        sums=np.asarray(groups)[draws].sum(axis=1)
        changes=dict(fp_removed=int((~pred['y']&base&~ours).sum()),fp_added=int((~pred['y']&~base&ours).sum()),
            fraud_recovered=int((pred['y']&~base&ours).sum()),fraud_lost=int((pred['y']&base&~ours).sum()))
        summaries[name]=dict(day_blocks=len(unique),draws=2000,
            fp_difference_interval=np.quantile(sums[:,0],[.025,.975]).tolist(),
            tp_difference_interval=np.quantile(sums[:,1],[.025,.975]).tolist(),
            unflagged_amount_difference_interval=np.quantile(sums[:,2],[.025,.975]).tolist(),changes=changes,
            limitation='Descriptive paired resampling of days in one independent synthetic holdout; not real-bank evidence or independent simulator seeds.')
    save('block_sensitivity.json',summaries)
    expected=read('test_results.json')['policies']
    for name in ['baseline','context_model','adaptive']:
        measured=metrics(pred['y'],pred['amount'],pred[name])
        for k,v in measured.items():assert np.isclose(v,expected[name][k],atol=1e-5,rtol=1e-10)
    event=dict(features={c:float(x[idx[0],j]) for j,c in enumerate(cols)},amount=float(meta['amount'][idx[0]]),mandatory_hold=False)
    save('example_input.json',event);save('example_decision.json',ContextPaymentLayer(OUT).decide(**event))
    result=dict(rows_identity_verified=len(rows),saved_model_replay_rows=256,modes_replayed=2,
        direct_history_rows=20,direct_feature_checks=checks,confirmed_label_delay_respected=True,
        training_label_embargo_respected=True,no_oracle_or_raw_ids=True,mandatory_hold_preserved=True,
        amount_alignment_rejected=True,protocol_hash_unchanged=True,hello_world_deleted=not (ROOT/'hello_world.py').exists())
    save('verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
