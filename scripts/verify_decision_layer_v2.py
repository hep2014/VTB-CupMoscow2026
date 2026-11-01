"""Verify point-in-time features, saved-model replay and frozen experiment outputs."""
import hashlib,json,pathlib
import numpy as np
import pandas as pd
from decision_layer_v2 import ROOT,OUT,CACHE,COLS,metrics,save
from payment_decision_layer import PaymentDecisionLayer

def main():
    read=lambda n:json.loads((OUT/n).read_text(encoding='utf8'))
    meta=dict(np.load(CACHE/'metadata.npz'));x=np.load(CACHE/'features.npy',mmap_mode='r')
    columns=json.loads((CACHE/'columns.json').read_text())
    predictions=dict(np.load(OUT/'test_predictions.npz'))
    test=meta['step']>=355;test_rows=np.flatnonzero(test)
    assert np.array_equal(meta['row_id'][test],predictions['row_id'])
    assert np.array_equal(meta['y'][test],predictions['y'])
    assert np.array_equal(meta['amount'][test],predictions['amount'])
    assert len(np.unique(predictions['row_id']))==len(predictions['y'])==1293523
    assert not any('balance' in c or 'fraud' in c or c in ['dest_code','step'] for c in columns)
    assert np.all(x[meta['history_count']<2,columns.index('dest_log_std')]==0)
    sample=np.linspace(0,len(test_rows)-1,128,dtype=int);idx=test_rows[sample]
    layer=PaymentDecisionLayer(OUT)
    replay=layer.predict_batch(x[idx],meta['amount'][idx],meta['history_count'][idx],meta['transfer'][idx])
    assert np.array_equal(replay['restricted'],predictions['adaptive'][sample])
    all_hold=layer.predict_batch(x[idx],meta['amount'][idx],meta['history_count'][idx],meta['transfer'][idx],np.ones(len(idx),bool))
    assert all_hold['restricted'].all() and (all_hold['reason']=='MANDATORY_HOLD').all()
    try:layer.predict_batch(x[idx],meta['amount'][idx][::-1],meta['history_count'][idx],meta['transfer'][idx])
    except ValueError:alignment_rejected=True
    else:raise AssertionError('Misaligned amount was not rejected')
    expected=read('test_results.json')['policies']
    for name in ['transaction_threshold','history_threshold','adaptive','simple_rule']:
        actual=metrics(predictions['y'],predictions['amount'],predictions[name])
        for key,v in actual.items():
            assert np.isclose(v,expected[name][key],rtol=1e-10,atol=1e-5),(name,key)
    lock=read('selection_lock.json');manifest=read('run_manifest.json')
    assert lock['protocol_sha256']==manifest['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    assert lock['test_evaluated_yet'] is False
    assert not (ROOT/'hello_world.py').exists()
    # Independent direct sums for 16 recipients; no cumulative implementation here.
    chosen=[]
    for seen in [False,True]:
        pool=np.flatnonzero((meta['history_count']>0)==seen)
        chosen.extend(pool[np.linspace(0,len(pool)-1,8,dtype=int)].tolist())
    ids=meta['row_id'][chosen]
    raw=pd.read_csv(ROOT/'tmp'/'paysim_audit'/'paysim_data.sql',sep='\t',skiprows=24,nrows=6362620,header=None,names=COLS,
        usecols=['source_row_number','step','amount','name_dest','transaction_type'],
        dtype={'source_row_number':'int64','step':'int32','amount':'float64','name_dest':'str','transaction_type':'category'})
    selected=raw[raw.source_row_number.isin(ids)].set_index('source_row_number')
    relevant=raw[raw.name_dest.isin(selected.name_dest)]
    del raw
    direct_checks=0
    for i in chosen:
        r=selected.loc[meta['row_id'][i]];history=relevant[(relevant.name_dest==r.name_dest)&(relevant.step<r.step)]
        a=history.amount.to_numpy();count=len(history);recent=history[history.step>=r.step-24]
        mean=float(a.mean()) if count else 0.;high=float(a.max()) if count else 0.;std=float(a.std()) if count>=2 else 0.
        expected_history={
            'dest_seen_before':int(count>0),'dest_log_count':np.log1p(count),'dest_log_mean':np.log1p(mean),
            'dest_log_std':np.log1p(std),'dest_log_max':np.log1p(high),
            'log_amount_over_mean':np.log1p(r.amount)-np.log1p(mean),
            'log_amount_over_max':np.log1p(r.amount)-np.log1p(high),
            'dest_transfer_fraction':float((history.transaction_type=='TRANSFER').mean()) if count else 0.,
            'dest_cashout_fraction':float((history.transaction_type=='CASH_OUT').mean()) if count else 0.,
            'dest_log_age':np.log1p(r.step-history.step.min()) if count else 0.,
            'dest_log_gap':np.log1p(r.step-history.step.max()) if count else np.log1p(744),
            'dest_log_recent_count':np.log1p(len(recent)),'dest_log_recent_amount':np.log1p(recent.amount.sum())}
        for column,v in expected_history.items():
            assert np.isclose(x[i,columns.index(column)],v,rtol=1e-4,atol=1e-3),(int(meta['row_id'][i]),column,x[i,columns.index(column)],v)
            direct_checks+=1
    # Paired day-block sensitivity, descriptive only: this is one simulator run.
    days=predictions['step']//24;unique=np.unique(days);rng=np.random.default_rng(20261101)
    draws=rng.integers(0,len(unique),size=(2000,len(unique)))
    bootstrap={}
    for ref in ['transaction_threshold','history_threshold']:
        groups=[]
        for day in unique:
            m=days==day;y=predictions['y'][m];a=predictions['amount'][m]
            ours=predictions['adaptive'][m];base=predictions[ref][m]
            groups.append([int((~y&ours).sum()-(~y&base).sum()),int((y&ours).sum()-(y&base).sum()),
                float(a[y&~ours].sum()-a[y&~base].sum()),int(y.sum())])
        sums=np.asarray(groups)[draws].sum(axis=1)
        diff_recall=np.divide(sums[:,1],sums[:,3],out=np.zeros(len(sums)),where=sums[:,3]>0)
        bootstrap[ref]=dict(day_blocks=len(unique),draws=2000,
            fp_difference_interval=np.quantile(sums[:,0],[.025,.975]).tolist(),
            count_recall_difference_interval=np.quantile(diff_recall,[.025,.975]).tolist(),
            unflagged_amount_difference_interval=np.quantile(sums[:,2],[.025,.975]).tolist(),
            limitation='Descriptive paired day-block resampling of one reused synthetic holdout; not independent external validation.')
    save('block_sensitivity.json',bootstrap)
    event=dict(features={c:float(x[idx[0],j]) for j,c in enumerate(columns)},amount=float(meta['amount'][idx[0]]),
        history_count=int(meta['history_count'][idx[0]]),is_transfer=int(meta['transfer'][idx[0]]),mandatory_hold=False)
    save('example_input.json',event);save('example_decision.json',layer.decide(**event))
    save('verification.json',dict(test_row_alignment=True,rows_verified=len(test_rows),saved_model_replay_rows=len(idx),
        saved_model_replay_identical=True,mandatory_hold_always_restricted=True,misaligned_input_rejected=alignment_rejected,
        independent_direct_history_checks=direct_checks,direct_history_rows=len(chosen),
        historical_std_zero_with_fewer_than_two_events=True,protocol_hash_unchanged=True,
        no_balances_or_targets_or_ids_in_features=True,source_file_deleted=True))
    print('Verification passed: aligned rows, direct history, saved-model replay, mandatory hold, frozen protocol.')

if __name__=='__main__':main()
