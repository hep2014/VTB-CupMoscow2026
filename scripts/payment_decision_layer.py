"""Load the frozen experiment models and apply the selected policy.

Input is point-in-time numeric features described in feature_manifest.json.
No balance, target label or raw customer/account ID is accepted as a feature.
RESTRICT is an experiment action, not a legal finding or payment instruction.
"""
from __future__ import annotations
import json,pathlib
import lightgbm as lgb
import numpy as np
from scipy.special import expit,logit

class PaymentDecisionLayer:
    def __init__(self,model_dir):
        folder=pathlib.Path(model_dir)
        read=lambda n:json.loads((folder/n).read_text(encoding='utf8'))
        schema=read('feature_manifest.json')
        self.columns=schema['columns'];self.base_columns=schema['base_columns']
        self.base_indices=[self.columns.index(c) for c in self.base_columns]
        self.transaction=lgb.Booster(model_file=str(folder/'transaction_model.txt'))
        self.history=lgb.Booster(model_file=str(folder/'history_model.txt'))
        self.transaction_calibration=read('transaction_calibration.json')
        self.history_calibration=read('history_calibration.json')
        self.policies=read('selected_policies.json')
        self.policy=self.policies['adaptive']

    @staticmethod
    def _calibrated(model,x,calibration):
        raw=model.predict(x,num_iteration=model.best_iteration)
        return calibration['intercept']+calibration['slope']*logit(np.clip(raw,1e-8,1-1e-8))

    def predict_batch(self,features,amount,history_count,is_transfer,mandatory_hold=None):
        x=np.asarray(features,dtype='float32')
        if x.ndim!=2 or x.shape[1]!=len(self.columns):raise ValueError('Feature schema mismatch')
        if not np.isfinite(x).all():raise ValueError('Features must be finite point-in-time values')
        a=np.asarray(amount,dtype=float);h=np.asarray(history_count,dtype=float);t=np.asarray(is_transfer,dtype=float)
        if any(v.shape!=(len(x),) for v in [a,h,t]):raise ValueError('Per-operation arrays must align')
        if not np.isfinite(a).all() or (a<0).any() or not np.isfinite(h).all() or (h<0).any():raise ValueError('Invalid amount/history count')
        if not np.isin(t,[0,1]).all():raise ValueError('is_transfer must be 0 or 1')
        for column,expected in [('log_amount',np.log1p(a)),('dest_log_count',np.log1p(h)),('type_TRANSFER',t)]:
            if not np.allclose(x[:,self.columns.index(column)],expected,rtol=1e-5,atol=1e-5):
                raise ValueError(f'Operation/context alignment mismatch: {column}')
        c=self._calibrated(self.transaction,x[:,self.base_indices],self.transaction_calibration)
        e=self._calibrated(self.history,x,self.history_calibration)
        p=self.policy['parameters'];w=p['expert_weight']
        priority=(1-w)*c+w*e+p['amount_alpha']*np.log1p(a/self.policy['reference_amount'])+p['history_beta']*np.log1p(h)+p['transfer_gamma']*t
        restricted=priority>=self.policy['threshold']
        hold=np.zeros(len(x),dtype=bool) if mandatory_hold is None else np.asarray(mandatory_hold,dtype=bool)
        if hold.shape!=(len(x),):raise ValueError('mandatory_hold must align with operations')
        restricted=restricted|hold
        return dict(action=np.where(restricted,'RESTRICT','PASS'),restricted=restricted,
            reason=np.where(hold,'MANDATORY_HOLD',np.where(restricted,'RISK_POLICY','BELOW_THRESHOLD')),
            priority=priority,threshold=self.policy['threshold'],
            transaction_risk=expit(c),history_risk=expit(e))

    def decide(self,features,amount,history_count,is_transfer,mandatory_hold=False):
        if isinstance(features,dict):
            missing=set(self.columns)-features.keys()
            extra=features.keys()-set(self.columns)
            if missing or extra:raise ValueError(f'Feature schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}')
            values=[features[c] for c in self.columns]
        else:values=features
        result=self.predict_batch([values],[amount],[history_count],[is_transfer],[mandatory_hold])
        return {key:(value[0].item() if isinstance(value[0],np.generic) else value[0])
            if isinstance(value,np.ndarray) else value for key,value in result.items()}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',default=str(pathlib.Path(__file__).resolve().parents[1]/'models'/'paysim_v2'))
    parser.add_argument('--input',required=True,help='JSON with features, amount, history_count, is_transfer, optional mandatory_hold')
    args=parser.parse_args()
    event=json.loads(pathlib.Path(args.input).read_text(encoding='utf8'))
    print(json.dumps(PaymentDecisionLayer(args.model_dir).decide(**event),ensure_ascii=False,indent=2))
