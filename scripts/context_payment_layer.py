"""Apply the frozen FDH contextual decision layer; no labels or oracle inputs.

Defaults to the validation-selected 0.5% false-intervention-budget mode.
That budget is a validation target, NOT a guaranteed future false-positive rate.
All INTERVENE actions are counted; mandatory holds always take precedence.
"""
import json,pathlib
import lightgbm as lgb
import numpy as np
from scipy.special import expit,logit

class ContextPaymentLayer:
    def __init__(self,model_dir,mode='low_friction',cap=.005):
        folder=pathlib.Path(model_dir)
        read=lambda n:json.loads((folder/n).read_text(encoding='utf8'))
        schema=read('feature_manifest.json')
        self.columns=schema['columns'];self.base_indices=[self.columns.index(c) for c in schema['base_columns']]
        self.baseline=lgb.Booster(model_file=str(folder/'baseline_model.txt'))
        self.context=lgb.Booster(model_file=str(folder/'context_model.txt'))
        self.bc=read('baseline_calibration.json');self.ec=read('context_calibration.json')
        if mode=='high_recall':config=read('selected_policies.json')
        elif mode=='low_friction':config=read('low_friction_selection_lock.json')['selections'][str(float(cap))]
        else:raise ValueError('Unknown operating mode')
        self.baseline_threshold=config['baseline_threshold'];self.policy=config['adaptive'];self.mode=mode

    @staticmethod
    def score(model,x,cal):
        raw=model.predict(x)
        return cal['intercept']+cal['slope']*logit(np.clip(raw,1e-8,1-1e-8))

    def predict_batch(self,features,amount,mandatory_hold=None):
        x=np.asarray(features,dtype='float32');a=np.asarray(amount,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.columns):raise ValueError('Feature schema mismatch')
        if a.shape!=(len(x),):raise ValueError('Amount must align with feature rows')
        if not np.isfinite(x).all() or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid numeric input')
        for name,value in [('amount',a),('log_amount',np.log1p(a))]:
            if not np.allclose(x[:,self.columns.index(name)],value,rtol=1e-5,atol=1e-5):
                raise ValueError('Amount/features row identity mismatch')
        hold=np.zeros(len(x),bool) if mandatory_hold is None else np.asarray(mandatory_hold,dtype=bool)
        if hold.shape!=(len(x),):raise ValueError('mandatory_hold must align')
        b=self.score(self.baseline,x[:,self.base_indices],self.bc);e=self.score(self.context,x,self.ec)
        p=self.policy['parameters'];w=p['expert_weight']
        priority=(1-w)*b+w*e+p['amount_alpha']*np.log1p(a/self.policy['reference_amount'])
        margin=p['guard_logit_margin'];guard=np.zeros(len(x),bool) if margin is None else b>=self.baseline_threshold+margin
        intervene=hold|guard|(priority>=self.policy['threshold'])
        return dict(action=np.where(intervene,'INTERVENE','PASS'),intervene=intervene,
            reason=np.where(hold,'MANDATORY_HOLD',np.where(guard,'BASELINE_GUARD',np.where(intervene,'CONTEXT_POLICY','BELOW_THRESHOLD'))),
            baseline_intervene=b>=self.baseline_threshold,priority=priority,
            baseline_risk=expit(b),context_risk=expit(e),threshold=self.policy['threshold'])

    def decide(self,features,amount,mandatory_hold=False):
        if isinstance(features,dict):
            missing=set(self.columns)-set(features);extra=set(features)-set(self.columns)
            if missing or extra:raise ValueError(f'Feature schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}')
            features=[features[k] for k in self.columns]
        result=self.predict_batch([features],[amount],[mandatory_hold])
        return {key:(v[0].item() if isinstance(v[0],np.generic) else v[0]) if isinstance(v,np.ndarray) else v for key,v in result.items()}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',default=str(pathlib.Path(__file__).resolve().parents[1]/'models'/'context_layer'))
    p.add_argument('--input',required=True)
    p.add_argument('--mode',choices=['low_friction','high_recall'],default='low_friction')
    p.add_argument('--cap',type=float,default=.005)
    args=p.parse_args();event=json.loads(pathlib.Path(args.input).read_text(encoding='utf8'))
    print(json.dumps(ContextPaymentLayer(args.model_dir,args.mode,args.cap).decide(**event),ensure_ascii=False,indent=2))
