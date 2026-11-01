"""Download public CBR HTML and optional BERT weights. Never uploads local data."""
import argparse, datetime, hashlib, json, pathlib, urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'VTB-case-local-research/1.0'})
    with urllib.request.urlopen(req,timeout=90) as response:
        return response.read()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',action='store_true');args=p.parse_args()
    out=ROOT/'data'/'public_sources';out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for y,q in [(2025,1),(2025,2),(2025,3),(2026,1),(2026,2)]:
        url=f'https://www.cbr.ru/statistics/ib/review_{q}q_{y}/'
        b=fetch(url);name=f'cbr_{y}q{q}.html';(out/name).write_bytes(b)
        manifest.append(dict(file=name,url=url,sha256=hashlib.sha256(b).hexdigest(),accessed_on=datetime.date.today().isoformat()))
    url='https://cbr.ru/press/event/?id=28361'
    b=fetch(url);(out/'cbr_client_communication_20260305.html').write_bytes(b)
    manifest.append(dict(file='cbr_client_communication_20260305.html',url=url,sha256=hashlib.sha256(b).hexdigest(),accessed_on=datetime.date.today().isoformat()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if args.model:
        model='cointegrated/rubert-tiny2'
        revision='e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae';target=ROOT/'.local'/'models'/'rubert-tiny2';target.mkdir(parents=True,exist_ok=True)
        wanted=['config.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','tokenizer.json','model.safetensors','README.md']
        hashes={}
        for name in wanted:
            b=fetch(f'https://huggingface.co/{model}/resolve/{revision}/{name}')
            (target/name).write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
            print('Downloaded',name,len(b),flush=True)
        if 'model.safetensors' not in hashes:raise RuntimeError('Safe weights not available')
        (target/'local_manifest.json').write_text(json.dumps(dict(model=model,revision=revision,sha256=hashes),indent=2),encoding='utf8')
    print('Public source snapshots saved.',flush=True)
if __name__=='__main__':main()
