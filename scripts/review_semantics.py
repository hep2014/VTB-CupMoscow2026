"""Offline RuBERT semantic triage of every review, not a calibrated classifier.

Uses normalized CLS embeddings as recommended by the original model card.
The cosine score ranks evidence for review; it is NOT a probability, a human
label, a fraud verdict, or an estimate of the prevalence of false declines.
All model weights must already be local. No inference API or upload is used.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, re
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from analyze_dump import OUT, ROOT, PRIVATE, write_json, write_csv

PROTOTYPES = {
 'no_explanation':[
    'Банк не объясняет причину блокировки перевода и не сообщает, когда снимет ограничение.',
    'Поддержка отвечает ожидайте, не называет основание ограничения и сроки разблокировки.'],
 'repeated_verification':[
    'После подтверждения перевода банк снова требует пройти проверку и ответить на те же вопросы.',
    'Я прошел идентификацию, но перевод снова заблокировали, нужно повторное подтверждение.'],
 'long_unblock':[
    'Мои деньги заблокированы уже неделю, сроки разблокировки постоянно переносят.',
    'Банк обещал снять ограничения вчера, но несколько дней не могу пользоваться деньгами.'],
 'branch_visit':[
    'Для разблокировки счета банк требует лично приехать в отделение с паспортом.',
    'Живу далеко от офиса банка, а без личного посещения отделения не снимают ограничения.'],
 'support_contact':[
    'После блокировки перевода я звоню в поддержку банка, сотрудники переводят между отделами.',
    'Обращался в чат и службу безопасности, но вопрос с ограничением денег не решен.'],
 'transfer_restricted':[
    'Банк отклонил мой перевод через СБП на собственный счет в другом банке.',
    'Не могу отправить или получить деньги, перевод заблокирован как подозрительный.'],
 'verification_burden':[
    'Чтобы подтвердить перевод, банк требует видеозвонок, фото паспорта или встречу с представителем.',
    'Банк навязывает сбор биометрии, без селфи и личной встречи нельзя разблокировать деньги.'],
 'spillover':[
    'Из-за одной подозрительной операции заблокировали все карты и доступ в личный кабинет.',
    'После блокировки перевода не проходят другие платежи, нельзя погасить кредит.'],
}
NEGATIVES = [
 'Карту доставили вовремя, все работает, поддержка помогла решить вопрос.',
 'Банк не начислил обещанный кешбэк и неверно рассчитал проценты по вкладу.',
 'Перевод по СБП прошел успешно, никаких ограничений не было.',
 'Я забыл пароль от приложения и заблокировал карту самостоятельно.',
]

def chunks(tokenizer,text,window=256,overlap=48):
    ids=tokenizer.encode(text,add_special_tokens=False)
    size=window-2
    return [tokenizer.build_inputs_with_special_tokens(ids[i:i+size]) for i in range(0,max(1,len(ids)),size-overlap)]

def encode(tokenizer,model,items,batch_size=16):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(items),batch_size):
            batch=tokenizer.pad({'input_ids':items[i:i+batch_size]},padding=True,return_tensors='pt')
            emb=model(**batch).last_hidden_state[:,0,:]
            emb=torch.nn.functional.normalize(emb,p=2,dim=1)
            out.append(emb.cpu().numpy())
    return np.concatenate(out,axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',default=str(ROOT/'.local'/'models'/'rubert-tiny2'))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    model_dir=pathlib.Path(args.model_dir)
    manifest=json.loads((model_dir/'local_manifest.json').read_text(encoding='utf8'))
    for name,digest in manifest['sha256'].items():
        with (model_dir/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==digest, name
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_safetensors=True)
    model.eval()
    reviews=json.loads((PRIVATE/'reviews_local.json').read_text(encoding='utf8'))
    flat=[];slices=[]
    for r in reviews:
        start=len(flat);flat.extend(chunks(tokenizer,r['review_text']));slices.append((start,len(flat)))
    vectors=encode(tokenizer,model,flat,args.batch_size)
    proto_texts=[s for ss in PROTOTYPES.values() for s in ss]+NEGATIVES
    proto=encode(tokenizer,model,[tokenizer.encode(s,add_special_tokens=True) for s in proto_texts],args.batch_size)
    similarities=vectors@proto.T
    labels=list(PROTOTYPES)
    ranking=[];details=[]
    for r,(start,end) in zip(reviews,slices):
        scores=[]
        for j,tag in enumerate(labels):
            block=similarities[start:end,2*j:2*j+2]
            chunk_index,prototype_index=np.unravel_index(np.argmax(block),block.shape)
            cosine=float(block.max())
            negative=float(similarities[start+chunk_index,len(labels)*2:].max())
            scores.append(dict(tag=tag,cosine=round(cosine,6),
                negative_margin=round(cosine-negative,6),chunk_index=int(chunk_index),
                prototype=PROTOTYPES[tag][prototype_index],
                evidence_chunk=tokenizer.decode(flat[start+chunk_index],skip_special_tokens=True)))
        scores.sort(key=lambda x:x['cosine'],reverse=True)
        details.append(dict(review_id=r['review_id'],source_code=r['source_code'],
            source_human_decision=r['human_decision'],chunk_count=end-start,
            scores=scores,annotation_status='unvalidated_machine_ranking'))
        ranking.append(dict(review_id=r['review_id'],source_code=r['source_code'],
            explicit_sbp_mention=bool(re.search(r'\bсбп\b',r['review_text'],re.I)),
            top1=scores[0]['tag'],top1_cosine=scores[0]['cosine'],
            top2=scores[1]['tag'],top2_cosine=scores[1]['cosine'],
            top3=scores[2]['tag'],top3_cosine=scores[2]['cosine'],
            source_human_decision=r['human_decision'],needs_human_review=True))
    write_json(PRIVATE/'bert_semantic_evidence_local.json',details)
    write_csv(PRIVATE/'bert_review_ranking.csv',ranking)
    # Empty target labels, separate from source labels: no fabricated human approvals.
    review_template=[]
    for r in reviews:
        row=dict(review_id=r['review_id'],source_code=r['source_code'],review_text=r['review_text'],
            source_human_decision=r['human_decision'],sbp_scope='',legal_framework='',
            reviewer_id='',reviewed_at='',notes='')
        row.update({tag:'' for tag in labels})
        review_template.append(row)
    write_csv(PRIVATE/'human_label_template_local.csv',review_template)
    np.savez_compressed(PRIVATE/'bert_embeddings_local.npz',embeddings=vectors,
        slices=np.array(slices),review_ids=np.array([r['review_id'] for r in reviews]))
    write_json(OUT/'bert_run.json',dict(model=manifest['model'],revision=manifest['revision'],
        reviews=len(reviews),chunks=len(flat),embedding_dimension=int(vectors.shape[1]),
        window_tokens=256,overlap_tokens=48,pooling='L2-normalized CLS',
        torch_version=torch.__version__,device='cpu',network='offline_only',
        prototypes=PROTOTYPES,negatives=NEGATIVES,
        trained_on_human_labels=False,calibrated=False,
        quality_metrics=None,
        limitation='Cosine rankings only; thresholds and precision/recall require independent human labels.'))
    print(json.dumps(dict(reviews=len(reviews),chunks=len(flat),dimension=int(vectors.shape[1]),
        model_revision=manifest['revision'],offline=True),indent=2))

if __name__=='__main__':main()
