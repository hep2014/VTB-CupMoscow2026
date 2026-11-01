"""Read selected PostgreSQL COPY blocks without restoring or executing dump SQL.

Only reads the archive/database. Human review decisions are never modified.
Run: python scripts/analyze_dump.py --pg-restore "C:/Program Files/PostgreSQL/17/bin/pg_restore.exe"
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, pathlib, re, subprocess, zipfile
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / '.local' / 'data' / 'handoff'
OUT = ROOT / 'research' / 'analysis'
PRIVATE = ROOT / '.local' / 'analysis'
EXPECTED = '9ab243e771159d8b5b8df18f12b671afdbc78f7c47849b1f6ea4aaa68b144a3e'
TABLES = ['review_candidates', 'sources', 'cbr_observations', 'dim_period',
          'dim_metric', 'dim_channel', 'fact_fraud_statistics', 'cbr_snapshots',
          'paysim_snapshots', 'creditcard_snapshots', 'simulation_runs',
          'ieee_snapshots', 'review_tags', 'reviews', 'review_candidate_decisions']

def unescape(value):
    if value == r'\N':
        return None
    special = {'b':'\b', 'f':'\f', 'n':'\n', 'r':'\r', 't':'\t', 'v':'\v', '\\':'\\'}
    def sub(m):
        s = m.group(1)
        if s.startswith('x'):
            return chr(int(s[1:], 16))
        if s[0].isdigit():
            return chr(int(s, 8))
        return special.get(s, s)
    return re.sub(r'\\(x[0-9a-fA-F]{1,2}|[0-7]{1,3}|.)', sub, value)

def read_copy(path):
    tables = {}
    name = None
    with path.open(encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n\r')
            if name:
                if line == r'\.':
                    name = None
                    continue
                cells = line.split('\t')
                if len(cells) != len(columns):
                    raise ValueError(f'COPY field count mismatch in {name}')
                tables[name].append(dict(zip(columns, map(unescape, cells))))
            elif line.startswith('COPY '):
                m = re.fullmatch(r'COPY ([\w.]+) \((.+)\) FROM stdin;', line)
                if not m:
                    raise ValueError(line)
                name, column_text = m.groups()
                columns = column_text.split(', ')
                tables[name] = []
    if name:
        raise ValueError('Incomplete COPY block')
    return tables

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def pg_array(text):
    # These tag arrays have simple alphanumeric members; fail on complex arrays.
    if not text or text == '{}':
        return []
    if not re.fullmatch(r'\{[a-z_,]+\}', text):
        raise ValueError(text)
    return text[1:-1].split(',')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pg-restore', default=r'C:\Program Files\PostgreSQL\17\bin\pg_restore.exe')
    parser.add_argument('--reuse-selected', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    dump = DATA / 'database.dump'
    if not dump.exists():
        with zipfile.ZipFile(ROOT / '.local' / 'vtb_antifraud_complete.zip') as z:
            z.extract('database.dump', DATA)
    with dump.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != EXPECTED:
        raise ValueError('Dump fingerprint mismatch')
    selected = DATA / 'selected_data.sql'
    if not args.reuse_selected:
        cmd = [args.pg_restore, '--data-only', '--no-owner', '--no-privileges']
        for table in TABLES:
            cmd += ['-t', table]
        subprocess.run(cmd + ['-f', str(selected), str(dump)], check=True)
    tables = read_copy(selected)
    for name, rows in tables.items():
        public = name.startswith('gold.') or name in {'meta.sources', 'silver.cbr_observations'}
        write_json((OUT if public else PRIVATE) / (name.replace('.', '_') + '.json'), rows)
    sources = {r['source_id']: r for r in tables['meta.sources']}
    reviews = tables['silver.review_candidates']
    for r in reviews:
        r['source_code'] = sources[r['source_id']]['source_code']
        r['proposed_tags'] = pg_array(r['proposed_tags'])
        r['confirmed_tags'] = pg_array(r['confirmed_tags'])
    assert len(reviews) == 338
    assert len({r['review_id'] for r in reviews}) == 338
    assert len({(r['bank_name'], r['normalized_text_sha256']) for r in reviews}) == 338
    write_json(PRIVATE / 'reviews_local.json', reviews)
    # Keep text-bearing outputs local; shared report uses aggregates and paraphrases.
    summary = {}
    for src in sorted({r['source_code'] for r in reviews}):
        group = [r for r in reviews if r['source_code'] == src]
        summary[src] = {
            'n':len(group),
            'proposed_relevance':dict(collections.Counter(r['proposed_relevance'] for r in group)),
            'human_decision':dict(collections.Counter(r['human_decision'] for r in group)),
            'months':dict(collections.Counter(r['published_month'] for r in group)),
            'proposed_tags':dict(collections.Counter(t for r in group for t in r['proposed_tags'])),
            'confirmed_tags':dict(collections.Counter(t for r in group if r['human_decision']=='approve' for t in r['confirmed_tags'])),
            'explicit_sbp_n':sum(bool(re.search(r'\bсбп\b|систем[аыуе]\s+быстрых\s+платеж', r['review_text'], re.I)) for r in group),
        }
    write_json(OUT / 'review_inventory.json', summary)
    observations = {r['observation_id']:r for r in tables['silver.cbr_observations']}
    facts = tables['gold.fact_fraud_statistics']
    assert len(facts) == 135
    assert len(observations) == 140
    for f in facts:
        o = observations[f['observation_id']]
        assert Decimal(f['metric_value']) == Decimal(o['metric_value'])
        assert all(f[k]==o[k] for k in ['metric_code','channel_code','customer_segment','observation_scope'])
        f['source_url'] = o['source_url']
        f['metric_unit'] = o['metric_unit']
    write_csv(OUT / 'cbr_facts.csv', facts)
    def value(p, c, m, segment='individuals'):
        rows=[f for f in facts if f['period_id']==p and f['channel_code']==c and f['metric_code']==m and f['customer_segment']==segment]
        if len(rows)!=1: return None
        return Decimal(rows[0]['metric_value'])
    quarterly=[]
    for p in sorted({f['period_id'] for f in facts}):
        n=value(p,'sbp','fraud_count')
        if n is None: continue
        v=value(p,'sbp','fraud_volume')*1000
        r=value(p,'sbp','reimbursement_share')
        card_n=value(p,'cards','fraud_count');card_v=value(p,'cards','fraud_volume')*1000
        quarterly.append(dict(period=p, sbp_fraud_count=int(n), sbp_fraud_rub=str(v),
            sbp_return_share_pct=str(r), sbp_mean_rub=str(v/n),
            sbp_not_returned_estimate_rub=str(v*(1-r/100)),
            cards_fraud_count=int(card_n), cards_fraud_rub=str(card_v),
            cards_to_sbp_count_ratio=str(card_n/n), sbp_to_cards_loss_ratio=str(v/card_v),
            sbp_to_cards_mean_ratio=str((v/n)/(card_v/card_n))))
    write_csv(OUT / 'cbr_sbp_quarterly.csv', quarterly)
    audit = dict(dump_sha256=digest, table_rows={k:len(v) for k,v in tables.items()},
        gold_silver_all_values_match=True, unique_reviews=True,
        limitations=['ML tables not fully scanned; inventory figures come from supplied quality metadata.',
                     'No transaction decisions or legitimate-transaction ground truth for SBP in this dump.',
                     'No new human approvals; proposed tags are not validated labels.'])
    write_json(OUT / 'audit.json', audit)
    print(json.dumps({'audit':audit,'reviews':summary,'quarters':quarterly},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
