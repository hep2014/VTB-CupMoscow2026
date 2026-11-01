"""Compare all 140 source observations to independently downloaded CBR tables."""
from html.parser import HTMLParser
from decimal import Decimal
import json, re
from analyze_dump import ROOT,OUT,write_json

class Tables(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.table=[]
        elif tag=='tr' and self.table is not None:self.row=[]
        elif tag in ('td','th') and self.row is not None:self.cell=''
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(self.cell);self.cell=None
        elif tag=='tr' and self.row is not None:
            self.table.append(self.row);self.row=None
        elif tag=='table' and self.table is not None:
            self.tables.append(self.table);self.table=None

def number(s):return Decimal(re.sub(r'\s','',s).replace(',','.'))

def main():
    rows=json.loads((OUT/'silver_cbr_observations.json').read_text(encoding='utf8'))
    lookups={}
    metrics=['fraud_count','fraud_volume','reimbursement_share','prevented_count','prevented_volume']
    individual=['cards','accounts','sbp','electronic_wallets','without_account']
    legal=['accounts','sbp']
    for y,q in [(2025,1),(2025,2),(2025,3),(2026,1),(2026,2)]:
        p=Tables();p.feed((ROOT/'data'/'public_sources'/f'cbr_{y}q{q}.html').read_text(encoding='utf8'))
        lookup={}
        for row in p.tables[0][1:]:
            label=row[0]
            roman=re.search(r'\b(IV|III|II|I)\b',label).group(1)
            quarter={'I':1,'II':2,'III':3,'IV':4}[roman]
            year=int(re.search(r'20\d\d',label).group())
            for m,v in zip(metrics,row[1:]):lookup[(year,quarter,'all','all',m)]=number(v)
        for segment,channels,t in [('individuals',individual,p.tables[1]),('legal_entities',legal,p.tables[2])]:
            for metric,row in zip(metrics[:3],t[1:]):
                for channel,v in zip(channels,row[1:]):lookup[(y,q,segment,channel,metric)]=number(v)
        lookups[(y,q)]=lookup
    checks=[]
    for r in rows:
        key=(int(r['observation_year']),int(r['observation_quarter']),r['customer_segment'],r['channel_code'],r['metric_code'])
        actual=lookups[(int(r['report_year']),int(r['report_quarter']))][key]
        checks.append(dict(observation_id=r['observation_id'],match=actual==Decimal(r['metric_value']),
            source_url=r['source_url'],dump_value=r['metric_value'],live_value=str(actual)))
    write_json(OUT/'cbr_live_verification.json',dict(checked=len(checks),matches=sum(c['match'] for c in checks),checks=checks))
    assert all(c['match'] for c in checks), 'CBR revisions detected; inspect mismatches before reporting.'
    print(f'CBR HTML verification: {len(checks)}/{len(checks)} match.')

if __name__=='__main__':main()
