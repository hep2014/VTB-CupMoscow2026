"""Evidence-linked AI reading of all 17 SBP mentions. Not human ground truth.

Deliberately separates a mention of SBP from an actual restriction of SBP.
Never modifies source decisions. Each evidence fragment is checked against source.
"""
import collections, json, pathlib, re
from analyze_dump import OUT, PRIVATE, write_json, write_csv

# Review index is only for readability. UUID and exact substring checks anchor rows.
# Codes: t transfer, s support, o opacity (reason or recovery deadline), d delay,
# v photo/video/representative identification, r repeated questions/restrictions,
# x spillover beyond one transfer. Delay >=2 days or missed promised deadline.
READING = {
22: ('outgoing', True, '2b44daa3', {
    't':'при покупке ( перевод по сбп)', 's':'обратилась в банк',
    'r':'по кругу задавала много-много раз', 'x':'заблокировала кредитку'}),
25: ('incoming', True, '18130156', {
    't':'люди не могут мне переводить средства по сбп', 's':'Звонил в ТБанк',
    'o':'неизвестно из-за чего заблокировали', 'd':'почему нужно ждать по пол месяца/месяцу',
    'x':'заблокировали личные кабинеты всех банков'}),
33: ('outgoing', True, '0eb8a376', {
    't':'отклонена операция перевода по СБП', 's':'договориться с поддержкой',
    'v':'предложил совершить видео звонок', 'x':'лишился возможности доступа в личный кабинет'}),
34: ('mixed_unclear', False, '95a119f6', {'context':'оплаты по сбп за год было 100 операций'}),
68: ('outgoing', True, '3675ea5f', {
    't':'на свой же счет в ВТБ через СБП', 's':'Позвонил в поддержку',
    'o':'без объяснения причин', 'd':'пятый день',
    'v':'фотографировал меня и мои документы', 'x':'все мои деньги по-прежнему заблокированы'}),
70: ('mixed_unclear', False, 'c0f16470', {'context':'редко делал переводы через него по СБП'}),
79: ('mixed_unclear', False, '29ed1a43', {'context':'перевел деньги по СБП на Т-банк'}),
105: ('outgoing', True, '76f8c14f', {
    't':'перечислить сумму 265 тысяч по СБП', 's':'сотрудники банка отказались',
    'x':'Заодно не были исполнены ночные регулярные автоплатежи'}),
128: ('outgoing', True, '5af250f3', {
    't':'блокировать внешний переводы по СБП', 's':'служба поддержки',
    'o':'не может мне сообщить в связи с чем связана блокировка',
    'd':'сроки проверки еще продлены на 1 день', 'v':'хотят со мной личной встречи',
    'r':'на периодической основе', 'x':'все деньги клиента блокировать'}),
131: ('outgoing', True, 'c2ff5e60', {
    't':'на Зенит через СБП. Перевод заблокировали', 's':'Служба безопасности',
    'x':'Заблокировали все мои карты'}),
163: ('outgoing', True, '27cd8e27', {
    't':'заблокировала перевод по СБП на 50000 руб.',
    'd':'Уже 15.06.2025 а карты Black и Platinum заблокированы',
    'x':'карты Black и Platinum заблокированы'}),
198: ('outgoing', True, 'af678381', {
    't':'на свой счёт по СБП', 's':'звонили из поддержки',
    'o':'Пояснить основания блокировки поддержка не может',
    'd':'Обещали решить вопрос до 4 февраля включительно. Вопрос на текущий момент не решён',
    'v':'фотографировал паспорт и меня с паспортом', 'x':'заблокировал карту'}),
201: ('incoming', True, '7da8aa38', {
    't':'перевести мне деньги по СБП', 's':'служба поддержки',
    'o':'Причем ничего больше Т-банк мне несообщил'}),
207: ('sbp_success_card_restricted', False, '13ce366d', {'context':'по СБП проблем не возникло'}),
245: ('outgoing', True, '73f25396', {
    't':'заблокировала дебетовую карту при переводе по СБП',
    's':'сотрудники не могут дать ответ', 'o':'не могут дать ответ, сколько это продлится',
    'd':'Карта блокирована с 15.07.2025', 'x':'заблокировала дебетовую карту'}),
248: ('outgoing', True, 'f8a1fc1f', {
    't':'ограничением при переводе средств на свой счет в другом банке по СБП',
    's':'Попытки обьяснить это сотрудникам банка', 'v':'требовать от меня совершение видео звонка'}),
294: ('historical_sbp_current_unclear', False, '57d06ed6', {'context':'получать переводы по СБП, собирал деньги от коллег на корпоратив, за что и был заблокирован'}),
}

LABELS = {
    't':'Ограничение перевода', 's':'Обращение / эскалация',
    'o':'Неясные причины / сроки', 'd':'Длительное ожидание',
    'v':'Фото, видео, представитель', 'r':'Повторные вопросы / ограничения',
    'x':'Затронуты карты / доступ',
}

def main():
    reviews=json.loads((PRIVATE/'reviews_local.json').read_text(encoding='utf8'))
    mentions={i for i,r in enumerate(reviews) if re.search(r'\bсбп\b|систем[аыуе]\s+быстрых\s+платеж',r['review_text'],re.I)}
    assert mentions==set(READING), (mentions,set(READING))
    annotations=[]
    for i,(channel,included,prefix,evidence) in READING.items():
        r=reviews[i]
        assert r['review_id'].startswith(prefix)
        for fragment in evidence.values():
            assert fragment in r['review_text'], (i,fragment)
        annotations.append(dict(review_id=r['review_id'], content_sha256=r['content_sha256'],
            annotation_origin='ai_evidence_reading_v1_not_human', channel=channel,
            include_in_primary_sbp_cohort=included, evidence=evidence,
            source_human_decision=r['human_decision'],
            note='Client narrative; neither unlawfulness nor false positive is established.'))
    write_json(PRIVATE/'sbp_evidence_annotations_local.json',annotations)
    included=[a for a in annotations if a['include_in_primary_sbp_cohort']]
    cohorts={'all_explicit':included,
             'outgoing':[a for a in included if a['channel']=='outgoing'],
             'incoming':[a for a in included if a['channel']=='incoming']}
    counts=[]
    for key,label in LABELS.items():
        for cohort,rows in cohorts.items():
            count=sum(key in r['evidence'] for r in rows)
            counts.append(dict(pain_code=key,pain=label,cohort=cohort,n=len(rows),count=count,
                pct=round(count/len(rows)*100,2),method='ai_evidence_reading_v1_not_human'))
    write_csv(OUT/'sbp_pain_counts.csv',counts)
    write_json(OUT/'sbp_cohort_summary.json',dict(
        mentions=len(mentions),primary_n=len(included),
        channels=dict(collections.Counter(a['channel'] for a in annotations)),
        exclusion_notes={
            reviews[34]['review_id']:'SBP mentioned as business turnover; restricted transfer channel unclear.',
            reviews[70]['review_id']:'SBP mentioned as past usage; channel of blocked transfer unspecified.',
            reviews[79]['review_id']:'Incoming SBP completed; outgoing blocked channel unspecified.',
            reviews[207]['review_id']:'Explicitly successful SBP; card-number transfer was blocked.',
            reviews[294]['review_id']:'Historical receipt of SBP, current episode channel unspecified.'},
        false_positive_count=None, true_legitimate_denominator=None,
        all_annotations_are_ai=True, all_source_human_decisions_preserved=True))
    print(json.dumps(counts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
