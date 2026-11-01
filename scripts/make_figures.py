"""Create deterministic, presentation-ready charts from analysis CSVs."""
import csv,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from analyze_dump import OUT,ROOT

FIG=ROOT/'assets'
BLUE='#0050E5';NAVY='#082A72';CYAN='#00AAE8';GREY='#63748D'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':NAVY,
 'axes.labelcolor':NAVY,'xtick.color':GREY,'ytick.color':NAVY,
 'svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig,name,**kwargs):
    fig.savefig(FIG/(name+'.png'),dpi=220,facecolor='white',**kwargs)
    fig.savefig(FIG/(name+'.svg'),facecolor='white',**kwargs)
    plt.close(fig)

def heatmap(compact=False):
    rows=list(csv.DictReader((OUT/'sbp_pain_counts.csv').open(encoding='utf-8-sig')))
    order=['s','x','o','d','v','r']
    cohorts=['all_explicit','outgoing','incoming']
    label={r['pain_code']:r['pain'] for r in rows}
    lookup={(r['pain_code'],r['cohort']):r for r in rows}
    values=np.array([[float(lookup[(p,c)]['pct']) for c in cohorts] for p in order])
    fig=plt.figure(figsize=(12,7.6))
    if not compact:
        fig.text(.05,.93,'Где возникает боль после ограничения СБП',fontsize=22,weight='bold')
        fig.text(.05,.875,'12 жалоб с явной связью ограничения и СБП • Т-Банк, 2025',fontsize=12,color=GREY)
    ax=fig.add_axes([.39,.235,.56,.55])
    cmap=LinearSegmentedColormap.from_list('vtb',['#F2F7FE','#B5D7FF','#387DE9','#0639A0'])
    ax.imshow(values,cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for i,p in enumerate(order):
        for j,c in enumerate(cohorts):
            row=lookup[(p,c)];val=values[i,j]
            ax.text(j,i-.04,f"{row['count']} / {row['n']}",ha='center',va='center',fontsize=18,weight='bold',color='white' if val>=65 else NAVY)
            ax.text(j,i+.27,f"{round(val):.0f}%",ha='center',va='center',fontsize=10,color='white' if val>=65 else NAVY)
    ax.set_xticks(range(3),['Все жалобы\nn = 12','Исходящий СБП\nn = 10','Входящий СБП\nn = 2'])
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=12,labelsize=11)
    ax.set_yticks(range(len(order)),[label[p].replace('Повторные вопросы / ограничения','Повторные вопросы /\nограничения') for p in order]);ax.tick_params(axis='y',length=0,pad=16,labelsize=11)
    ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(order),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=4);ax.tick_params(which='minor',bottom=False,left=False,top=False,right=False)
    for s in ax.spines.values():s.set_visible(False)
    if compact:
        save(fig,'sbp_pain_heatmap_compact',bbox_inches='tight',pad_inches=.08)
        return
    fig.text(.05,.145,'Цвет = доля в соответствующей группе. Один отзыв может содержать несколько болей.',fontsize=10,color=GREY)
    fig.text(.05,.102,'Предварительная разметка ИИ с фрагментами-доказательствами; требуется проверка человеком.',fontsize=10,color=GREY)
    fig.text(.05,.06,'Выборка жалоб не представляет клиентов банка и не измеряет долю ложных отказов.',fontsize=10,color=GREY)
    save(fig,'sbp_pain_heatmap')

def comparison():
    q=list(csv.DictReader((OUT/'cbr_sbp_quarterly.csv').open(encoding='utf-8-sig')))[-1]
    fig=plt.figure(figsize=(12,6.75))
    fig.text(.05,.92,'СБП: меньше эпизодов, выше денежный ущерб',fontsize=21,weight='bold')
    fig.text(.05,.855,'Операции без добровольного согласия • физические лица • II квартал 2026',fontsize=11,color=GREY)
    panels=[('Количество операций, тыс.',[float(q['cards_fraud_count'])/1000,float(q['sbp_fraud_count'])/1000],['237,0','129,1']),
            ('Объем операций, млрд ₽',[float(q['cards_fraud_rub'])/1e9,float(q['sbp_fraud_rub'])/1e9],['1,303','2,741']),
            ('Средняя сумма, тыс. ₽',[float(q['cards_fraud_rub'])/float(q['cards_fraud_count'])/1000,float(q['sbp_mean_rub'])/1000],['5,50','21,24'])]
    for i,(title,vals,labels) in enumerate(panels):
        ax=fig.add_axes([.07+i*.315,.29,.25,.43])
        ax.bar([0,1],vals,color=['#B3C8EB',BLUE],width=.56)
        for j,(v,label) in enumerate(zip(vals,labels)):ax.text(j,v+max(vals)*.07,label,ha='center',fontsize=17,weight='bold',color=NAVY)
        ax.set_title(title,fontsize=11,pad=22,loc='left');ax.set_xticks([0,1],['Карты','СБП'],fontsize=11)
        ax.set_ylim(0,max(vals)*1.35);ax.set_yticks([])
        ax.tick_params(axis='x',length=0)
        for s in ax.spines.values():s.set_visible(False)
    fig.text(.07,.17,'В 1,84 раза меньше операций',fontsize=11,color=GREY)
    fig.text(.385,.17,'В 2,10 раза больше ущерба',fontsize=11,color=BLUE,weight='bold')
    fig.text(.70,.17,'В 3,86 раза выше средняя сумма',fontsize=10.5,color=GREY)
    fig.text(.05,.075,'Источник: Банк России, review_2q_2026. Средняя сумма и отношения рассчитаны по таблице ЦБ.',fontsize=9,color=GREY)
    fig.text(.05,.035,'Это состоявшиеся ОБДС, а не ошибочно отклоненные законные переводы.',fontsize=9,color=GREY)
    save(fig,'cbr_sbp_vs_cards')

def trend():
    rows=list(csv.DictReader((OUT/'cbr_sbp_quarterly.csv').open(encoding='utf-8-sig')))
    lookup={r['period']:r for r in rows};periods=['20251','20252','20253','20254','20261','20262']
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    fig.subplots_adjust(left=.07,right=.96,bottom=.25,top=.72,wspace=.26)
    fig.text(.05,.92,'СБП: динамика зарегистрированных хищений',fontsize=21,weight='bold')
    fig.text(.05,.85,'Физические лица • квартальный срез 2025–2026',fontsize=11,color=GREY)
    for ax,key,scale,title in zip(axes,['sbp_fraud_count','sbp_fraud_rub'],[1000,1e9],['Количество ОБДС, тыс.','Объем ОБДС, млрд ₽']):
        vals=[float(lookup[p][key])/scale if p in lookup else np.nan for p in periods]
        ax.bar(range(6),vals,color=[BLUE]*3+['#CCCCCC']+[CYAN]*2,width=.58)
        for i,v in enumerate(vals):
            if not np.isnan(v):ax.text(i,v+max(x for x in vals if not np.isnan(x))*.035,f'{v:.1f}'.replace('.',','),ha='center',fontsize=10)
        ax.set_xticks(range(6),['I\n2025','II\n2025','III\n2025','IV\n2025','I\n2026','II\n2026'],fontsize=9)
        ax.set_ylim(0,max(x for x in vals if not np.isnan(x))*1.22);ax.set_title(title,loc='left',fontsize=11,pad=15)
        ax.spines[['top','right','left']].set_visible(False);ax.set_yticks([]);ax.tick_params(length=0)
        ax.text(3,ax.get_ylim()[1]*.3,'нет\nсреза',ha='center',fontsize=10,color=GREY)
    fig.text(.05,.105,'Источник: Банк России; сверены 140 наблюдений с сохраненными официальными HTML-страницами.',fontsize=9,color=GREY)
    fig.text(.05,.055,'IV квартал 2025: детализация СБП отсутствует в дампе. Пропуск не заменен нулем.',fontsize=9,color=GREY)
    save(fig,'cbr_sbp_trend')

if __name__=='__main__':
    FIG.mkdir(parents=True,exist_ok=True)
    heatmap(compact=True);comparison();trend();print('Saved 3 PNG + 3 SVG charts to',FIG)
