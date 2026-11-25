"""Render charts from the included aggregate evaluation, using matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
metrics = json.loads((root/'results/fresh-aggregate.json').read_text())['metrics']
out = root/'docs/charts'
out.mkdir(parents=True, exist_ok=True)
names = ['R1-A7','without_cc_operator-shared','R1-mmBERT','Luna-v1.2']
labels = ['A7 shared intent','Round-two selected','Fine-tuned mmBERT','LLM reference']
vals = [metrics[n]['site_alarm']['micro']['f1'] for n in names]
intervals = [metrics[n]['site_alarm']['micro']['f1_ci95'] for n in names]
fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
y = np.arange(len(names))
ax.barh(y, vals, color=['#315f8c','#8198b1','#8198b1','#8198b1'])
ax.errorbar(vals, y, xerr=np.array([[v-ci[0] for v,ci in zip(vals,intervals)], [ci[1]-v for v,ci in zip(vals,intervals)]]), fmt='none', color='#20252b', capsize=4)
ax.set(yticks=y, yticklabels=labels, xlim=(0,1), xlabel='Weighted site-alarm F1 (95% site bootstrap interval)', title='Fresh evaluation: graded-page target')
ax.invert_yaxis()
for i,v in enumerate(vals): ax.text(.98,i,f'{v:.3f}',va='center',ha='right',fontsize=9)
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'fresh-site-alarm.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
niches=list(metrics['R1-A7']['page_commercial']['niche'])
for k,name,c in [('R1-A7','A7 page commercial','#315f8c'),('without_cc_operator-shared','Round-two operator','#b06b37')]:
 typ='page_commercial' if k=='R1-A7' else 'site_operator'
 arr=[metrics[k][typ]['niche'][n]['f1'] or 0 for n in niches]
 ax.plot(arr,niches,'o-',label=name,color=c)
ax.set(xlim=(0,1),xlabel='Weighted F1 (different targets)',title='Fresh evaluation by niche');ax.legend();ax.grid(axis='x',alpha=.2)
fig.savefig(out/'fresh-niches.png',dpi=180)
