"""Scientific plots from completed aggregate diagnostics, not pending timings."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root=Path(__file__).resolve().parent.parent
a=json.loads((root/'analysis/preliminary-aggregates.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.6,4.3))
points=a['activity']['activity']
axes[0].plot([x['alpha100']/100 for x in points],
             [100*x['active_edges']/x['total_edges'] for x in points],color='#2269a8',lw=2)
axes[0].set(xlabel='Query alpha',ylabel='Active edges / all graph edges',
            title='Laion: full-graph activity',xlim=(0,1),ylim=(20,60))
axes[0].yaxis.set_major_formatter(PercentFormatter())
axes[0].grid(alpha=.22)
columns=['inactive_candidate_parent_fraction','inactive_returned_parent_fraction','inactive_true_returned_parent_fraction']
for j,row in enumerate(a['trace']['records']):
    values=[100*row['pooled_fractions'][k] for k in columns]
    axes[1].bar([i+(j-.5)*.34 for i in range(3)],values,width=.32,
                label='alpha='+str(row['alpha']),color=('#2269a8','#cf7826')[j])
axes[1].set_xticks(range(3),['Rerank\ncandidates','Returned\ntop-10','Returned\ntrue neighbors'])
axes[1].set(ylabel='Inactive first-discovery parent fraction',ylim=(0,60),title='Existing no-filter trace: L=80, r=40')
axes[1].yaxis.set_major_formatter(PercentFormatter())
axes[1].legend(frameon=False,loc='upper right')
axes[1].grid(axis='y',alpha=.22)
axes[1].set_axisbelow(True)
fig.text(.5,.025,'Right: 500 tuning queries; recall differs (91.58% vs 94.18%). Discovery provenance is not causal necessity.',
         ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.08,1,1))
out=root/'figures';out.mkdir(exist_ok=True)
for ext in ('png','svg','pdf'):fig.savefig(out/('preliminary-activity-and-provenance.'+ext),dpi=180)
print(out/'preliminary-activity-and-provenance.png')
