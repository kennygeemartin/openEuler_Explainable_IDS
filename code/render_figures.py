#!/usr/bin/env python3
"""Regenerate nine publication-friendly figures from recorded verified metrics.
Usage: python code/render_figures.py
Requires matplotlib and numpy; offline. Saves figures/Figure01.png ... Figure09.png.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BASE=Path(__file__).resolve().parent.parent
M=json.loads((BASE/'results'/'UNSW_Benchmark_Verified_Results.json').read_text())
OUT=BASE/'figures'; OUT.mkdir(exist_ok=True)
BLUE='#14345D'; TEAL='#168E97'; ORANGE='#E3A03C'; GREEN='#2D9871'; PURPLE='#8170AB'; GRAY='#718096'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':220,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
def finish(n):
 plt.tight_layout();plt.savefig(OUT/f'Figure{n:02d}.png',bbox_inches='tight',pad_inches=.23);plt.close()
def panel(ax,x,y,w,h,header,sub,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',facecolor='#F1F7FC',edgecolor=color,lw=2));ax.text(x+w/2,y+h*.64,header,ha='center',va='center',weight='bold',color=BLUE,fontsize=10);ax.text(x+w/2,y+h*.26,sub,ha='center',va='center',color=GRAY,fontsize=8)
def arrow(ax,start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,color=BLUE,lw=1.8))
fig,ax=plt.subplots(figsize=(15,4.7));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
panels=[('Network flow\nfeatures','12 mapped fields',BLUE),('Preprocessing','Log + standardize',TEAL),('Detector','Boosted stumps',PURPLE),('XAI / evidence','Additive attribution',ORANGE),('Risk decision','Allow / review / act',GREEN)]
for i,(a,b,c) in enumerate(panels):panel(ax,.025+i*.201,.48,.165,.31,a,b,c)
for i in range(4):arrow(ax,(.192+i*.201,.635),(.224+i*.201,.635))
panel(ax,.2,.10,.60,.17,'Audit and operational safeguards','Approvals / service protection / expiry / rollback',GRAY)
ax.set_title('Proposed openEuler-oriented supervisory architecture',weight='bold',color=BLUE);finish(1)
fig,ax=plt.subplots(figsize=(12,5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
panel(ax,.29,.76,.42,.15,'UNSW-NB15-derived mapped extract','5,000 public sample records',BLUE)
arrow(ax,(.5,.75),(.5,.61));panel(ax,.29,.47,.42,.14,'Unique, consistently labelled','4,112 retained; 888 discarded',ORANGE)
parts=list(M['split_counts'].items())
for j,(key,part) in enumerate(parts):
 x=.02+.25*j;panel(ax,x,.14,.21,.17,key.replace('_',' ').title(),str(part['records'])+' records',[BLUE,PURPLE,ORANGE,GREEN][j]);arrow(ax,(.5,.46),(x+.105,.32))
ax.set_title('Four-way leakage-screened study partition',weight='bold',color=BLUE);finish(2)
fig,ax=plt.subplots(figsize=(10,4.5));keys=[k.replace('_',' ').title() for k in M['split_counts']];parts=list(M['split_counts'].values());xs=np.arange(4)
ax.bar(xs,[x['normal'] for x in parts],label='Benign',color=TEAL);ax.bar(xs,[x['intrusion'] for x in parts],bottom=[x['normal'] for x in parts],label='Intrusion',color=ORANGE)
for j,p in enumerate(parts):ax.text(j,p['records']+20,str(p['records']),ha='center',color=BLUE)
ax.set_xticks(xs,keys);ax.set_ylabel('Records');ax.set_ylim(0,2800);ax.legend(frameon=False);ax.set_title('Dataset partitions after duplicate screening',weight='bold');finish(3)
fig,ax=plt.subplots(figsize=(10,4.6));models=M['metrics'];names=['Logistic','Boosted stumps','Calibrated'];v=[models['logistic'],models['boosted_stumps'],models['calibrated_frozen']];xs=np.arange(3);w=.24
for j,(key,c) in enumerate([('accuracy',BLUE),('f1',TEAL),('auroc',ORANGE)]):
 vals=[100*t[key] for t in v];ax.bar(xs+(j-1)*w,vals,w,color=c,label=key.upper())
 for xx,val in zip(xs+(j-1)*w,vals):ax.text(xx,val+1,f'{val:.1f}',ha='center',fontsize=9)
ax.set_xticks(xs,names);ax.set_ylim(0,112);ax.legend(frameon=False,ncol=3);ax.set_ylabel('Score (%)');ax.set_title('Independent-test classifier comparison',weight='bold');finish(4)
fig,ax=plt.subplots(figsize=(6.4,5));m=models['calibrated_frozen'];matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]]);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=510)
for (i,j),n in np.ndenumerate(matrix):ax.text(j,i,str(n),ha='center',va='center',fontsize=18,fontweight='bold',color='white' if n>250 else BLUE)
ax.set_xticks([0,1],['Predicted normal','Predicted attack']);ax.set_yticks([0,1],['Actual normal','Actual attack']);ax.set_title('Calibrated test confusion matrix (threshold 0.75)');finish(5)
fig,ax=plt.subplots(figsize=(9,4.5));labs=['Expected calibration error','Brier score'];pre=[M['expected_calibration_error']['raw'],models['boosted_stumps']['brier']];post=[M['expected_calibration_error']['after_isotonic'],models['calibrated_frozen']['brier']];xs=np.arange(2);w=.33
for delta,vals,c,lab in [(-w/2,pre,GRAY,'Raw'),(w/2,post,TEAL,'Calibrated')]:
 ax.bar(xs+delta,vals,w,color=c,label=lab)
 for x,y in zip(xs+delta,vals):ax.text(x,y+.001,f'{y:.4f}',ha='center',fontsize=10)
ax.set_xticks(xs,labs);ax.set_ylim(0,.072);ax.set_ylabel('Error (smaller is better)');ax.legend(frameon=False);ax.set_title('Independent-test probability quality',weight='bold');finish(6)
fig,ax=plt.subplots(figsize=(10,5));g=M['feature_gain_ranking'][:7][::-1];ax.barh([x['name'].replace('_',' ') for x in g],[x['gain'] for x in g],color=BLUE)
ax.set_xlabel('Accumulated training split gain');ax.set_title('Model-derived feature importance (not causal attribution)',weight='bold');finish(7)
fig,axes=plt.subplots(1,3,figsize=(14,4.6));pol=M['response_policy_simulation'];xs=np.arange(len(pol));labels=['None','Block all','Risk-tier','Evidence gate']
for ax,key,title,upper,col in zip(axes,['attack_intervention_pct','benign_intervention_pct','cost_per_record'],['Attack action coverage','Benign action burden','Assumed cost per flow'],[95,5,.29],[TEAL,ORANGE,PURPLE]):
 vals=[p[key] for p in pol];bars=ax.bar(xs,vals,color=col)
 for bar,z in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,z+upper*.015,f'{z:.2f}',ha='center',fontsize=8)
 ax.set_ylim(0,upper);ax.set_xticks(xs,labels,rotation=24,ha='right');ax.set_title(title);ax.set_ylabel('%' if key!='cost_per_record' else 'Normalized units')
fig.suptitle('Simulated response-policy consequences (not live actuation)',fontsize=13,weight='bold');finish(8)
fig,ax=plt.subplots(figsize=(10,4.7));a=models['boosted_stumps'];b=models['calibrated_frozen'];labs=['Precision','Recall','False positive rate'];xx=np.arange(3);w=.35
for delta,v,c,name in [(-w/2,a,GRAY,'Raw 0.50'),(w/2,b,GREEN,'Calibrated 0.75')]:
 vals=[100*v['precision'],100*v['recall'],100*v['fpr']];ax.bar(xx+delta,vals,w,color=c,label=name)
 for x,y in zip(xx+delta,vals):ax.text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
ax.set_xticks(xx,labs);ax.set_ylim(0,110);ax.set_ylabel('Percent (%)');ax.legend(frameon=False);ax.set_title('Threshold and calibration decision trade-off',weight='bold');finish(9)
print('Generated nine figure PNG files in',OUT)
