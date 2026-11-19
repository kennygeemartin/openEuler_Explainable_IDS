/**
 * Public UNSW-NB15-derived 12D extract experiment (Node >= 18).
 * Dataset: PhantomNet repository, data/external_benchmarks/unsw_nb15_canonical_12d.csv
 * Provenance: https://github.com/sriram21-09/PhantomNet/blob/main/docs/ML-8_DATASET_PROVENANCE.md
 * Does not use original UNSW official train/test partitions.
 * Model code intentionally uses only Node built-ins for exact reproducibility.
 * Run: node code/UNSW_NB15_Extract_Experiment.mjs [optional_local_dataset_path]
 */
import fs from 'node:fs';
const url='https://raw.githubusercontent.com/sriram21-09/PhantomNet/main/data/external_benchmarks/unsw_nb15_canonical_12d.csv';
const csv=process.argv[2] ? fs.readFileSync(process.argv[2],'utf8') : await (async()=>{let r=await fetch(url);if(!r.ok)throw Error('Dataset download failed '+r.status);return r.text()})();
const lines=csv.trim().split(/\r?\n/),names=lines[0].split(',').slice(0,-1),m=names.length;
let inputRows=lines.slice(1).map(t=>{let z=t.split(',');return{x:z.slice(0,12).map(Number),y:+z[12]}});
if(inputRows.length!==5000||names.length!==12)throw Error('Unexpected source dataset shape');
let recordGroups=new Map();
for(let r of inputRows){let key=r.x.join(',');if(!recordGroups.has(key))recordGroups.set(key,{x:r.x,labels:new Set()});recordGroups.get(key).labels.add(r.y);}
let rows=[...recordGroups.values()].filter(v=>v.labels.size===1).map(v=>({x:v.x,y:[...v.labels][0]}));
if(rows.length!==4112)throw Error('Unexpected deduplication result: '+rows.length);
const rng=(seed)=>()=>{seed=(seed+0x6D2B79F5)|0;let t=Math.imul(seed^(seed>>>15),1|seed);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};
let random=rng(20261119),groups=[[],[]];rows.forEach((r,i)=>groups[r.y].push(i));
groups.forEach(g=>{for(let j=g.length-1;j>0;j--){let k=Math.floor(random()*(j+1));[g[j],g[k]]=[g[k],g[j]];}});
let ix={train:[],cal:[],val:[],test:[]};for(let g of groups){let n=g.length,a=Math.floor(n*.60),b=Math.floor(n*.15),c=Math.floor(n*.10);ix.train.push(...g.slice(0,a));ix.cal.push(...g.slice(a,a+b));ix.val.push(...g.slice(a+b,a+b+c));ix.test.push(...g.slice(a+b+c));}
const log1=x=>Math.sign(x)*Math.log1p(Math.abs(x)),mu=Array(m).fill(0),sd=Array(m).fill(0);
for(let i of ix.train)for(let j=0;j<m;j++)mu[j]+=log1(rows[i].x[j]);mu.forEach((v,j)=>mu[j]=v/ix.train.length);
for(let i of ix.train)for(let j=0;j<m;j++)sd[j]+=(log1(rows[i].x[j])-mu[j])**2;
sd.forEach((v,j)=>sd[j]=Math.sqrt(v/ix.train.length)||1);
rows.forEach(r=>r.x=r.x.map((v,j)=>Math.max(-8,Math.min(8,(log1(v)-mu[j])/sd[j]))));
const sigmoid=z=>1/(1+Math.exp(-Math.max(-35,Math.min(35,z))));
function trainLogistic(steps=220){let b=Math.log(ix.train.filter(i=>rows[i].y).length/ix.train.filter(i=>!rows[i].y).length),w=Array(m).fill(0);
 for(let e=0;e<steps;e++){let gw=Array(m).fill(0),gb=0;for(let i of ix.train){let r=rows[i],p=sigmoid(b+r.x.reduce((s,x,j)=>s+x*w[j],0)),err=p-r.y;gb+=err;for(let j=0;j<m;j++)gw[j]+=err*r.x[j];}
 let rate=.26/(1+e/110);b-=rate*gb/ix.train.length;for(let j=0;j<m;j++)w[j]-=rate*(gw[j]/ix.train.length+.005*w[j]);}
 return {bias:b,weights:w,predict:x=>sigmoid(b+x.reduce((s,v,j)=>s+v*w[j],0))};}
const thresholds=Array.from({length:m},(_,j)=>{let s=ix.train.map(i=>rows[i].x[j]).sort((a,b)=>a-b),out=[];for(let k=1;k<=18;k++){let v=s[Math.floor(k*s.length/19)];if(!out.includes(v))out.push(v);}return out;});
function trainBoostedStumps(nTrees=160){let p0=ix.train.reduce((s,i)=>s+rows[i].y,0)/ix.train.length,bias=Math.log(p0/(1-p0)),logits=Array(rows.length).fill(bias),trees=[],lr=.085,lambda=6;
 for(let t=0;t<nTrees;t++){let best={gain:-1e9},totalG=0,totalH=0;
 let grad=ix.train.map(i=>{let p=sigmoid(logits[i]),g=rows[i].y-p,h=p*(1-p);totalG+=g;totalH+=h;return[i,g,h];});
 for(let j=0;j<m;j++)for(let th of thresholds[j]){let gl=0,hl=0;for(let [i,g,h] of grad)if(rows[i].x[j]<=th){gl+=g;hl+=h;}let gr=totalG-gl,hr=totalH-hl;if(hl<2||hr<2)continue;let gain=gl*gl/(hl+lambda)+gr*gr/(hr+lambda)-totalG*totalG/(totalH+lambda);
 if(gain>best.gain)best={j,th,gain,left:Math.max(-2,Math.min(2,lr*gl/(hl+lambda))),right:Math.max(-2,Math.min(2,lr*gr/(hr+lambda)))};}
 if(best.gain<=0)break;trees.push(best);for(let i=0;i<rows.length;i++)logits[i]+=rows[i].x[best.j]<=best.th?best.left:best.right;}
 return{bias,trees,predict:x=>sigmoid(bias+trees.reduce((s,t)=>s+(x[t.j]<=t.th?t.left:t.right),0))};}
function auc(y,p){let o=y.map((v,i)=>[p[i],v]).sort((a,b)=>a[0]-b[0]),n1=y.reduce((a,b)=>a+b,0),n0=y.length-n1;if(!n0||!n1)return null;let posSum=0;for(let a=0;a<o.length;){let b=a+1;while(b<o.length&&o[b][0]===o[a][0])b++;let avg=(a+1+b)/2;for(let k=a;k<b;k++)if(o[k][1])posSum+=avg;a=b;}return(posSum-n1*(n1+1)/2)/(n1*n0);}
function measures(id,p,th=.5){let tn=0,fp=0,fn=0,tp=0;id.forEach((i,k)=>{let y=rows[i].y,z=p[k]>=th;if(y){if(z)tp++;else fn++;}else{if(z)fp++;else tn++;}});let prec=tp/(tp+fp)||0,rec=tp/(tp+fn)||0;let yy=id.map(i=>rows[i].y),n=yy.length;return{n,tn,fp,fn,tp,acc:(tn+tp)/n,precision:prec,recall:rec,f1:2*prec*rec/(prec+rec)||0,fpr:fp/(fp+tn),auroc:auc(yy,p),brier:p.reduce((s,v,k)=>s+(v-yy[k])**2,0)/n};}
function isotonic(id,pRaw){let sorted=id.map(i=>[pRaw[i],rows[i].y]).sort((a,b)=>a[0]-b[0]),blocks=[];for(let [score,y] of sorted){blocks.push({lo:score,hi:score,sum:y,n:1});while(blocks.length>1){let b=blocks.at(-1),a=blocks.at(-2);if(a.sum/a.n<=b.sum/b.n)break;blocks.splice(-2,2,{lo:a.lo,hi:b.hi,sum:a.sum+b.sum,n:a.n+b.n});}}return v=>{let lo=0,hi=blocks.length-1;while(lo<hi){let mid=(lo+hi)>>1;if(blocks[mid].hi<v)lo=mid+1;else hi=mid;}return blocks[lo].sum/blocks[lo].n;};}
function ece(id,pr,nBins=10){let s=0;for(let b=0;b<nBins;b++){let n=0,pp=0,yy=0;id.forEach((i,k)=>{if(pr[k]>=b/nBins&&((b===nBins-1&&pr[k]<=1)||pr[k]<(b+1)/nBins)){n++;pp+=pr[k];yy+=rows[i].y;}});if(n)s+=n/id.length*Math.abs(pp/n-yy/n);}return s;}
function reliability(id,p,nBins=10){let out=[];for(let b=0;b<nBins;b++){let n=0,pp=0,yy=0;id.forEach((i,k)=>{if(p[k]>=b/nBins&&((b===nBins-1&&p[k]<=1)||p[k]<(b+1)/nBins)){n++;pp+=p[k];yy+=rows[i].y;}});out.push({bin:b,n,meanProbability:n?pp/n:null,observedRate:n?yy/n:null});}return out;}
const lr=trainLogistic(),bst=trainBoostedStumps(),pRaw=rows.map(r=>bst.predict(r.x)),pLog=rows.map(r=>lr.predict(r.x)),calibrate=isotonic(ix.cal,pRaw),pCal=rows.map((r,i)=>calibrate(pRaw[i]));
const bs=ix.val.filter(i=>rows[i].y===0).map(i=>pCal[i]).sort((a,b)=>a-b),th=bs[Math.min(bs.length-1,Math.floor(.97*bs.length))];
const get=(p,id)=>id.map(i=>p[i]),cls={};for(let key of Object.keys(ix))cls[key]={n:ix[key].length,benign:ix[key].filter(i=>rows[i].y===0).length,attack:ix[key].filter(i=>rows[i].y===1).length};
const groupsOfFeatures=[[0,4,5,8],[6,7,9],[2,10,11]];
const evidenceGate=(r)=>groupsOfFeatures.map(g=>g.reduce((s,j)=>s+Math.max(0,r.x[j]),0)/g.length>.35).filter(Boolean).length>=2;
function policy(mode){let a={alert:0,attackAction:0,benignAction:0,benignBlock:0,cost:0};for(let i of ix.test){let r=rows[i],p=pCal[i],alert=p>=th,act=0;if(mode===1&&alert)act=3;if(mode>=2&&alert){act=1;if(p>=Math.max(.88,th+.05))act=2;if(p>=Math.max(.985,th+.13))act=3;if(mode===3&&act>=2&&!evidenceGate(r))act=1;}
 a.alert+=+alert;a.attackAction+=+(r.y===1&&act>=2);a.benignAction+=+(r.y===0&&act>=2);a.benignBlock+=+(r.y===0&&act===3);a.cost+=r.y?[1,.96,.42,.10][act]:[0,.015,.28,1.5][act];}
return{name:['No response','Block every alert','Risk tiers','Risk + evidence gate'][mode],alertPct:100*a.alert/ix.test.length,attackCoveragePct:100*a.attackAction/cls.test.attack,benignActionPct:100*a.benignAction/cls.test.benign,benignBlockPct:100*a.benignBlock/cls.test.benign,cost:a.cost/ix.test.length};}
const gains=names.map((n,j)=>({name:n,gain:bst.trees.filter(t=>t.j===j).reduce((s,t)=>s+t.gain,0),nTrees:bst.trees.filter(t=>t.j===j).length})).sort((a,b)=>b.gain-a.gain);
const means=bst.trees.map(t=>ix.train.reduce((s,i)=>s+(rows[i].x[t.j]<=t.th?t.left:t.right),0)/ix.train.length);
const localSample=ix.test.find(i=>rows[i].y===1&&pCal[i]>=th);
let contrib=Array(m).fill(0),offset=bst.bias+means.reduce((a,b)=>a+b,0);
for(let t=0;t<bst.trees.length;t++){let tree=bst.trees[t];contrib[tree.j]+=(rows[localSample].x[tree.j]<=tree.th?tree.left:tree.right)-means[t];}
let localMargin=offset+contrib.reduce((a,b)=>a+b,0);
const shift=[];for(let a of [0,.15,.3,.6]){let rr=rng(20261120),p=[];for(let i of ix.test){let x=rows[i].x.map(v=>v+a*Math.sqrt(-2*Math.log(Math.max(rr(),1e-12)))*Math.cos(2*Math.PI*rr()));p.push(calibrate(bst.predict(x)));}shift.push({noiseSD:a,...measures(ix.test,p,th),ece:ece(ix.test,p)});}
const out={sourceUrl:url,sourceRows:inputRows.length,cleanedRows:rows.length,removedDuplicateOrAmbiguous:inputRows.length-rows.length,featureNames:names,partitions:cls,seed:20261119,trainModel:'logistic gradient boosting with 160 decision stumps; Newton leaf update; 18 cutpoints / feature',modelParameters:{nTrees:bst.trees.length,shrinkage:.085,l2:6,logisticEpochs:220},threshold:th,validationFpr:measures(ix.val,get(pCal,ix.val),th).fpr,models:{logistic:measures(ix.test,get(pLog,ix.test)),boostedStumps:measures(ix.test,get(pRaw,ix.test)),calibrated05:measures(ix.test,get(pCal,ix.test)),calibratedFrozen:measures(ix.test,get(pCal,ix.test),th)},calibration:{rawECE:ece(ix.test,get(pRaw,ix.test)),calibratedECE:ece(ix.test,get(pCal,ix.test)),rawBins:reliability(ix.test,get(pRaw,ix.test)),calBins:reliability(ix.test,get(pCal,ix.test))},gainRanking:gains,localExplanation:{recordIndex:localSample,observedLabel:rows[localSample].y,calibratedScore:pCal[localSample],offset,featureContributions:contrib,logitReconstruction:localMargin,modelLogit:Math.log(bst.predict(rows[localSample].x)/(1-bst.predict(rows[localSample].x)))},responsePolicies:[0,1,2,3].map(policy),noiseSensitivity:shift};
fs.writeFileSync('UNSW_Benchmark_Metrics.json',JSON.stringify(out,null,2));
fs.writeFileSync('UNSW_Heldout_Predictions.csv','test_index,label,raw_probability,calibrated_probability,alert\n'+ix.test.map(i=>[i,rows[i].y,pRaw[i],pCal[i],+(pCal[i]>=th)].join(',')).join('\n'));
console.log(JSON.stringify({partitions:cls,threshold:th,models:out.models,calibration:{rawECE:out.calibration.rawECE,calibratedECE:out.calibration.calibratedECE},responsePolicies:out.responsePolicies,noiseSensitivity:shift},null,2));
