"""12 controlled pupil/iris cases and 6 tissue cases, selected by fixed index."""
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, shift
import matplotlib.pyplot as plt

import pupil_ladder as pl
import label_ladder as ll
from atlas_common import ROOT, METRICS, score, calibrate, aggregate, write, gallery


def eyes():
    dest=ROOT/'results/figures/synthetic_atlas';dest.mkdir(exist_ok=True)
    cases=[];rows=[]
    for seed in (301,302,303):
        rng=np.random.default_rng(seed)
        targets=pl.people(4,rng); templates=pl.people(40,rng); donors=pl.people(24,rng)
        mean=np.mean([pl.render(p[0],rng) for p in templates],axis=0)
        refs=[pl.render(p[0],rng) for p in donors]
        for k,p in enumerate(targets):
            truth=pl.render(p[0],rng)
            donor_ids=rng.choice(len(refs),16,replace=False)
            reference=[pl.scores(refs[j],truth) for j in donor_ids]
            occluded=truth.copy();occluded[pl.Y<43]=.6
            images={'Target':truth,'Repeat':pl.render(p[0],rng),'Dilated +30%':pl.render(p[0],rng,1.3),
                    'Other eye':pl.render(p[1],rng),'Light change':np.clip(.85*truth+.08,0,1),
                    'Shift 3 px':shift(truth,(0,3),order=1,mode='constant',cval=.9),
                    'Blurred target':gaussian_filter(truth,1.5),'Occlusion':occluded,
                    'Stranger':refs[donor_ids[0]],'Mean':mean}
            row={'seed':seed,'target_index':k,'scores':{}}
            for name,im in images.items():
                if name=='Target':continue
                raw=pl.scores(im,truth)
                row['scores'][name]={'raw':raw,'c':{m:pl.c_score(raw[m],[r[m] for r in reference],m) for m in pl.METRICS}}
            rows.append(row);cases.append({'images':images})
            fig,axes=plt.subplots(2,len(images),figsize=(20,5.5))
            for j,(name,im) in enumerate(images.items()):
                axes[0,j].imshow(im,cmap='gray',vmin=0,vmax=1);axes[0,j].set_title(name,fontsize=10)
                axes[1,j].imshow(pl.segment(im),cmap='gray',vmin=0,vmax=1)
                for ax in axes[:,j]:ax.set(xticks=[],yticks=[])
            axes[0,0].set_ylabel('Intensity');axes[1,0].set_ylabel('Estimated pupil')
            fig.suptitle(f'Synthetic eye case {len(cases):02d}: fixed seed/index, not score-selected',fontsize=15)
            fig.text(.5,.02,'Light/shift/blur/occlusion transform the target image; they are not new identities. Dilation preserves latent eye parameters.\n'
                     'Illustrative model; threshold masks under occlusion are not ground truth. All 13 metric values saved.',ha='center',fontsize=10)
            fig.subplots_adjust(top=.84,bottom=.19,wspace=.05,hspace=.08)
            fig.savefig(dest/f'eye_{len(cases):02d}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    for page in range(2):
        gallery(cases[page*6:(page+1)*6],dest/f'eyes_gallery_{page+1}.png',f'Synthetic eyes: cases {page*6+1}-{page*6+6}',
                'Twelve fixed cases in total; real data are presented separately. Each row shares the same target identity for repeat and acquisition perturbations.')
    write(ROOT/'results/synthetic_atlas_eyes.json',{'n_targets':12,'seeds':[301,302,303],'selection':'first four targets per seed',
        'cases':rows,'aggregate':aggregate(rows),'limits':'Descriptive mean c only; references shared within seed; no confidence intervals. Parameters not biologically calibrated.'})


def brains():
    dest=ROOT/'results/figures/synthetic_atlas';rows=[];cases=[]
    for prior in (0,.25,1):
        rng=np.random.default_rng(101);acq,_,zyg=ll.people(ll.BASE,60,rng,prior_scale=prior)
        total=sum(acq[(f,t,'L','S0',0)] for f in range(60) for t in range(2))
        for f in (0,1):
            truth=acq[(f,0,'L','S0',0)]
            pool=[j for j in range(60) if j!=f]
            refs=[acq[(int(j),int(rng.integers(2)),'L','S1',0)] for j in rng.choice(pool,16,replace=False)]
            pop=(total-acq[(f,0,'L','S0',0)]-acq[(f,1,'L','S0',0)])/118
            images={'Target':truth,'Repeat':acq[(f,0,'L','S0',1)],'Other day':acq[(f,0,'L','S1',0)],
                    'Other hemisphere':acq[(f,0,'R','S0',0)],'Co-twin':acq[(f,1,'L','S0',0)],'Stranger':refs[0],'Mean':pop}
            ref_scores=[score(im,truth,ll.MASK) for im in refs]
            row={'prior_scale':prior,'family_index':f,'zygosity':zyg[f],'scores':{}}
            for name,im in images.items():
                if name!='Target':
                    raw=score(im,truth,ll.MASK);row['scores'][name]={'raw':raw,'c':calibrate(raw,ref_scores)}
            rows.append(row)
            filled={k:np.where(ll.MASK,im,truth) for k,im in images.items()}
            cases.append({'images':filled,'mask':ll.MASK})
            fig,axes=plt.subplots(2,7,figsize=(15,5.4))
            for j,(name,im) in enumerate(filled.items()):
                axes[0,j].imshow(im,cmap='gray',vmin=60,vmax=140);axes[0,j].set_title(name,fontsize=10)
                axes[1,j].imshow(ll.segment(im),cmap='gray',vmin=0,vmax=1)
                for ax in axes[:,j]:ax.contour(ll.MASK,[.5],colors=['#e58c21'],linewidths=.8);ax.set(xticks=[],yticks=[])
            fig.suptitle(f'Synthetic tissue case {len(cases):02d}: prior {prior:g}, {zyg[f]} co-twin',fontsize=15)
            fig.text(.5,.02,'Target unchanged outside the orange ROI. Other hemisphere is an independent side component in aligned coordinates.\n'
                     'Threshold segmentation, not real MRI or FastSurfer. Same two families reused across the three prior settings.',ha='center',fontsize=10)
            fig.subplots_adjust(top=.85,bottom=.18,wspace=.06,hspace=.08)
            fig.savefig(dest/f'tissue_{len(cases):02d}.png',dpi=155,bbox_inches='tight');plt.close(fig)
    gallery(cases,dest/'tissue_gallery.png','Six synthetic tissue conditions: two families x three priors',
            'Rows 1-2: prior 0; rows 3-4: prior 0.25; rows 5-6: prior 1. Same families across priors; not six independent people.',60,140,True)
    write(ROOT/'results/synthetic_atlas_tissue.json',{'cases':rows,'n_families':2,'n_conditions':6,'aggregate':aggregate(rows)})


if __name__=='__main__':
    torch.set_num_threads(1);eyes();brains();print('Synthetic atlas complete: 12 eyes, 6 tissue conditions')
