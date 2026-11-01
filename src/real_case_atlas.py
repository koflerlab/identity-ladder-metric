"""Local-only public-data illustrations. Raw images/identifiers never enter Git.

User requested real visual cases. Store images, source mappings and per-case scores
under the private data root; write only aggregates and methods to the repository.
No new acquisition, registration, clinical segmentation or identity identification.
"""
import sys
import json
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

import iris_codes as ic
import iris_calibration as iris
import baseline_v0 as bv
import image_ladder as il
from atlas_common import PRIVATE, ROOT, METRICS, score, calibrate, aggregate, write, gallery


def eye_item(path, flip=False):
    image=ic.load_grey(path,flip)
    seg=ic.segment(image)
    strip=ic.normalise(image,seg)
    _,valid=ic.log_gabor_code(strip)
    good=.15<=seg['ratio']<=.75 and seg['offset']<=.25 and valid.mean()>=ic.MIN_VALID_FRACTION
    if not good:raise ValueError('numerical segmentation QC')
    return {'image':image,'seg':seg,'strip':strip,'path':str(path),'flip':flip}


def eyes():
    dest=PRIVATE/'real_eyes';dest.mkdir(exist_ok=True)
    table,conflicts=iris.utiris_table();table=table[table.sensor=='NIR'].sort_values(['person','eye','path'])
    eligible=[];attempts=0;failures=0;people_screened=0
    for person,g in table.groupby('person',sort=True):
        people_screened+=1
        candidates={}
        for side,want in [('L',2),('R',1)]:
            passed=[]
            for row in g[g.eye==side].itertuples():
                attempts+=1
                try:item=eye_item(row.path,side=='R')
                except (ValueError,RuntimeError):failures+=1;continue
                passed.append(item)
                if len(passed)==want:break
            candidates[side]=passed
        if len(candidates['L'])==2 and len(candidates['R'])==1:
            eligible.append({'person':person,**candidates})
        if len(eligible)==22:break
    if len(eligible)<22:raise RuntimeError('Insufficient eligible eyes for disjoint pools')
    targets,templates,refs=eligible[:6],eligible[6:14],eligible[14:22]
    mean=np.mean([p['L'][0]['strip'] for p in templates],axis=0)
    mask=np.zeros((ic.RADIAL,ic.ANGULAR),bool);mask[4:12,50:82]=True
    rows=[];views=[]
    for i,p in enumerate(targets):
        t=p['L'][0]['strip']
        ref_left=[bv.place(t,r['L'][0]['strip'],mask,mode='sd') for r in refs]
        ref_right=[bv.place(t,r['R'][0]['strip'],mask,mode='sd') for r in refs]
        ref_l_scores=[score(v,t,mask) for v in ref_left];ref_r_scores=[score(v,t,mask) for v in ref_right]
        fills={'Repeat capture':bv.place(t,p['L'][1]['strip'],mask,mode='sd'),
               'Other eye (flipped)':bv.place(t,p['R'][0]['strip'],mask,mode='sd'),
               'Stranger':ref_left[0], 'Mean: raw':bv.place(t,mean,mask,mode='raw'),
               'Mean: affine':bv.place(t,mean,mask,mode='affine'),
               'Blurred stranger':bv.place(t,gaussian_filter(refs[0]['L'][0]['strip'],2),mask,mode='sd')}
        row={'case':i+1,'person':p['person'],'sources':{k:[v['path'] for v in p[k]] for k in ('L','R')},'scores':{}}
        for name,im in fills.items():
            raw=score(im,t,mask);reference=ref_r_scores if name.startswith('Other eye') else ref_l_scores
            row['scores'][name]={'raw':raw,'c':calibrate(raw,reference)}
        rows.append(row)
        raw_items=[p['L'][0],p['L'][1],p['R'][0],refs[0]['L'][0]]
        native_labels=['Target eye','Repeat capture','Other eye (flipped)','Stranger eye']
        views.append({'images':{k:v['image'] for k,v in zip(native_labels,raw_items)}})
        fig=plt.figure(figsize=(16,7));grid=fig.add_gridspec(2,28,height_ratios=[3,1])
        for j,(name,item) in enumerate(zip(native_labels,raw_items)):
            ax=fig.add_subplot(grid[0,j*7:(j+1)*7]);ax.imshow(item['image'],cmap='gray',vmin=0,vmax=255)
            for boundary,color in [('pupil','#e58c21'),('iris','#1e9bc6')]:
                x,y,r=item['seg'][boundary];ax.add_patch(Circle((x,y),r,fill=False,color=color,lw=1))
            ax.set_title(name,fontsize=12);ax.set(xticks=[],yticks=[])
        for j,(name,im) in enumerate({'Target strip':t,**fills}.items()):
            ax=fig.add_subplot(grid[1,j*4:(j+1)*4]);ax.imshow(im,cmap='gray',vmin=0,vmax=255,aspect='auto')
            ax.contour(mask,[.5],colors=['#e58c21'],linewidths=1);ax.set_title(name,fontsize=10);ax.set(xticks=[],yticks=[])
        fig.suptitle(f'Real UTIRIS NIR case {i+1:02d}: repeat eye, fellow eye and donor fills',fontsize=16)
        fig.text(.5,.02,'Orange/blue circles: algorithm-estimated pupil/limbus, not ground truth. Lower row: normalized iris strips with a filled patch.\n'
                 'Donor alignment/intensity fit uses visible strip pixels only. Template and reference people are disjoint from targets.\n'
                 'Repeat captures are from the same sitting; fellow-eye labels rely on dataset metadata. Local research copy; no redistribution.',ha='center',fontsize=10)
        fig.subplots_adjust(top=.86,bottom=.20,hspace=.25,wspace=.4)
        fig.savefig(dest/f'case_{i+1:02d}.png',dpi=150,bbox_inches='tight');plt.close(fig)
        print(f'Real iris case {i+1}/6 saved',flush=True)
    gallery(views,dest/'gallery.png','Six real UTIRIS NIR cases: same eye, fellow eye, stranger',
            'First six numerically eligible people, not selected by similarity score. Same-sitting repeat captures; other-eye view is horizontally flipped.\n'
            'Local research illustrations; source filenames and identities are omitted. Full pages include estimated boundaries and all donor-fill strips.',0,255)
    write(dest/'case_ledger.json',{'cases':rows,'template_people':[p['person'] for p in templates],
        'reference_people':[p['person'] for p in refs],'template_paths':[p['L'][0]['path'] for p in templates],
        'reference_paths':[[p['L'][0]['path'],p['R'][0]['path']] for p in refs]})
    write(ROOT/'results/real_atlas_iris.json',{'dataset':'UTIRIS V.1, NIR','n_targets':6,'n_template_people':8,'n_reference_people':8,
        'people_screened':people_screened,'images_screened':attempts,'numerical_qc_failures':failures,'parser_side_conflicts_all_sensors':conflicts,
        'mask_shape':[16,128],'mask_pixels':int(mask.sum()),'aggregate':aggregate(rows),
        'selection':'First six eligible people in stable person/path order; disjoint 8-person template and 8-person reference pools',
        'limits':['Descriptive selected cases only; numerical circle-fit QC is not ground-truth segmentation.',
                  'Known dataset side-label issues remain possible after excluding explicit folder/filename conflicts.',
                  'Within-sitting repeats, not longitudinal acquisitions. Eight reference people: coarse c resolution.',
                  'All donor strips use visible-tissue alignment; means shown raw and affine-fitted separately.',
                  'Photographs, per-case scores and source mappings remain in the private local atlas; no raw images in Git.'],
        'source':'https://utiris.wordpress.com/2014/03/04/university-of-tehran-iris-image-repository/'})


def mri():
    import nibabel as nib
    dest=PRIVATE/'real_mri';dest.mkdir(exist_ok=True)
    source=Path(il.PRIVATE)
    m=pd.read_csv(source/'manifest.csv').fillna('');qc=pd.read_csv(source/'qc.csv')
    bad=set(il.qc_outliers(qc));m=m[(m.dataset=='QTIM')&~m.scan.isin(bad)].copy()
    brain=il.block_mean(nib.load(source/'template/mni2009a_sym_2mm_mask.nii.gz').get_fdata())>=.5
    z=17;brain2=brain[:,:,z].T
    mask=np.zeros(brain2.shape,bool);mask[15:26,5:13]=True;mask &= brain2
    def image(row):
        with np.load(source/'features'/f'{row.scan}.npz',allow_pickle=False) as data:
            v=np.maximum(0,100+30*data['intensity'])
        return np.where(brain2,v[:,:,z].T,0)
    people={p:g.sort_values(['years','session','scan']) for p,g in m.groupby('person',sort=True)}
    targets=[g for g in people.values() if g.session.nunique()>=2][:6]
    if len(targets)<6:raise RuntimeError('Not enough longitudinal MRI cases')
    target_families={g.iloc[0].family for g in targets}
    pool=[g.iloc[0] for g in people.values() if g.iloc[0].family not in target_families]
    if len(pool)<12:raise RuntimeError('Not enough unrelated MRI donor/template people')
    refs,templates=pool[:8],pool[8:]
    ref_images=[image(r) for r in refs];mean=np.mean([image(r) for r in templates],axis=0)
    rows=[];views=[]
    for i,g in enumerate(targets):
        base=g.iloc[0];later=g[g.session!=base.session].iloc[0]
        truth=image(base);other=image(later)
        fills={'Later visit':other,'Other hemisphere':truth[:,::-1], 'Stranger':ref_images[0],
               'Population mean':mean,'Blurred stranger':gaussian_filter(ref_images[0],1)}
        ref_scores=[score(im,truth,mask) for im in ref_images]
        ref_mirror_scores=[score(im[:,::-1],truth,mask) for im in ref_images]
        row={'case':i+1,'person':base.person,'family':base.family,'target_scan':base.scan,'later_scan':later.scan,
             'target_source':base.path,'later_source':later.path,'scores':{}}
        filled={'Target':truth}
        for name,im in fills.items():
            raw=score(im,truth,mask)
            row['scores'][name]={'raw':raw,'c':calibrate(raw,ref_mirror_scores if name=='Other hemisphere' else ref_scores)}
            filled[name]=np.where(mask,im,truth)
        rows.append(row);views.append({'images':filled,'mask':mask})
        fig,axes=plt.subplots(2,6,figsize=(14,7),gridspec_kw={'height_ratios':[2,1]})
        for j,(name,im) in enumerate(filled.items()):
            axes[0,j].imshow(im,cmap='gray',vmin=0,vmax=180,origin='lower',interpolation='nearest')
            axes[0,j].contour(mask,[.5],colors=['#e58c21'],linewidths=1)
            if j==0:axes[0,j].contour(mask[:,::-1],[.5],colors=['#199bc2'],linewidths=1)
            axes[0,j].set_title(name,fontsize=11)
            axes[1,j].imshow(im[15:26,5:13],cmap='gray',vmin=0,vmax=180,origin='lower',interpolation='nearest')
            for ax in axes[:,j]:ax.set(xticks=[],yticks=[])
        fig.suptitle(f'Real QTIM MRI case {i+1:02d}: same person later, opposite hemisphere, donor tissue',fontsize=15)
        fig.text(.5,.015,'Existing skull-stripped, registered T1 derivatives on a 6-mm grid; this is NOT native-resolution MRI or a segmentation result.\n'
                 'Orange: evaluated patch; blue: its mirror location. Lower row enlarges the same patch without adding detail.\n'
                 'Fixed template-space slice/ROI; same-cohort donors, target families excluded. Later visit is years apart, not a repeat sitting. QTIM CC0.',ha='center',fontsize=10)
        fig.subplots_adjust(top=.88,bottom=.20,hspace=.10,wspace=.12)
        fig.savefig(dest/f'case_{i+1:02d}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    gallery(views,dest/'gallery.png','Six real QTIM MRI cases on the registered 6-mm grid',
            'Target, longitudinal visit, within-scan opposite hemisphere, unrelated donor, mean and blurred donor. Same template-space ROI.\n'
            'Derived low-resolution T1 intensities; no new registration or segmentation. Source participant identifiers omitted. QTIM / OpenNeuro ds004169 v1.0.8, CC0.',0,180,True)
    write(dest/'case_ledger.json',{'cases':rows,'reference_scans':[r.scan for r in refs],'template_scans':[r.scan for r in templates]})
    write(ROOT/'results/real_atlas_mri.json',{'dataset':'QTIM OpenNeuro ds004169 v1.0.8','license':'CC0 (local dataset_description.json)',
        'cache_run':il.RUN,'grid_mm':6,'n_target_people':6,'n_target_families':len(target_families),'n_reference_people':len(refs),
        'n_template_people':len(templates),'slice_index':z,'roi_bounds_display':[15,26,5,13],'mask_pixels':int(mask.sum()),
        'aggregate':aggregate(rows),'selection':'First six QC-passing people with distinct visits in sorted manifest order',
        'limits':['Derived 6-mm intensities: interpolation, smoothing and registration can affect scores.',
                  'Not a native-resolution or BraTS inpainting model benchmark. No segmentation performed.',
                  'Later visits include ageing/time differences; not test-retest reliability.',
                  'Opposite hemisphere shares scan-level nuisance. Reference side relation matched; protocol fields not individually verified.',
                  'Small selected case atlas; descriptive averages only; no confidence intervals.',
                  'Per-case scores, images and source mappings remain in the private data folder.']})
    print('Real MRI atlas complete: 6 people, 2 visits each',flush=True)


def native_mri():
    """Native-resolution context only; measurements remain on the registered grid."""
    import nibabel as nib
    ledger=json.loads((PRIVATE/'real_mri/case_ledger.json').read_text())
    cases=[]
    for row in ledger['cases']:
        images={}
        for prefix,key in [('Baseline','target_source'),('Later visit','later_source')]:
            ni=nib.as_closest_canonical(nib.load(row[key]))
            a=ni.get_fdata(dtype=np.float32)
            finite=a[np.isfinite(a)&(a>0)]
            lo,hi=np.percentile(finite,[1,99])
            # Fixed relative field-of-view plane, not selected for appearance or similarity.
            z=round(.62*(a.shape[2]-1));y=round(.55*(a.shape[1]-1))
            images[prefix+' axial']=np.clip((a[:,:,z].T[::-1]-lo)/max(hi-lo,1e-6),0,1)
            if prefix=='Baseline':images['Baseline coronal']=np.clip((a[:,y,:].T[::-1]-lo)/max(hi-lo,1e-6),0,1)
        cases.append({'images':images})
    gallery(cases,PRIVATE/'real_mri/native_gallery.png','Six real QTIM cases: native T1 anatomy for visual context',
            'Original image grids, canonical orientation; fixed relative field-of-view slices and per-volume display windows.\n'
            'These native views are NOT registered across visits and are NOT the images used for patch scores. Corresponding-patch comparisons use the separate registered 6-mm figures. QTIM CC0.')
    print('Native MRI context gallery saved: 6 cases')


if __name__=='__main__':
    torch.set_num_threads(1)
    try:
        {'iris':eyes,'mri':mri,'native':native_mri}[sys.argv[1]]()
    except Exception as exc:
        (PRIVATE/'atlas_error_local.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise RuntimeError(f'Local atlas failed ({type(exc).__name__}); details kept locally') from None
