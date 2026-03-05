#!/usr/bin/env python
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np


# percent_path="bin/percent.tsv"
percent_path = "CryoNetRefine/data/output/percent.tsv"
# import percentile reference values
percentiles=pd.read_csv(percent_path,sep="\t",header=0)
metrics=[
    "CC_mask","CC_volume","CC_peaks","CC_box","CC_mc","CC_sc",
    "rama_favored","rotamer_favored","clashscore","rama_outliers",
    "rotamer_outliers","cbeta_deviations","Bond","Angle","rama_z",
    "cablam_outliers",
    "cis_proline","cis_general","twisted_proline","twisted_general",
    "Chirality","Planarity","Dihedral","molprobity_score",
    "emringer_score","QScore","CSscore"

    ]
reversed_metrics=[
    "clashscore","rama_outliers","rotamer_outliers","cbeta_deviations",
    "Bond","Angle","rama_z","cablam_outliers"
]
ref_percentiles={
    "CC_mask_ref_perc": 0.22,
    "CC_volume_ref_perc": 0.25,
    "CC_peaks_ref_perc": 0.75,
    "CC_box_ref_perc": 0.5,
    "CC_mc_ref_perc": 0.22,
    "CC_sc_ref_perc": 0.25,
    "rama_favored_ref_perc": 0.47,
    "rotamer_favored_ref_perc": 0.51,
    "clashscore_ref_perc": 0.31,
    "rama_outliers_ref_perc": 0.08,
    "rotamer_outliers_ref_perc": 0.3,
    "cbeta_deviations_ref_perc": 0.04,
    "Bond_ref_perc": 0.04,
    "Angle_ref_perc": 0.05,
    "rama_z_ref_perc": 0.46
}

# add percentile values to each metric
def add_percentile(score_dict):
    for k in metrics:
        if k not in percentiles:
            continue
        if k not in score_dict:
            continue
            
        v=score_dict[k]
        if k=="rama_z":v=abs(v)
        if k in reversed_metrics:
            idx=np.searchsorted(percentiles[k][::-1],v,side="right")
            idx=len(percentiles)-idx
        else:
            idx=np.searchsorted(percentiles[k],v)
        idx=min(idx,len(percentiles)-1)
        ip=percentiles.iloc[idx]["percent"]
        score_dict[k+"_perc"]=ip
        # print(k,v,ip)

    #  calibrate rama_out, rotamer_out, cbeta_deviations
    if score_dict['rama_outliers']==0: 
        score_dict['rama_outliers_perc']=1.0
    if score_dict['rotamer_outliers']==0:
        score_dict['rotamer_outliers_perc']=1.0
    if score_dict['cbeta_deviations']==0:
        score_dict['cbeta_deviations_perc']=1.0

    # add reference percentiles
    for k,v in ref_percentiles.items():
        score_dict[k]=v
    return score_dict

def parse_vcx(vcx_path,key='metrics'):
    vc_lines=open(vcx_path,'r').readlines()
    vc_dict=dict()
    for l in vc_lines:
        items=l.strip().split()
        score=dict()
        emdb=items[0]
        for i in range(0,(len(items)-1)//2):
            score[items[i*2+1].strip(":")]=float(items[i*2+2])
        vc_dict[key]=add_percentile(score)
    return vc_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def update_status(sfile, jdict):
    if Path(sfile).exists():
        jdata = json.load(open(sfile))
        jdata.update(jdict)
    else:
        jdata = jdict
    json.dump(jdata, open(sfile, 'w'), indent=4, cls=NpEncoder)

def vcx2json(vcx_path, status_path='', key='metrics'):
    assert os.path.exists(vcx_path)
    if not os.path.exists(status_path):
        status_path = Path(vcx_path).parent / "status" 

    vc_dict = parse_vcx(vcx_path, key)

    update_status(status_path, jdict={key: vc_dict[key],'valfile':vcx_path})

if __name__ == "__main__":
    # user defined
    vcx_path=sys.argv[1]
    if len(sys.argv)==2:
        vcx2json(vcx_path)
    elif len(sys.argv)==3:
        status_path=sys.argv[2]
        vcx2json(vcx_path, status_path)
    elif len(sys.argv)==4:
        status_path=sys.argv[2]
        key=sys.argv[3]
        vcx2json(vcx_path, status_path, key)


