import os,json
from pathlib import Path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def update_status(pdb_dir, jdict):
    # jdict = {'msg': 'Out of memory', 'error_code':0, "stg": 5, "progress": 10}
    sfile = f'{pdb_dir}/status'
    if Path(sfile).exists():
        jdata = json.load(open(sfile))
        jdata.update(jdict)
        json.dump(jdata, open(sfile, 'w'), indent=4, cls=NpEncoder)
