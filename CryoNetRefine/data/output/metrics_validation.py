import os, sys
import logging
import argparse
# from pyrosetta import logging_support
import time
import subprocess
import math
# import argparse
from pathlib import Path
import re
import json

phenix_env = "/opt/phenix-1.21.1-5286/phenix_env.sh"
# ChimeraX: optional env script to source before running; executable path or name
chimerax_cmd = "/usr/bin/chimerax"  # full path or "chimerax" to use PATH

def parse_args():
    parse = argparse.ArgumentParser(description='Refinement Pipeline Options')   
    parse.add_argument('pdb_file', type=str, help='pdb file for refinement')  
    parse.add_argument('map_file', type=str, help='map file for refinement')  
    parse.add_argument("-r","--resolution",type=float,help="resolution of map file (optional)",required=False,default=None)
    args = parse.parse_args()
    return args
BFACTOR_DEFAULT = "1.00 "  # default b-factor

# Geometry goal
def rama_outliers_goal(rama_out: float, max=1.0, min=0.0):
    return float(rama_out) >= min and rama_out <= max


def rama_favored_goal(rama_fav: float, max=100.0, min=95.0):
    return float(rama_fav) >= min and rama_fav <= max


def rotamer_outliers_goal(rot_out: float, max=1.0, min=0.0):
    return float(rot_out) >= min and rot_out <= max


def rotamer_favored_goal(rot_fav: float, max=100.0, min=95.0):
    return float(rot_fav) >= min and rot_fav <= max


def cbeta_deviations_goal(cbeta_dev: float, max=1.0, min=0.0):
    return float(cbeta_dev) >= min and cbeta_dev <= max


def Bond_goal(rms_bond: float, max=0.02, min=0.0):
    return float(rms_bond) <= max and rms_bond >= min


def Angle_goal(rms_angle: float, max=2.0, min=0.0):
    return float(rms_angle) >= min and rms_angle <= max


def Clash_goal(clashscore: float, max: float = 10.0, min: float = 0.0):
    v = float(clashscore)
    return v <= max and v >= min


def rama_z_goal(rama_z: float, max=2.0, min=-2.0):
    return float(rama_z) >= min and rama_z <= max


# ============================================================
# QScore + CSscore helpers (merged; previously in val_metrics.py)
# ============================================================

def parse_vcx_line(line: str):
    """
    Parse a single-line vcx record like:
      3j2p CC_mask: 0.57 ... QScore: 0.35 CSscore: 0.62
    Returns dict of metrics (excludes leading id token).
    """
    parts = line.strip().split()
    if len(parts) <= 1:
        return {}
    kv = parts[1:]
    metrics = {}
    for i in range(len(kv) // 2):
        k = kv[i * 2].rstrip(":")
        try:
            v = float(kv[i * 2 + 1])
        except ValueError:
            continue
        metrics[k] = v
    return metrics


def load_vcx_metrics(vcx_path: str | Path):
    p = Path(vcx_path)
    if (not p.exists()) or p.stat().st_size == 0:
        return {}
    try:
        line = p.read_text().splitlines()[0]
    except Exception:
        return {}
    return parse_vcx_line(line)


def vcx_has_fields(vcx_path: str | Path, required: tuple[str, ...]) -> bool:
    m = load_vcx_metrics(vcx_path)
    return all(k in m for k in required)


def compute_qscore_chimerax(
    pdb_path: str | Path,
    map_path: str | Path,
    resolution: float,
    sigma: float = 0.4,
    chimerax_cmd: str | None = None,
    timeout_sec: int = 600,
) -> float:
    """
    Run ChimeraX qscore (headless) and return overall mean Q-Score.
    Uses module-level chimerax_cmd/chimerax_env if not passed.
    If chimerax_env is set, runs `source chimerax_env && chimerax_cmd ...` before calling ChimeraX.
    Returns NaN if ChimeraX/qscore fails or output can't be parsed.
    """
    if chimerax_cmd is None:
        chimerax_cmd = globals()["chimerax_cmd"]
    pdb_path = str(pdb_path)
    map_path = str(map_path)
    try:
        cmd = (
            f"open {pdb_path}; open {map_path}; "
            f"qscore #1 tovolume #2 usegui false mapresolution {float(resolution)} "
            f"referencegaussiansigma {float(sigma)}; exit;"
        )
        cp = subprocess.run(
                [chimerax_cmd, "--nogui", "--cmd", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        out = cp.stdout or ""
        m = re.search(
            r"Overall mean Q-Score:\s*([0-9]*\.?[0-9]+)",
            out,
            flags=re.IGNORECASE,
        )
        if not m:
            return float("nan")
        return float(m.group(1))
    except Exception:
        return float("nan")


def _is_finite_number(x) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf)


def CC_goal(cc_value: float, low_95: float, high_95: float) -> bool:
    cc_val = float(cc_value)
    return cc_val >= float(low_95) and cc_val <= float(high_95)

CC_BOUNDS = {
    "CC_mask": {"low_95": 0.4131, "high_95": 0.8347},
    "CC_volume": {"low_95": 0.4642, "high_95": 0.8156},
    "CC_peaks": {"low_95": 0.1032, "high_95": 0.7493},
    "CC_box": {"low_95": 0.3950, "high_95": 0.7951},
    "CC_mc": {"low_95": 0.4711, "high_95": 0.8372},
    "CC_sc": {"low_95": 0.4800, "high_95": 0.8168},
}

def _cc_goal(metric: str, v: float) -> bool:
    b = CC_BOUNDS[metric]
    return CC_goal(v, b["low_95"], b["high_95"])


def Q_score_goal(q_score: float, resolution: float) -> bool:
    """
    Align with cal_CS_score.py:
      Q_mean = -0.0016*d^3 + 0.0434*d^2 - 0.3956*d + 1.3366
      Q_low_95 = Q_mean - 0.126
      pass if q_score >= Q_low_95
    """
    d = float(resolution)
    q = float(q_score)
    q_mean = (-0.0016 * d**3 + 0.0434 * d**2 - 0.3956 * d + 1.3366)
    q_low_95 = q_mean - 0.126
    return q >= q_low_95

# These key names must match the field names in vcx exactly (case-sensitive)
CS_GOAL_FUNCS = {
    "rama_outliers": lambda v, res: rama_outliers_goal(v),
    "rama_favored": lambda v, res: rama_favored_goal(v),
    "rotamer_outliers": lambda v, res: rotamer_outliers_goal(v),
    "rotamer_favored": lambda v, res: rotamer_favored_goal(v),
    "cbeta_deviations": lambda v, res: cbeta_deviations_goal(v),
    "Bond": lambda v, res: Bond_goal(v),
    "Angle": lambda v, res: Angle_goal(v),
    "clashscore": lambda v, res: Clash_goal(v),
    "rama_z": lambda v, res: rama_z_goal(v),
    "QScore": lambda v, res: Q_score_goal(v, res),
    "CC_mask": lambda v, res: _cc_goal("CC_mask", v),
    "CC_volume": lambda v, res: _cc_goal("CC_volume", v),
    "CC_peaks": lambda v, res: _cc_goal("CC_peaks", v),
    "CC_box": lambda v, res: _cc_goal("CC_box", v),
    "CC_mc": lambda v, res: _cc_goal("CC_mc", v),
    "CC_sc": lambda v, res: _cc_goal("CC_sc", v),
}

def compute_csscore(metrics_dict: dict, resolution: float) -> float:
    """
    Compute CSscore in [0,1]:
      score = (#goals satisfied) / len(CS_GOAL_FUNCS)
    Missing/non-finite metrics contribute 0 (fail).
    """
    score = 0
    for k, func in CS_GOAL_FUNCS.items():
        if k not in metrics_dict:
            continue
        v = metrics_dict.get(k)
        if not _is_finite_number(v):
            continue
        try:
            if func(float(v), float(resolution)):
                score += 1
        except Exception:
            pass
    return score / float(len(CS_GOAL_FUNCS))


metrics = [
    "rama_outliers",
    "rama_favored",
    "rotamer_outliers",
    "rotamer_favored",
    "cbeta_deviations",
    "Bond",
    "Angle",
    "rama_z",
]
cc_metrics = ["CC_mask", "CC_box", "CC_mc", "CC_sc"]
goal_funcs = [m + "_goal" for m in metrics]

logger = logging.getLogger(__file__)
consoleHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
# logger.setLevel(logging.ERROR)
logger.setLevel(logging.INFO)

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def parse_vc(ccfile, per_chain_cc=False):

    CC_mask, CC_volume, CC_peaks, CC_box = None, None, None, None
    CC_mc, CC_sc = None, None
    clashscore = None
    rama_outliers, rama_allowed, rama_favored = None, None, None

    is_rama_outliers = False
    is_rotamer_outliers = False

    rotamer_outliers, rotamer_allowed, rotamer_favored = None, None, None
    cbeta_deviations = None
    chain_cc = {}
    chain_cc_region = False
    Bond, Angle, Chirality, Planarity, Dihedral = None, None, None, None, None
    # if per_chain_cc :
    #     Per chain:
    rama_z = None
    molprobity_score = None
    emringer_score = None
    with open(ccfile, "r") as f:
        output = f.readlines()
        # output = output.split('\n')
        CIS_PROLINE = 0
        CIS_GENERAL = 0
        TWISTED_PROLINE = 0
        TWISTED_GENERAL = 0
        for idx in range(len(output)):
            if output[idx].startswith("    All-atom Clashscore"):
                clashscore = float(output[idx].split(":")[1].strip())
            elif output[idx].startswith("    Ramachandran Plot:"):
                is_rama_outliers = True
                is_rotamer_outliers = False
            elif output[idx].startswith("    Rotamer:"):
                is_rama_outliers = False
                is_rotamer_outliers = True
            # elif output[idx].startswith('  ROTAMER OUTLIERS :'):
            # rotamer_outliers = float(output[idx].split(':')[1].strip())
            # rotamer_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())

            # rotamer_allowed = round(random.uniform(0, 10),2)
            # rotamer_favored = 100 - rotamer_allowed

            elif is_rama_outliers and output[idx].startswith("      Outliers :"):
                rama_outliers = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rama_outliers and output[idx].startswith("      Allowed  :"):
                rama_allowed = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rama_outliers and output[idx].startswith("      Favored  :"):
                rama_favored = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )

            elif is_rotamer_outliers and output[idx].startswith("      Outliers :"):
                rotamer_outliers = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rotamer_outliers and output[idx].startswith("      Allowed  :"):
                rotamer_allowed = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rotamer_outliers and output[idx].startswith("      Favored  :"):
                rotamer_favored = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )

            elif output[idx].startswith("    Bond      :"):
                Bond = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Angle     :"):
                Angle = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Chirality :"):
                Chirality = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Planarity :"):
                Planarity = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Dihedral  :"):
                Dihedral = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    whole:"):
                rama_z = float(output[idx].split(":")[1].strip().split()[0].strip())

            elif output[idx].startswith("    Cbeta Deviations :"):
                cbeta_deviations = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Cis-proline     :"):
                CIS_PROLINE = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Cis-general     :"):
                CIS_GENERAL = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Twisted Proline :"):
                TWISTED_PROLINE = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Twisted General :"):
                TWISTED_GENERAL = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("Map-model CC (overall)"):
                CC_mask, CC_volume, CC_peaks, CC_box = [
                    float(line.split(":")[1].strip())
                    for line in output[idx + 2 : idx + 6]
                ]
            elif output[idx].startswith("Main chain:"):
                CC_mc = float(output[idx + 2].strip().split()[0])
                chain_cc_region = False
            elif per_chain_cc and chain_cc_region:
                fields = output[idx].strip().split()
                # import pdb;pdb.set_trace()
                print(output[idx].strip())
                chain_cc[fields[0]] = fields[1]
            elif per_chain_cc and output[idx].startswith("chain ID  CC"):
                chain_cc_region = True
            elif output[idx].startswith("Side chain:"):
                try:
                    # breakpoint()
                    CC_sc = float(output[idx + 2].strip().split()[0])
                except ValueError:
                    CC_sc = 0.0
            # 添加 MolProbity score 解析
            elif "MolProbity score" in output[idx] and "=" in output[idx]:
                # 处理格式: "  MolProbity score      =   2.48"
                try:
                    parts = output[idx].split("=")
                    if len(parts) >= 2:
                        molprobity_score = float(parts[1].strip())
                except (ValueError, IndexError):
                    pass
            # EMRinger Score（phenix.emringer 输出，可追加到 .vc）
            elif "EMRinger Score:" in output[idx]:
                try:
                    # 格式: "EMRinger Score: 2.523319"
                    emringer_score = float(
                        output[idx].split("EMRinger Score:")[1].strip().split()[0]
                    )
                except (ValueError, IndexError):
                    pass
    # import pdb;pdb.set_trace()
    return {
        "CC_mask": CC_mask,
        "CC_volume": CC_volume,
        "CC_peaks": CC_peaks,
        "CC_box": CC_box,
        "CC_mc": CC_mc,
        "CC_sc": CC_sc,
        "CC_per_chain": chain_cc,
        "clashscore": clashscore,
        "rama_outliers": rama_outliers,
        "rama_allowed": rama_allowed,
        "rama_favored": rama_favored,
        "rotamer_outliers": rotamer_outliers,
        "rotamer_allowed": rotamer_allowed,
        "rotamer_favored": rotamer_favored,
        "cbeta_deviations": cbeta_deviations,
        "cis_proline": CIS_PROLINE,
        "cis_general": CIS_GENERAL,
        "twisted_proline": TWISTED_PROLINE,
        "twisted_general": TWISTED_GENERAL,
        "Bond": Bond,
        "Angle": Angle,
        "Chirality": Chirality,
        "Planarity": Planarity,
        "Dihedral": Dihedral,
        "rama_z": rama_z,
        "molprobity_score": molprobity_score,  # 添加到返回字典
        "emringer_score": emringer_score,
    }


def load_vc(vc_path):
    line = open(vc_path, "r").readlines()[0]
    kv = line.strip().split()[1:]
    vc_dict = {}
    for i in range(len(kv) // 2):
        vc_dict[kv[i * 2].strip(":")] = float(kv[i * 2 + 1])
    return vc_dict


def score_vc(vc: dict):
    clash10WT = 3
    clash20WT = 1
    ccWT = 5
    ccVal = 0.73

    weight_sum = 0
    # geometry metrics (no clashscore)
    for m, f in zip(metrics, goal_funcs):
        goal = globals()[f](vc[m])
        if goal:
            weight_sum += 1

    # clashscore
    if vc["clashscore"] < 10:
        weight_sum += clash10WT
    elif vc["clashscore"] < 20:
        weight_sum += clash20WT
    # cc
    cc_avg = sum([vc[cc] for cc in cc_metrics]) / 4.0
    if cc_avg > ccVal:
        weight_sum += ccWT

    return weight_sum


def select_vc(vc_path: list):
    assert len(vc_path) > 0
    json_res = dict()

    ret_vc = None
    max_wt = 0
    for p in vc_path:
        vc = load_vc(p)
        cur_wt = score_vc(vc)
        json_res[os.path.basename(p)] = cur_wt
        # print(vc_path[i])
        # print(cur_wt)
        if cur_wt > max_wt:
            ret_vc = p
            max_wt = cur_wt

    wkdir = os.path.dirname(ret_vc)
    emdb = os.path.basename(ret_vc).split("_")[0].split("-")[0]
    json_str = json.dumps(json_res, indent=4)
    res_path=vc_path[0].replace("rsr.vcx","refine.json")
    open(res_path, "w").write(json_str)

    return ret_vc


def save_vc(inputfile, suffix=None):
    inf = Path(inputfile)

    if suffix is None:
        basename = os.path.basename(inputfile)
        emdb = basename.split(".")[0].split("_")[0].split("-")[0]
        suffix = basename.replace(emdb, "")

    per_chain_cc = False
    cc_dict = parse_vc(inputfile, per_chain_cc)
    pdbname = inf.name.replace(suffix, "").replace(".log", "")
    names = pdbname.split("_")
    if len(names) == 2:
        pdbname = names[0]
        mapname = names[1]
    else:
        mapname = "---"
    line = f"{pdbname}"
    for k in cc_dict.keys():
        if k != "CC_per_chain" and cc_dict[k] is not None:
            line += f" {k}: {cc_dict[k]}"
    # line = f"{pdbname} CC_mask: {cc_dict['CC_mask']} CC_volume: {cc_dict['CC_volume']} CC_peaks: {cc_dict['CC_peaks']} CC_box: {cc_dict['CC_box']} CC_mc: {cc_dict['CC_mc']} CC_sc: {cc_dict['CC_sc']}  "
    print(line)
    if inputfile.endswith(".vc"):
        outpath = inputfile.replace(".vc", ".vcx")
    else:
        outpath = f"{inputfile}.vcx"

    if not "None" in line:
        with open(outpath, "w") as f:
            f.write(line + "\n")


def _write_vcx_from_metrics(vcx_path: str, pdb_id: str, metrics_dict: dict):
    """
    Write single-line vcx. Skip None values.
    """
    line = f"{pdb_id}"
    for k, v in metrics_dict.items():
        if k == "CC_per_chain":
            continue
        if v is None:
            continue
        line += f" {k}: {v}"
    if "None" not in line:
        with open(vcx_path, "w") as f:
            f.write(line + "\n")


def _needs_vcx_update(vcx_path: str) -> bool:
    # Missing or contains "None" -> rebuild
    if is_none_vcx(vcx_path):
        return True
    # Ensure we have required fields (value may be NaN, but key must exist)
    return not vcx_has_fields(vcx_path, ("QScore", "CSscore", "emringer_score"))


def is_none_vcx(path):
    if os.path.exists(path):

        with open(path, "r") as f:
            vcx = f.read()
            if len(vcx) > 0 and "None" in vcx:
                return True
            else:
                return False
    else:
        return True


def reset_bfactor(pdb_path: str, bfactor_value: str = "0.00"):
    """
    Reset bfactor values in PDB/CIF file to a default value using gemmi
    Ensures auth_comp_id field is present (copied from label_comp_id if missing).

    NOTE:
    - gemmi.cif.Loop does not support assigning to .tags/.values (read-only properties in some builds).
    - To stay robust across gemmi versions, we:
      1) use gemmi to reset B-factors and write a fresh mmCIF to a temp file
      2) post-process the written mmCIF text to inject _atom_site.auth_comp_id if missing
    """
    try:
        import gemmi
        import os
        import shutil
        import shlex
        
        # Read structure using gemmi
        structure = gemmi.read_structure(pdb_path)
        
        # Convert bfactor_value to float
        bfactor_float = float(bfactor_value)
        
        # Iterate through all models, chains, residues, and atoms
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.b_iso = bfactor_float
        
        # Write back to file
        if pdb_path.endswith(".pdb"):
            structure.write_minimal_pdb(pdb_path)
        else:
            tmp_path = pdb_path + ".tmp"
            bak_path = pdb_path + ".bak"

            # Write via gemmi first
            doc = structure.make_mmcif_document()
            doc.write_file(tmp_path)

            # Post-process written CIF to ensure _atom_site.auth_comp_id exists.
            # We copy values from _atom_site.label_comp_id.
            with open(tmp_path, "r") as f:
                lines = f.readlines()

            out_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                out_lines.append(line)
                if line.strip() != "loop_":
                    i += 1
                    continue

                # Peek forward for an _atom_site loop
                j = i + 1
                tags = []
                while j < len(lines) and lines[j].lstrip().startswith("_atom_site."):
                    tags.append(lines[j].strip())
                    j += 1

                if not tags:
                    i += 1
                    continue

                # We already appended 'loop_' line; remove the already appended tag lines
                # and re-emit them with possible insertion.
                out_lines = out_lines[:-1]  # remove loop_ we appended, we'll re-add below

                has_auth_comp_id = any(t == "_atom_site.auth_comp_id" for t in tags)
                try:
                    label_idx = tags.index("_atom_site.label_comp_id")
                except ValueError:
                    # Can't fix without label_comp_id; just keep original as-is.
                    out_lines.append("loop_\n")
                    out_lines.extend([t + "\n" for t in tags])
                    i += 1
                    continue

                insert_pos = None
                if not has_auth_comp_id:
                    if "_atom_site.auth_asym_id" in tags:
                        insert_pos = tags.index("_atom_site.auth_asym_id")
                    else:
                        insert_pos = label_idx + 1
                    tags = tags[:insert_pos] + ["_atom_site.auth_comp_id"] + tags[insert_pos:]

                # Emit updated loop header
                out_lines.append("loop_\n")
                out_lines.extend([t + "\n" for t in tags])

                # Now copy data rows for this loop.
                k = j
                while k < len(lines):
                    l = lines[k]
                    s = l.strip()
                    if s == "" or s.startswith("#") or s == "loop_" or s.startswith("_"):
                        break

                    if not has_auth_comp_id:
                        # Insert auth_comp_id token (same as label_comp_id)
                        parts = shlex.split(l, posix=True)
                        if len(parts) >= len(tags) - 1:
                            # parts currently correspond to original tag count
                            label_val = parts[label_idx]
                            parts.insert(insert_pos, label_val)
                            out_lines.append(" ".join(parts) + "\n")
                        else:
                            # If parsing weird, keep line as-is
                            out_lines.append(l)
                    else:
                        out_lines.append(l)
                    k += 1

                # Continue from where we stopped (k)
                i = k
                continue
            # Atom-site loop might have been rewritten; write back
            # Backup original then replace
            if not os.path.exists(bak_path):
                shutil.copy2(pdb_path, bak_path)
            with open(tmp_path, "w") as f:
                f.writelines(out_lines)
            os.replace(tmp_path, pdb_path)
        
        logger.info(f"Reset bfactor to {bfactor_value} for {pdb_path}")
        return True
    except Exception as e:
        logger.error(f"Error resetting bfactor for {pdb_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_validation(map_path: str, pdb_path: str, r: float):
    wkdir = os.path.dirname(pdb_path)
    # emdb = os.path.basename(pdb_path).replace(in_suffix+".pdb", "")
    emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0].split("-")[0]

    # log_path=pdb_path.replace(".pdb", ".vc")
    log_path = pdb_path.replace(".pdb", ".vc").replace(".cif", ".vc")
    vcx_path = log_path.replace(".vc", ".vcx")
    vc_incomplete = False
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                vc_incomplete = "Clashscore" not in f.read()
        except Exception:
            vc_incomplete = True
    need_phenix_run = (not os.path.exists(log_path)) or vc_incomplete
    need_vcx_update = _needs_vcx_update(vcx_path)
    ret = 0
    if need_phenix_run:
        # Reset bfactor before validation (skip for CIF files to avoid format issues)
        logger.info(f"Reset bfactor to {BFACTOR_DEFAULT.strip()} for {pdb_path}")
        reset_bfactor(pdb_path, BFACTOR_DEFAULT.strip())

        cmd0 = f"rm -f {log_path}; "
        os.system(cmd0)

        cmd1 = f"source {phenix_env} && phenix.map_model_cc {pdb_path} {map_path} resolution={r} >> {log_path}"
        # cmd2=f"phenix.molprobity {pdb_path} >> {log_path}_geo"
        cmd2 = f"phenix.molprobity {pdb_path} coot=False probe_dots=False>> {log_path}"
        logger.info(cmd1)
        logger.info(cmd2)

        t = time.perf_counter()
        ret = os.system(cmd1)
        ret = os.system(cmd2)
        validation_time = time.perf_counter() - t
        logger.info("Validation time for {:s}: {:.4f}s".format(emdb, validation_time))
        need_vcx_update = True

    # Ensure EMRinger is present in the same .vc log
    need_emringer = False
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                need_emringer = "EMRinger Score:" not in f.read()
        except Exception:
            need_emringer = True
    if need_emringer:
        logger.info(f"Computing EMRinger via phenix.emringer for {emdb}...")
        # Reset bfactor before emringer to be consistent
        reset_bfactor(pdb_path, BFACTOR_DEFAULT.strip())
        cmd3 = f"cd {wkdir}; phenix.emringer {pdb_path} {map_path} >> {log_path}"
        logger.info(cmd3)
        t_em = time.perf_counter()
        ret_em = os.system(cmd3)
        dt_em = time.perf_counter() - t_em
        if ret_em != 0:
            logger.warning(f"phenix.emringer failed for {emdb} (exit={ret_em}, time={dt_em:.2f}s)")
            ret = ret_em
        else:
            logger.info(f"EMRinger finished for {emdb} (time={dt_em:.2f}s)")
        need_vcx_update = True
    # Always (re)build vcx when requested, in THIS function only.
    if need_vcx_update and os.path.exists(log_path):
        metrics_dict = parse_vc(log_path, per_chain_cc=False)
        # Ensure required field is present even if parsing failed
        if metrics_dict.get("emringer_score", None) is None:
            metrics_dict["emringer_score"] = float("nan")
        try:
            ems = float(metrics_dict.get("emringer_score"))
            if math.isfinite(ems):
                logger.info(f"EMRinger score for {emdb}: {ems:.6f}")
            else:
                logger.warning(f"EMRinger score for {emdb}: NaN (phenix.emringer unavailable or parse failed)")
        except Exception:
            logger.warning(f"EMRinger score for {emdb}: NaN (phenix.emringer unavailable or parse failed)")
        # QScore via ChimeraX (may be NaN if unavailable)
        q_t0 = time.perf_counter()
        logger.info(f"Computing QScore via ChimeraX for {emdb} (resolution={r}, sigma=0.4)...")
        q = compute_qscore_chimerax(pdb_path=pdb_path, map_path=map_path, resolution=r)
        q_dt = time.perf_counter() - q_t0
        if math.isfinite(float(q)):
            logger.info(f"QScore for {emdb}: {q:.4f} (time={q_dt:.2f}s)")
        else:
            logger.warning(f"QScore for {emdb}: NaN (ChimeraX/qscore unavailable or parse failed; time={q_dt:.2f}s)")
        metrics_dict["QScore"] = q
        # CSscore uses same goal set as cal_CS_score.py (includes QScore goal)
        cs_t0 = time.perf_counter()
        cs = compute_csscore(metrics_dict, r)
        cs_dt = time.perf_counter() - cs_t0
        logger.info(f"CSscore for {emdb}: {cs:.4f} (time={cs_dt:.3f}s)")
        metrics_dict["CSscore"] = cs
        _write_vcx_from_metrics(vcx_path, pdb_id=emdb, metrics_dict=metrics_dict)
        ret = 0

    if ret == 0 and os.path.exists(log_path):
        return True, log_path
    else:
        return False, log_path

def run_validation_or_not(pdb_path):
    log_path = pdb_path.replace(".pdb", ".vc").replace(".cif", ".vc")
    vcx_path = pdb_path.replace(".vc", ".vcx")
    if (
        os.path.exists(log_path)
        and os.path.getsize(log_path) > 0
        and os.path.exists(vcx_path)
    ):
        comple = False
        with open(log_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                if "All-atom Clashscore" in l:
                    comple = True
        if comple:
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    args = parse_args()
    pdb_path = args.pdb_file
    # Extract pdb-id (first four characters)
    filename = os.path.basename(pdb_path)  # e.g., "6cvm_te3_a.cif"
    # Extract the first four characters as pdb-id
    pdb_id = filename[:4]  # e.g., "6cvm"
    map_path = args.map_file
    r = args.resolution
    run_validation(map_path, pdb_path, r)