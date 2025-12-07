from __future__ import division, print_function
import sys, os
import torch
import iotbx.pdb, iotbx.phil
from libtbx.utils import null_out
import mmtbx.model
from mmtbx.secondary_structure import manager as ss_manager
from mmtbx.secondary_structure import proteins, nucleic_acids


# Add project root to sys.path so that 'CryoNetRefine' package can be imported
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Need to go up to cryonet.refine_xk root directory (3 levels: geometry -> libs -> CryoNetRefine -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CryoNetRefine.libs.protein import get_protein_from_file_path

# SS_TYPE=['L','H','S']
ss2idx={"L":0,"H":1,"S":2}

'''prepare secondary structures'''

sec_str_master_phil_str = """
secondary_structure
  .style = box auto_align noauto
{
  protein
    .style = box auto_align
  {
    enabled = True
      .type = bool
      .style = noauto
      .help = Turn on secondary structure restraints for protein
    search_method = *ksdssp from_ca cablam
      .type = choice
      .help = Particular method to search protein secondary structure.
    distance_ideal_n_o = 2.9
      .type = float
      .short_caption = Ideal N-O distance
      .help = Target length for N-O hydrogen bond
    distance_cut_n_o = 3.5
      .type = float
      .short_caption = N-O distance cutoff
      .help = Hydrogen bond with length exceeding this value will not be \
       established
    remove_outliers = True
      .type = bool
      .short_caption = Filter out h-bond outliers in SS
      .style = tribool
      .help = If true, h-bonds exceeding distance_cut_n_o length will not be \
       established
    restrain_hbond_angles = True
      .type = bool
      .short_caption = Restrain angles around hbonds in alpha helices
    %s
    %s
  }
  nucleic_acid
    .caption = If sigma and slack are not defined for nucleic acids, the \
      overall default settings for protein hydrogen bonds will be used \
      instead.
    .style = box auto_align
  {
    enabled = True
      .type = bool
      .style = noauto
      .help = Turn on secondary structure restraints for nucleic acids
    %s
  }
  ss_by_chain = True
    .type = bool
    .help = Only applies if search_method = from_ca. \
            Find secondary structure only within individual chains. \
            Alternative is to allow H-bonds between chains. Can be \
            much slower with ss_by_chain=False. If your model is complete \
            use ss_by_chain=True. If your model is many fragments, use \
            ss_by_chain=False.
    .short_caption = Secondary structure by chain
    .expert_level = 1
  from_ca_conservative = False
    .type = bool
    .help = various parameters changed to make from_ca method more \
      conservative, hopefully to closer resemble ksdssp.
    .short_caption = Conservative mode of from_ca
  max_rmsd = 1
    .type = float
    .help = Only applies if search_method = from_ca. \
            Maximum rmsd to consider two chains with identical sequences \
            as the same for ss identification
    .short_caption = Maximum rmsd
    .expert_level = 3
  use_representative_chains = True
    .type = bool
    .help = Only applies if search_method = from_ca. \
            Use a representative of all chains with the same sequence. \
            Alternative is to examine each chain individually. Can be \
            much slower with use_representative_of_chain=False if there \
            are many symmetry copies. Ignored unless ss_by_chain is True.
    .short_caption = Use representative chains
    .expert_level = 3
  max_representative_chains = 100
    .type = float
    .help = Only applies if search_method = from_ca. \
            Maximum number of representative chains
    .short_caption = Maximum representative chains
    .expert_level = 3

  enabled = False
    .short_caption = Use secondary structure restraints
    .type = bool
    .style = noauto bold
    .help = Turn on secondary structure restraints (main switch)
}
""" % (
    proteins.helix_group_params_str,
    proteins.sheet_group_params_str,
    nucleic_acids.dna_rna_params_str,
)

# pdb_path="tmp.pdb"
pdb_path=sys.argv[1]
pickle_path=sys.argv[2]


pdb_inp = iotbx.pdb.input(file_name=pdb_path)
model = mmtbx.model.manager(model_input=pdb_inp)
model.add_crystal_symmetry_if_necessary()
m = model.deep_copy()
pdb_hierarchy = m.get_hierarchy()
asc = m.get_atom_selection_cache()
# print(m.atom_counts())

# Get residue counts

'''
count=0
for chain in model.chains():
    print(" Chain:", chain.id)
    for rg in chain.residue_groups():
        for ag in rg.atom_groups():
            # print(f"  Residue: {ag.resname} {rg.resid()} {rg.atoms_size()}")
            count+=1
            # for atom in ag.atoms():
                # print(f"   Atom: {atom.name:>4}  {atom.xyz}")
                # count+=1
print(f"Residue counts: {count}")
'''

sec_str_master_phil = iotbx.phil.parse(sec_str_master_phil_str)
ss_params = sec_str_master_phil.fetch().extract()
ss_params.secondary_structure.protein.search_method = "from_ca"
ss_params.secondary_structure.from_ca_conservative = True

ssm = ss_manager(
    pdb_hierarchy,
    atom_selection_cache=asc,
    geometry_restraints_manager=None,
    sec_str_from_pdb_file=None,
    # params=None, 
    params=ss_params.secondary_structure,
    was_initialized=False,
    mon_lib_srv=None,
    verbose=-1,
    log=null_out(),
    # log=sys.stdout,
)
filtered_ann = ssm.actual_sec_str.deep_copy()
filtered_ann.remove_short_annotations(
    helix_min_len=4, sheet_min_len=4, keep_one_stranded_sheets=True
)
helix_sel = torch.tensor(asc.selection(filtered_ann.overall_helices_selection()).as_numpy_array(),dtype=bool)
sheet_sel = torch.tensor(asc.selection(filtered_ann.overall_sheets_selection()).as_numpy_array(),dtype=bool)

prot=get_protein_from_file_path(pdb_path)
prot_len = prot.aatype.shape[0]
aa_type = torch.tensor(prot.aatype,dtype=int)
atom_pos = torch.tensor(prot.atom14_positions, requires_grad=True)
atom_mask=torch.tensor(prot.atom14_mask)
atom2res=torch.tensor(prot.atom14_mask.nonzero()[0])
atom_len=atom2res.shape[0]
# loop:0, helix:1, sheet:2
ss_types_atom=torch.zeros(atom_len,dtype=int)
ss_types_atom[helix_sel]=1
ss_types_atom[sheet_sel]=2
ss_types_res=torch.zeros(prot_len,dtype=int)
for i in range(prot_len):
    ss_types_res[i]=ss_types_atom[torch.where(atom2res==i)[0][2]]

torch.save(ss_types_res,pickle_path)
