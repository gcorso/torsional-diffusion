import pickle, tqdm
from argparse import ArgumentParser
import pandas as pd
from rdkit.Chem import AllChem

from utils.xtb import *

parser = ArgumentParser()
parser.add_argument('--in_confs', type=str, required=True, help='Pickle with input conformers')
parser.add_argument('--skip', type=int, default=1, help='Frequency for running procedure')
parser.add_argument('--out_confs', type=str, required=True, help='Path to output pickle')
parser.add_argument('--mmff', action='store_true', default=False, help='Whether to optimize with MMFF')
parser.add_argument('--level', type=str, default="normal", help='xTB optimization level')
parser.add_argument('--xtb_energy', action='store_true', default=False, help='Whether to comput xTB energies')
parser.add_argument('--xtb_path', type=str, default=None, help='Specifies local path to xTB installation')
parser.add_argument('--limit', type=int, default=32, help='Limit in the number of conformers')
args = parser.parse_args()

"""
    Takes as input a dictionary of generated conformers, performs MMFF or xTB relaxations 
    and computes the properties of each conformer
"""

test_data = pd.read_csv('data/DRUGS/test_smiles_corrected.csv').values
test_data = test_data[::args.skip]
print('Optimizing', len(test_data), 'mols')

mols = pickle.load(open(args.in_confs, 'rb'))
new_mols = {}
for _, _, smi in tqdm.tqdm(test_data):
    if smi not in mols:
        print('Model failure', smi)
        continue
    print(smi)
    confs = mols[smi][:args.limit]
    new_confs = []
    for conf in tqdm.tqdm(confs):
        if args.mmff: AllChem.MMFFOptimizeMoleculeConfs(conf, mmffVariant='MMFF94s')
        if args.xtb_path:
            if xtb_energy:
                success = xtb_optimize(conf, args.level, path_xtb=args.xtb_path)
                if not success: continue
            res = xtb_energy(conf, dipole=True, path_xtb=args.xtb_path)
            if not res: continue
            conf.xtb_energy, conf.xtb_dipole, conf.xtb_gap, conf.xtb_runtime = res['energy'], res['dipole'], res['gap'], res['runtime']
        new_confs.append(conf)
    new_mols[smi] = new_confs
open(args.out_confs, 'wb').write(pickle.dumps(new_mols))
            
            
            