from argparse import ArgumentParser
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pickle
import pandas as pd
from tqdm import tqdm
import yaml
import os.path as osp

from utils.utils import get_model
from diffusion.sampling import *

parser = ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Path to folder with trained model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--out', type=str, help='Path to the output pickle file')
parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles.csv', help='Path to csv file with list of smiles and number conformers')
parser.add_argument('--pre_mmff', action='store_true', default=False, help='Whether to run MMFF on the local structure conformer')
parser.add_argument('--post_mmff', action='store_true', default=False, help='Whether to run MMFF on the final generated structures')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether avoid randomising the torsions of the seed conformer')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--seed_confs', default=None, help='Path to directly specify the seed conformers')
parser.add_argument('--seed_mols', default=None, help='Path to directly specify the seed molecules (instead of from SMILE)')
parser.add_argument('--single_conf', action='store_true', default=False, help='Whether to start from a single local structure')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--limit_mols', type=int, default=None, help='Limit to the number of molecules')
parser.add_argument('--confs_per_mol', type=int, default=None, help='If set for every molecule this number of conformers is generated, '
                                                                    'otherwise 2x the number in the csv file')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE instead of the SDE')
parser.add_argument('--likelihood', choices=['full', 'hutch'], default=None, help='Technique to compute likelihood')
parser.add_argument('--dump_pymol', type=str, default=None, help='Whether to save .pdb file with denoising dynamics')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--water', action='store_true', default=False, help='Whether to compute xTB energy in water')
parser.add_argument('--batch_size', type=int, default=32, help='Number of conformers generated in parallel')
parser.add_argument('--xtb', type=str, default=None, help='If set, it indicates path to local xtb main directory')
parser.add_argument('--no_energy', action='store_true', default=False, help='If set skips computation of likelihood, energy etc')
args = parser.parse_args()

"""
    Generates conformers for a list of molecules' SMILE given a trained model
    Saves a pickle with dictionary with the SMILE as key and the RDKit molecules with generated conformers as value 
"""

if args.likelihood:
    assert args.ode or args.no_model


def embed_func(mol, numConfs):
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=5)
    return mol


still_frames = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size

if args.seed_confs:
    print("Using local structures from", args.seed_confs)
    with open(args.seed_confs, 'rb') as f:
        seed_confs = pickle.load(f)
elif args.seed_mols:
    print("Using molecules from", args.seed_mols)
    with open(args.seed_mols, 'rb') as f:
        seed_confs = pickle.load(f)

with open(f'{args.model_dir}/model_parameters.yml') as f:
    args.__dict__.update(yaml.full_load(f))
args.batch_size = batch_size  # override the training one
if not args.no_model:
    model = get_model(args)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

test_data = pd.read_csv(args.test_csv).values
if args.limit_mols:
    test_data = test_data[:args.limit_mols]

conformer_dict = {}
if args.tqdm:
    test_data = tqdm(enumerate(test_data), total=len(test_data))
else:
    test_data = enumerate(test_data)


def sample_confs(raw_smi, n_confs, smi):
    print(raw_smi)
    if args.seed_confs:
        mol, data = get_seed(raw_smi, seed_confs=seed_confs, dataset=args.dataset)
    elif args.seed_mols:
        mol, data = get_seed(smi, seed_confs=seed_confs, dataset=args.dataset)
        mol.RemoveAllConformers()
    else:
        mol, data = get_seed(smi, dataset=args.dataset)
    if not mol:
        print('Failed to get seed', smi)
        return None

    n_rotable_bonds = int(data.edge_mask.sum())
    if args.seed_confs:
        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=args.single_conf, smi=raw_smi,
                                      pdb=args.dump_pymol, seed_confs=seed_confs)
    else:
        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=args.single_conf,
                                      pdb=args.dump_pymol, embed_func=embed_func, mmff=args.pre_mmff)
    if not conformers:
        print("Failed to embed", smi)
        return None

    if not args.no_random and n_rotable_bonds > 0.5:
        conformers = perturb_seeds(conformers, pdb)

    if not args.no_model and n_rotable_bonds > 0.5:
        conformers = sample(conformers, model, args.sigma_max, args.sigma_min, args.inference_steps,
                            args.batch_size, args.ode, args.likelihood, pdb)

    if args.dump_pymol:
        if not osp.isdir(args.dump_pymol):
            os.mkdir(args.dump_pymol)
        pdb.write(f'{args.dump_pymol}/{smi_idx}.pdb', limit_parts=5)

    mols = [pyg_to_mol(mol, conf, args.post_mmff, rmsd=not args.no_energy) for conf in conformers]
    if args.likelihood:
        if n_rotable_bonds < 0.5:
            print(f"Skipping mol {smi} with 0 rotable bonds")
            return None
    for mol, data in zip(mols, conformers):
        populate_likelihood(mol, data, water=args.water, xtb=args.xtb)

    if args.xtb:
        mols = [mol for mol in mols if mol.xtb_energy]
    return mols


for smi_idx, (raw_smi, n_confs, smi) in test_data:
    if type(args.confs_per_mol) is int:
        mols = sample_confs(raw_smi, args.confs_per_mol, smi)
    else:
        mols = sample_confs(raw_smi, 2 * n_confs, smi)
    if not mols: continue
    if not args.no_energy:
        rmsd = [mol.rmsd for mol in mols]
        dlogp = np.array([mol.euclidean_dlogp for mol in mols])
        if args.xtb:
            energy = np.array([mol.xtb_energy for mol in mols])
        else:
            energy = np.array([mol.mmff_energy for mol in mols])
        F, F_std = (0, 0) if args.no_energy else free_energy(dlogp, energy)
        print(
            f'{smi_idx} rotable_bonds={mols[0].n_rotable_bonds} n_confs={len(rmsd)}',
            f'rmsd={np.mean(rmsd):.2f}',
            f'F={F:.2f}+/-{F_std:.2f}',
            f'energy {np.mean(energy):.2f}+/-{bootstrap((energy,), np.mean).standard_error:.2f}',
            f'dlogp {np.mean(dlogp):.2f}+/-{bootstrap((dlogp,), np.mean).standard_error:.2f}',
            smi,
            flush=True
        )
    else:
        print(f'{smi_idx} rotable_bonds={mols[0].n_rotable_bonds} n_confs={len(mols)}', smi, flush=True)
    conformer_dict[smi] = mols

# save to file
if args.out:
    with open(f'{args.out}', 'wb') as f:
        pickle.dump(conformer_dict, f)
print('Generated conformers for', len(conformer_dict), 'molecules')
