import glob, os, pickle, random, tqdm
from collections import defaultdict
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment

from utils.standardization import *

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
parser.add_argument('--worker_id', type=int, required=True, help='Worker id to determine correct portion')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory for the pickles')
parser.add_argument('--jobs_per_worker', type=int, default=1000, help='Number of molecules for each worker')
parser.add_argument('--root', type=str, default='data/DRUGS/drugs/', help='Directory with molecules pickle files')
parser.add_argument('--popsize', type=int, default=15, help='Population size for differential evolution')
parser.add_argument('--max_iter', type=int, default=15, help='Maximum number of iterations for differential evolution')
parser.add_argument('--confs_per_mol', type=int, default=30, help='Maximum number of conformers to take for each molecule')
parser.add_argument('--mmff', action='store_true', default=False, help='Whether to relax seed conformers with MMFF before matching')
parser.add_argument('--no_match', action='store_true', default=False, help='Whether to skip conformer matching')
parser.add_argument('--boltzmann', choices=['top', 'resample'], default=None, help='If set, specifies a different conformer selection policy')
args = parser.parse_args()

"""
    Refers to the process of conformer matching to run before the start of training, takes the conformers from
    a subset of the pickle files in the root directory and saves a final pickle for all of the. Example script:
    
    for i in $(seq 0, 299); do
        python standardize_confs.py --out_dir data/DRUGS/standardized_pickles --root data/DRUGS/drugs/ --confs_per_mol 30 --worker_id $i --jobs_per_worker 1000 &
    done
"""

REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)


def sort_confs(confs):
    return sorted(confs, key=lambda conf: -conf['boltzmannweight'])


def resample_confs(confs, max_confs=None):
    weights = [conf['boltzmannweight'] for conf in confs]
    weights = np.array(weights) / sum(weights)
    k = min(max_confs, len(confs)) if max_confs else len(confs)
    return random.choices(confs, weights, k=k)


def log_error(err):
    print(err)
    long_term_log[err] += 1
    return None


def conformer_match(name, confs):
    long_term_log['confs_seen'] += len(confs)

    if args.boltzmann == 'top':
        confs = sort_confs(confs)

    limit = args.confs_per_mol if args.boltzmann != 'resample' else None
    confs = clean_confs(name, confs, limit=limit)
    if not confs: return log_error("no_clean_confs")

    if args.boltzmann == 'resample':
        confs = resample_confs(confs, max_confs=args.confs_per_mol)

    if args.confs_per_mol:
        confs = confs[:args.confs_per_mol]

    n_confs = len(confs)

    new_confs = []

    mol_rdkit = copy.deepcopy(confs[0]['rd_mol'])
    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds: return log_error("no_rotable_bonds")

    mol_rdkit.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)

    if mol_rdkit.GetNumConformers() != n_confs:
        return log_error("rdkit_no_embed")
    if args.mmff:
        try:
            mmff_func(mol_rdkit)
        except:
            return log_error("mmff_error")

    if not args.no_match:
        cost_matrix = [[get_von_mises_rms(confs[i]['rd_mol'], mol_rdkit, rotable_bonds, j) for j in range(n_confs)] for
                       i in range(n_confs)]
        cost_matrix = np.asarray(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        row_ind, col_ind = np.arange(len(confs)), np.arange(len(confs))

    iterable = tqdm.tqdm(enumerate(confs), total=len(confs))

    for i, conf in iterable:
        mol = conf['rd_mol']
        conf_id = int(col_ind[i])

        try:
            mol_rdkit_single = copy.deepcopy(mol_rdkit)
            [mol_rdkit_single.RemoveConformer(j) for j in range(n_confs) if j != conf_id]
            optimize_rotatable_bonds(mol_rdkit_single, mol, rotable_bonds,
                                     popsize=args.popsize, maxiter=args.max_iter)
            rmsd = AllChem.AlignMol(REMOVE_HS(mol_rdkit_single), REMOVE_HS(mol))
            long_term_log['confs_success'] += 1

        except Exception as e:
            print(e)
            long_term_log['confs_fail'] += 1
            continue

        conf['rd_mol'] = mol_rdkit_single
        conf['rmsd'] = rmsd
        conf['num_rotable_bonds'] = len(rotable_bonds)
        new_confs.append(conf)

        long_term_rmsd_cache.append(rmsd)
    return new_confs


root = args.root
files = sorted(glob.glob(f'{root}*.pickle'))
files = files[args.worker_id * args.jobs_per_worker:(args.worker_id + 1) * args.jobs_per_worker]
master_dict = {}
print(len(files), 'jobs')
long_term_rmsd_cache = []
long_term_log = defaultdict(int)

for i, f in enumerate(files):
    with open(f, "rb") as pkl:
        mol_dic = pickle.load(pkl)
    confs = mol_dic['conformers']
    name = mol_dic["smiles"]

    try:
        new_confs = conformer_match(name, confs)
    except Exception as e:
        print(e)
        long_term_log['mol_other_failure'] += 1
        new_confs = None

    if not new_confs:
        print(f'{i} Failure nconfs={len(confs)} smi={name}')
    else:
        num_rotable_bonds = new_confs[0]['num_rotable_bonds']
        rmsds = [conf['rmsd'] for conf in new_confs]
        print(
            f'{i} Success nconfs={len(new_confs)}/{len(confs)} bonds={num_rotable_bonds} rmsd={np.mean(rmsds):.2f} smi={name}')
        mol_dic['conformers'] = new_confs
        master_dict[f[len(root):-7]] = mol_dic

    if (i + 1) % 20 == 0:
        update = {
             'mols_processed': i + 1,
             'mols_success': len(master_dict),
             'mean_rmsd': np.mean(long_term_rmsd_cache)
        } | long_term_log
        print(update)
print('ALL RMSD', np.mean(long_term_rmsd_cache))
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
with open(args.out_dir + '/' + str(args.worker_id).zfill(3) + '.pickle', 'wb') as f:
    pickle.dump(master_dict, f)
