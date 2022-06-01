import os.path
from multiprocessing import Pool

from rdkit import Chem
import numpy as np
import glob, pickle, random
import os.path as osp
import torch, tqdm
import copy
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.transforms import BaseTransform
from collections import defaultdict

from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, drugs_types
from utils.torsion import get_transformation_mask, modify_conformer


class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi, boltzmann_weight=False):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.boltzmann_weight = boltzmann_weight

    def __call__(self, data):
        # select conformer
        if self.boltzmann_weight:
            data.pos = random.choices(data.pos, data.weights, k=1)[0]
        else:
            data.pos = random.choice(data.pos)

        try:
            edge_mask, mask_rotate = data.edge_mask, data.mask_rotate
        except:
            edge_mask, mask_rotate = data.mask_edges, data.mask_rotate
            data.edge_mask = torch.tensor(data.mask_edges)

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = sigma * torch.ones(data.num_nodes)

        torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
        data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates)
        data.edge_rotate = torch.tensor(torsion_updates)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')


class ConformerDataset(Dataset):
    def __init__(self, root, split_path, mode, types, dataset, transform=None, num_workers=1, limit_molecules=None,
                 cache=None, pickle_dir=None, boltzmann_resampler=None):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol

        super(ConformerDataset, self).__init__(root, transform)
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.boltzmann_resampler = boltzmann_resampler

        if cache: cache += "." + mode
        self.cache = cache
        if cache and os.path.exists(cache):
            print('Reusing preprocessing from cache', cache)
            with open(cache, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            self.datapoints = self.preprocess_datapoints(root, split_path, pickle_dir, mode, num_workers, limit_molecules)
            if cache:
                print("Caching at", cache)
                with open(cache, "wb") as f:
                    pickle.dump(self.datapoints, f)

        if limit_molecules:
            self.datapoints = self.datapoints[:limit_molecules]


    def preprocess_datapoints(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]

        self.open_pickles = {}
        if pickle_dir:
            smiles = [(i // mols_per_pickle, smi[len(root):-7]) for i, smi in zip(split, smiles)]
            if limit_molecules:
                smiles = smiles[:limit_molecules]
            self.current_pickle = (None, None)
            self.pickle_dir = pickle_dir
        else:
            smiles = [smi[len(root):-7] for smi in smiles]

        print('Preparing to process', len(smiles), 'smiles')
        datapoints = []
        if num_workers > 1:
            p = Pool(num_workers)
            p.__enter__()
        with tqdm.tqdm(total=len(smiles)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for t in map_fn(self.filter_smiles, smiles):
                if t:
                    datapoints.append(t)
                pbar.update()
        if num_workers > 1: p.__exit__(None, None, None)
        print('Fetched', len(datapoints), 'mols successfully')
        print(self.failures)
        if pickle_dir: del self.current_pickle
        return datapoints

    def filter_smiles(self, smile):

        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    return False
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                return False
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
                self.failures['raw_pickle_not_found'] += 1
                return False
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)

        smile = mol_dic['smiles']

        if '.' in smile:
            self.failures['dot_in_smile'] += 1
            return False

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            return False

        mol = mol_dic['conformers'][0]['rd_mol']
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            return False

        if N < 4:
            self.failures['mol_too_small'] += 1
            return False

        data = self.featurize_mol(mol_dic)
        if not data:
            self.failures['featurize_mol_failed'] += 1
            return False

        edge_mask, mask_rotate = get_transformation_mask(data)
        if np.sum(edge_mask) < 0.5:
            self.failures['no_rotable_bonds'] += 1
            return False

        data.edge_mask = torch.tensor(edge_mask)
        data.mask_rotate = mask_rotate
        return data

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        if self.boltzmann_resampler:
            self.boltzmann_resampler.try_resample(data)
        return copy.deepcopy(data)

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        pos = []
        weights = []
        for conf in confs:
            mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            weights.append(conf['boltzmannweight'])
            correct_mol = mol

            if self.boltzmann_resampler is not None:
                # torsional Boltzmann generator uses only the local structure of the first conformer
                break

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            return None

        data = featurize_mol(correct_mol, self.types)
        normalized_weights = list(np.array(weights) / np.sum(weights))
        if np.isnan(normalized_weights).sum() != 0:
            print(name, len(confs), len(pos), weights)
            normalized_weights = [1 / len(weights)] * len(weights)
        data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights

        return data

    def resample_all(self, resampler, temperature=None):
        ess = []
        for data in tqdm.tqdm(self.datapoints):
            ess.append(resampler.resample(data, temperature=temperature))
        return ess


def construct_loader(args, modes=('train', 'val'), boltzmann_resampler=None):
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    transform = TorsionNoiseTransform(sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                      boltzmann_weight=args.boltzmann_weight)
    types = qm9_types if args.dataset == 'qm9' else drugs_types

    for mode in modes:
        dataset = ConformerDataset(args.data_dir, args.split_path, mode, dataset=args.dataset,
                                   types=types, transform=transform,
                                   num_workers=args.num_workers,
                                   limit_molecules=args.limit_train_mols,
                                   cache=args.cache,
                                   pickle_dir=args.std_pickles,
                                   boltzmann_resampler=boltzmann_resampler)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False if mode == 'test' else True)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
