import random
from utils.featurization import featurize_mol, featurize_mol_from_smiles
from utils.torsion import *
from diffusion.likelihood import *
import torch, copy
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

from utils.utils import time_limit, TimeoutException
from utils.visualise import PDBFile
from spyrmsd import molecule, graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
still_frames = 10


def try_mmff(mol):
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        return True
    except Exception as e:
        return False


def get_seed(smi, seed_confs=None, dataset='drugs'):
    if seed_confs:
        if smi not in seed_confs:
            print("smile not in seeds", smi)
            return None, None
        mol = seed_confs[smi][0]
        data = featurize_mol(mol, dataset)

    else:
        mol, data = featurize_mol_from_smiles(smi, dataset=dataset)
        if not mol:
            return None, None
    data.edge_mask, data.mask_rotate = get_transformation_mask(data)
    data.edge_mask = torch.tensor(data.edge_mask)
    return mol, data


def embed_seeds(mol, data, n_confs, single_conf=False, smi=None, embed_func=None, seed_confs=None, pdb=None, mmff=False):
    if not seed_confs:
        embed_num_confs = n_confs if not single_conf else 1
        try:
            mol = embed_func(mol, embed_num_confs)
        except Exception as e:
            print(e.output)
            pass
        if len(mol.GetConformers()) != embed_num_confs:
            print(len(mol.GetConformers()), '!=', embed_num_confs)
            return [], None
        if mmff: try_mmff(mol)

    if pdb: pdb = PDBFile(mol)
    conformers = []
    for i in range(n_confs):
        data_conf = copy.deepcopy(data)
        if single_conf:
            seed_mol = copy.deepcopy(mol)
        elif seed_confs:
            seed_mol = random.choice(seed_confs[smi])
        else:
            seed_mol = copy.deepcopy(mol)
            [seed_mol.RemoveConformer(j) for j in range(n_confs) if j != i]

        data_conf.pos = torch.from_numpy(seed_mol.GetConformers()[0].GetPositions()).float()
        data_conf.seed_mol = copy.deepcopy(seed_mol)
        if pdb:
            pdb.add(data_conf.pos, part=i, order=0, repeat=still_frames)
            if seed_confs:
                pdb.add(data_conf.pos, part=i, order=-2, repeat=still_frames)
            pdb.add(torch.zeros_like(data_conf.pos), part=i, order=-1)

        conformers.append(data_conf)
    if mol.GetNumConformers() > 1:
        [mol.RemoveConformer(j) for j in range(n_confs) if j != 0]
    return conformers, pdb


def perturb_seeds(data, pdb=None):
    for i, data_conf in enumerate(data):
        torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
        data_conf.pos = modify_conformer(data_conf.pos, data_conf.edge_index.T[data_conf.edge_mask],
                                         data_conf.mask_rotate, torsion_updates)
        data_conf.total_perturb = torsion_updates
        if pdb:
            pdb.add(data_conf.pos, part=i, order=1, repeat=still_frames)
    return data


def sample(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None, pg_weight_log_0=None, pg_repulsive_weight_log_0=None,
           pg_weight_log_1=None, pg_repulsive_weight_log_1=None, pg_kernel_size_log_0=None,
           pg_kernel_size_log_1=None, pg_langevin_weight_log_0=None, pg_langevin_weight_log_1=None,
           pg_invariant=False, mol=None):

    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    if pg_weight_log_0 is not None and pg_weight_log_1 is not None:
        edge_index, edge_mask = conformers[0].edge_index, conformers[0].edge_mask
        edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

        for p in edge_index.T:
            edge_list[p[0]].append(p[1])

        rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

        dihedral = []
        for a, b in rot_bonds:
            c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
            d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
            dihedral.append((c.item(), a.item(), b.item(), d.item()))
        dihedral_numpy = np.asarray(dihedral)
        dihedral = torch.tensor(dihedral)

        if pg_invariant:
            try:
                with time_limit(10):
                    mol = molecule.Molecule.from_rdkit(mol)

                    aprops = mol.atomicnums
                    am = mol.adjacency_matrix

                    # Convert molecules to graphs
                    G = graph.graph_from_adjacency_matrix(am, aprops)

                    # Get all the possible graph isomorphisms
                    isomorphisms = graph.match_graphs(G, G)
                    isomorphisms = [iso[0] for iso in isomorphisms]
                    isomorphisms = np.asarray(isomorphisms)

                    # filter out those having an effect on the dihedrals
                    dih_iso = isomorphisms[:, dihedral_numpy]
                    dih_iso = np.unique(dih_iso, axis=0)

                    if len(dih_iso) > 32:
                        print("reduce isomorphisms from", len(dih_iso), "to", 32)
                        dih_iso = dih_iso[np.random.choice(len(dih_iso), replace=False, size=32)]
                    else:
                        print("isomorphisms", len(dih_iso))
                    dih_iso = torch.from_numpy(dih_iso).to(device)

            except TimeoutException as e:
                print("Timeout generating with non invariant kernel")
                pg_invariant = False

    for batch_idx, data in enumerate(loader):

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):

            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()

            t = sigma_idx / steps   # t is really 1-t
            pg_weight = 10**(pg_weight_log_0 * t + pg_weight_log_1 * (1 - t)) if pg_weight_log_0 is not None and pg_weight_log_1 is not None else 0.0
            pg_repulsive_weight = 10**(pg_repulsive_weight_log_0 * t + pg_repulsive_weight_log_1 * (1 - t)) if pg_repulsive_weight_log_0 is not None and pg_repulsive_weight_log_1 is not None else 1.0

            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood)
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            if pg_weight > 0:
                n = data.num_graphs
                if pg_invariant:
                    S, D, _ = dih_iso.shape
                    dih_iso_cat = dih_iso.reshape(-1, 4)
                    tau = get_torsion_angles(dih_iso_cat, data_gpu.pos, n)
                    tau_diff = tau.unsqueeze(1) - tau.unsqueeze(0)
                    tau_diff = torch.fmod(tau_diff + 3 * np.pi, 2 * np.pi) - np.pi
                    tau_diff = tau_diff.reshape(n, n, S, D)
                    tau_matrix = torch.sum(tau_diff ** 2, dim=-1, keepdim=True)
                    tau_matrix, indices = torch.min(tau_matrix, dim=2)
                    tau_diff = torch.gather(tau_diff, 2, indices.unsqueeze(-1).repeat(1, 1, 1, D)).squeeze(2)
                else:
                    tau = get_torsion_angles(dihedral, data_gpu.pos, n)
                    tau_diff = tau.unsqueeze(1) - tau.unsqueeze(0)
                    tau_diff = torch.fmod(tau_diff+3*np.pi, 2*np.pi)-np.pi
                    assert torch.all(tau_diff < np.pi + 0.1) and torch.all(tau_diff > -np.pi - 0.1), tau_diff
                    tau_matrix = torch.sum(tau_diff**2, dim=-1, keepdim=True)

                kernel_size = 10 ** (pg_kernel_size_log_0 * t + pg_kernel_size_log_1 * (1 - t)) if pg_kernel_size_log_0 is not None and pg_kernel_size_log_1 is not None else 1.0
                langevin_weight = 10 ** (pg_langevin_weight_log_0 * t + pg_langevin_weight_log_1 * (1 - t)) if pg_langevin_weight_log_0 is not None and pg_langevin_weight_log_1 is not None else 1.0

                k = torch.exp(-1 / kernel_size * tau_matrix)
                repulsive = torch.sum(2/kernel_size*tau_diff*k, dim=1).cpu().reshape(-1) / n

                perturb = (0.5 * g ** 2 * eps * score) + langevin_weight * (0.5 * g ** 2 * eps * score + g * np.sqrt(eps) * z)
                perturb += pg_weight * (g ** 2 * eps * (score + pg_repulsive_weight * repulsive))

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)

            if pdb:
                for conf_idx in range(data.num_graphs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()

    return conformers


def pyg_to_mol(mol, data, mmff=False, rmsd=True, copy=True):
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)
    coords = data.pos
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except Exception as e:
            pass
    try:
        if rmsd:
            mol.rmsd = AllChem.GetBestRMS(
                Chem.RemoveHs(data.seed_mol),
                Chem.RemoveHs(mol)
            )
        mol.total_perturb = data.total_perturb
    except:
        pass
    mol.n_rotable_bonds = data.edge_mask.sum()
    if not copy: return mol
    return deepcopy(mol)


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        for i, d in enumerate(data_list):
            d.idx = i
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def apply_torsion_and_update_pos(self, data, torsion_updates):

        pos_new, torsion_updates = perturb_batch(data, torsion_updates, split=True, return_updates=True)
        for i, idx in enumerate(data.idx):
            try:
                self.data[idx].total_perturb += torsion_updates[i]
            except:
                pass
            self.data[idx].pos = pos_new[i]
