import numpy as np
import torch
from rdkit.Chem import AllChem
from scipy.stats import bootstrap

from utils.torsion import perturb_batch
from utils.xtb import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def divergence(model, data, data_gpu, method):
    return {
        'full': divergence_full,
        'hutch': divergence_hutch
    }[method](model, data, data_gpu)


def mmff_energy(mol):
    energy = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')).CalcEnergy()
    return energy


def divergence_full(model, data, data_gpu, eps=0.01):
    score = data_gpu.edge_pred.cpu().numpy()
    if type(data.mask_rotate) is list:
        n_confs = len(data.mask_rotate)
    else:
        n_confs = 1
    n_bonds = score.shape[0] // n_confs
    div = 0
    for i in range(n_bonds):
        perturb = np.zeros_like(score)
        perturb[i::n_bonds] = eps
        data_gpu.pos = perturb_batch(data, perturb).to(device)
        with torch.no_grad():
            data_gpu = model(data_gpu)
        div += (data_gpu.edge_pred[i::n_bonds].cpu().numpy() - score[i::n_bonds]) / eps
    return div


def divergence_hutch(model, data, data_gpu, eps=0.001):
    score = data_gpu.edge_pred.cpu().numpy()
    if type(data.mask_rotate) is list:
        n_confs = len(data.mask_rotate)
    else:
        n_confs = 1
    n_bonds = score.shape[0] // n_confs
    perturb = 2 * eps * (np.random.randint(0, 2, score.shape[0]) - 0.5)
    data_gpu.pos = perturb_batch(data, perturb).to(device)
    with torch.no_grad():
        data_gpu = model(data_gpu)
    diff = (data_gpu.edge_pred.cpu().numpy() - score)
    div = [d @ p for d, p in zip(diff.reshape(n_confs, n_bonds), perturb.reshape(n_confs, n_bonds))]
    div = np.array(div) / eps ** 2
    return div


def inertia_tensor(pos):  # n, 3
    if type(pos) != np.ndarray:
        pos = pos.numpy()
    pos = pos - pos.mean(0, keepdims=True)
    n = pos.shape[0]
    I = (pos ** 2).sum() * np.eye(3) - (pos.reshape(n, 1, 3) * pos.reshape(n, 3, 1)).sum(0)
    return I


def dx_dtau(pos, edge, mask):
    u, v = pos[edge]
    bond = u - v
    bond = bond / np.linalg.norm(bond)
    u_side, v_side = pos[~mask] - u, pos[mask] - u
    u_side, v_side = np.cross(u_side, bond), np.cross(v_side, bond)
    return u_side, v_side


def log_det_jac(data):
    pos = data.pos
    if type(data.pos) != np.ndarray:
        pos = pos.numpy()

    pos = pos - pos.mean(0, keepdims=True)
    I = inertia_tensor(pos)
    jac = []
    for edge, mask in zip(data.edge_index.T[data.edge_mask], data.mask_rotate):
        dx_u, dx_v = dx_dtau(pos, edge, mask)
        dx = np.zeros_like(pos)
        dx[~mask] = dx_u
        dx = dx - dx.mean(0, keepdims=True)
        L = np.cross(pos, dx).sum(0)
        omega = np.linalg.inv(I) @ L
        dx = dx - np.cross(omega, pos)
        jac.append(dx.flatten())
    jac = np.array(jac)
    _, D, _ = np.linalg.svd(jac)
    return np.sum(np.log(D))


kT = 0.592
def free_energy(dlogp, energy, bootstrap_=True):
    def _F(arr):
        arr_max = np.max(arr)
        return -kT * (arr_max + np.log(np.exp(arr - arr_max).mean()))

    arr = -energy / kT - dlogp
    F = _F(arr)
    if not bootstrap_: return F
    F_std = bootstrap((arr,), _F, vectorized=False).standard_error
    return F, F_std


def populate_likelihood(mol, data, water=False, xtb=None):
    try:
        mol.dlogp = data.dlogp
    except:
        mol.dlogp = 0
    mol.inertia_tensor = inertia_tensor(data.pos)
    mol.log_det_jac = log_det_jac(data)
    mol.euclidean_dlogp = mol.dlogp - 0.5 * np.log(np.abs(np.linalg.det(mol.inertia_tensor))) - mol.log_det_jac
    mol.mmff_energy = mmff_energy(mol)
    if not xtb: return
    res = xtb_energy(mol, dipole=True, path_xtb=xtb)
    if res:
        mol.xtb_energy, mol.xtb_dipole, mol.xtb_gap, mol.xtb_runtime = res['energy'], res['dipole'], res['gap'], res['runtime']
    else:
        mol.xtb_energy = None
    if water:
        mol.xtb_energy_water = xtb_energy(mol, water=True, path_xtb=xtb)['energy']
