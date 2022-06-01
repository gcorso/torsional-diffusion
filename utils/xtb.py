import os, subprocess
from rdkit import Geometry
from rdkit.Chem import rdmolfiles

my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    os.mkdir(my_dir)


def xtb_energy(mol, path_xtb, water=False, dipole=False):
    path = f"/tmp/{os.getpid()}.xyz"
    rdmolfiles.MolToXYZFile(mol, path)
    cmd = [path_xtb, path, '--iterations', str(1000)]
    if water:
        cmd += ['--alpb', 'water']
    if dipole:
        cmd += ['--dipole']
    n_tries = 3
    result = {}
    for i in range(n_tries):
        try:
            out = subprocess.check_output(cmd, stderr=open('/dev/null', 'w'), cwd=my_dir)
            break
        except subprocess.CalledProcessError as e:
            if i == n_tries-1:
                print('xtb_energy did not converge')
                return result #print(e.returncode, e.output)
    if dipole:
        dipole = [line for line in out.split(b'\n') if b'full' in line][1]
        result['dipole'] = float(dipole.split()[-1])
        
    runtime = out.split(b'\n')[-8].split()
    result['runtime'] = float(runtime[-2]) + 60*float(runtime[-4]) + 3600*float(runtime[-6]) + 86400*float(runtime[-8])                    
    
    energy = [line for line in out.split(b'\n') if b'TOTAL ENERGY' in line]
    result['energy'] = 627.509 * float(energy[0].split()[3])
    
    gap = [line for line in out.split(b'\n') if b'HOMO-LUMO GAP' in line]
    result['gap'] = 23.06 * float(gap[0].split()[3])
    
    return result
    
def xtb_optimize(mol, level, path_xtb):
    in_path = f'{my_dir}/xtb.xyz'
    out_path = f'{my_dir}/xtbopt.xyz'
    if os.path.exists(out_path): os.remove(out_path)
    try:
        rdmolfiles.MolToXYZFile(mol, in_path)
        cmd = [path_xtb, in_path, "--opt", level]
        out = subprocess.check_output(cmd, stderr=open('/dev/null', 'w'), cwd=my_dir)
        runtime = out.split(b'\n')[-12].split()
        runtime = float(runtime[-2]) + 60*float(runtime[-4]) + 3600*float(runtime[-6]) + 86400*float(runtime[-8])
        out = open(out_path).read().split('\n')[2:-1]
        coords = []
        for line in out:
            _, x, y, z = line.split()
            coords.append([float(x),float(y),float(z)])

        conf = mol.GetConformer()

        for i in range(mol.GetNumAtoms()):
            x,y,z = coords[i]
            conf.SetAtomPosition(i, Geometry.Point3D(x,y,z))
        return runtime
    except Exception as e:
        print(e)
        return None
    