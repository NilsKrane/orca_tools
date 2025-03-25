import numpy as np
import math
import os
from scipy.signal import convolve2d
from .orca_molecule import Molecule
from .cube_utils import Cube

def huang_rhys(ground: Molecule, excited: Molecule, warning=True) -> list:
    # S = omega / 2 hbar * mwdiff**2
    mwdiff = (excited.coords_masscentered-ground.coords_masscentered)*np.sqrt(excited.mass)

    # fac = (1E-10 m/Angstrom)^2 * 1.66E-27 kg/u / hbar(in eV) / hbar(in SI)
    fac = 1e-20*1.660539e-27/6.582119e-16/1.054571e-34/2
    if type(excited.mwnm) != type(None): 
        proj = np.sum(mwdiff[:,:,None]*excited.mwnm,axis=(1,0))
        S = fac*excited.freqs*proj**2
    elif type(ground.mwnm) != type(None):
        if warning:
            print('Warning: Vibrational normal modes of excited state molecule not defined. Will use normal modes of ground state assuming negligible Duschinsky rotation.')
        proj = np.sum(mwdiff[:,:,None]*ground.mwnm,axis=(1,0))
        S = fac*ground.freqs*proj**2
    else:
        print('Error: Neither molecule object provides vibrational normal modes.')
        S = None
    return S
    
def parab_shift(ground: Molecule, excited: Molecule, warning=True) -> list:
    # output in sqrt(u)*Angstrom. To switch into units of delta_Q use multiplicator:
    # fac = np.sqrt(2)*1e-10*np.sqrt(1.660539e-27)*np.sqrt(1.602176e-19)/1.054571e-34 * np.sqrt*(MODE_ENERGY_IN_eV)
    # with delta_Q being half the dislocation of the classical turning point

    mwdiff = (excited.coords_masscentered-ground.coords_masscentered)*np.sqrt(excited.mass)
    if type(excited.mwnm) != type(None): 
        proj = np.sum(mwdiff[:,:,None]*excited.mwnm,axis=(1,0))
    elif type(ground.mwnm) != type(None):
        if warning:
            print('Warning: Vibrational normal modes of excited state molecule not defined. Will use normal modes of ground state assuming negligible Duschinsky rotation.')
        proj = np.sum(mwdiff[:,:,None]*ground.mwnm,axis=(1,0))
    else:
        print('Error: Neither molecule object provides vibrational normal modes.')
        proj = None
    return proj

def relaxation_energy(ground: Molecule, excited: Molecule, warning=True) -> float:
    if type(excited.freqs) != type(None): 
        return np.sum(huang_rhys(ground,excited, warning=warning)*excited.freqs)
    elif type(ground.freqs) != type(None):
        return np.sum(huang_rhys(ground,excited, warning=warning)*ground.freqs)
    else:
        print('Error: Neither molecule object provides vibrational normal modes.')

        
### ----------------------------------------------------------------------------

def FC_max_n(S: float, threshold=1e-2) -> int:
    def poisson(S,n):
        return np.exp(-S)*S**n/math.factorial(n)
    amax = poisson(S,0)
    n=1
    a = poisson(S,1)
    while amax * threshold < a:
        n += 1 
        a = poisson(S,n)
        amax = max(amax,a)
    return n

def FC_spectrum_mode(S: float, hw: float, dx=0.25e-3, threshold=1e-3) -> np.array:
    n = FC_max_n(S,threshold)
    length = int(np.round((n-1)*hw/dx)+1)
    a = np.zeros((length))
    for i in range(n):
        a[int(np.round(i*hw/dx))]=np.exp(-S)*S**i/math.factorial(i)
    return a

def FC_spectrum(freqs: np.array, HR: np.array, dx=0.25e-3, threshold=1e-3) -> np.array:
    assert freqs.shape == HR.shape, "Freqs and Huang-Rhys factors do not match in shape"
    spec = np.ones((1,))
    for S,hw in zip(HR,freqs):
        if S > threshold and hw > 0:
            spec = np.convolve(spec,FC_spectrum_mode(S,hw,dx,threshold))
    spec = crop_spec(spec,threshold)
    return spec, np.linspace(0,(len(spec)-1)*dx,len(spec))

def crop_spec(spec: np.array, threshold: float) -> np.array:
    return np.resize(spec, (np.where(spec > np.max(spec)*threshold)[0][-1]+1,))

### ----------------------------------------------------------------------------

def IETS_map(molecule: Molecule, MO: int, mode: int, amplitude=0.5,z_range=(1.5,), mass_centered = True, cleanup=True, **kwargs):
    if mode < 0:
        cube = Cube(molecule.make_cube(MO,coords=molecule.dislocated_coords(mode,0, mass_centered,**kwargs), z_range=z_range, cleanup=cleanup, **kwargs))
    else:
        cube = Cube(molecule.make_cube(MO,coords=molecule.dislocated_coords(mode,amplitude,mass_centered,**kwargs), z_range=z_range, cleanup=cleanup,**kwargs))
        cube_n = Cube(molecule.make_cube(MO,coords=molecule.dislocated_coords(mode,-amplitude,mass_centered,**kwargs), z_range=z_range, cleanup=cleanup, **kwargs))
        cube.data -= cube_n.data
    if cleanup:
        os.remove(cube.filename)
    return cube

def IETS_map_stack(molecule: Molecule, MO: int, spin=0, z_range=(1.5,), maxmode=None, update=0, **kwargs) -> str:
    cubes = [Cube(molecule.make_cube(MO=MO,spin=spin,z_range=z_range,**kwargs))]
    if maxmode is None: maxmode = len(molecule.freqs)
    if update:
        print(f"Calculating {maxmode} modes: ",end="")
    for mode in range(maxmode):
        if update and mode % update == 0:
            print(f"{mode:03d}",end="..")
        cubes.append(IETS_map(molecule,MO,mode,spin=spin,z_range=z_range,**kwargs))
    if update:
        print("done!")
    freqs = np.insert(molecule.freqs[:maxmode], 0, 0)
    return cubes, freqs


### ----------------------------------------------------------------------------

def lorentzian(gamma: float, dx=0.25e-3, threshold=1e-3) -> np.array:
    offset = np.sqrt(abs(gamma**2/threshold))
    lorx = np.linspace(-offset,offset,int(2*offset/dx)+1)
    return gamma/np.pi/(lorx**2+gamma**2), offset, lorx


def convolve_spec(spec: np.array, specx: np.array, lineshape: np.array, offset: float, normalize=True) -> np.array:
    conv = np.convolve(spec,lineshape)
    conv_x = np.linspace(-offset+specx[0],specx[-1]+offset,len(lineshape)+len(spec)-1)
    if normalize:
        conv *= 1/np.max(conv)
    return conv_x, conv


### ----------------------------------------------------------------------------

def rotational_normalmodes(coords,mass):
    center_of_mass = np.sum(coords*mass[None,:], axis=1)/np.sum(mass)
    coords -= center_of_mass[:,None]

    def inertia_tensor(coords,mass):
        x,y,z = coords
        Ixx = np.sum(mass * (y**2 + z**2))
        Iyy = np.sum(mass * (x**2 + z**2))
        Izz = np.sum(mass * (x**2 + y**2))
        Ixy = -np.sum(mass * x * y)
        Iyz = -np.sum(mass * y * z)
        Ixz = -np.sum(mass * x * z)
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    inertia, X = np.linalg.eig(inertia_tensor(coords,mass))    
    
    P = np.matmul(X,coords)
    R = np.zeros((3,len(mass),3))
    R[:,:,0] = (P[1,:][None,:]*X[:,2][:,None]-P[2,:][None,:]*X[:,1][:,None])*np.sqrt(mass[None,:])
    R[:,:,1] = (P[2,:][None,:]*X[:,0][:,None]-P[0,:][None,:]*X[:,2][:,None])*np.sqrt(mass[None,:])
    R[:,:,2] = (P[0,:][None,:]*X[:,1][:,None]-P[1,:][None,:]*X[:,0][:,None])*np.sqrt(mass[None,:])
    R = R.reshape((-1,3),order='F')
    return R / np.linalg.norm(R, axis=0, keepdims=True)


def translational_normalmodes(mass):
    T = np.concatenate([np.eye(3)]*len(mass),axis=0)*np.sqrt(np.repeat(mass,3))[:,None]
    return T / np.linalg.norm(T, axis=0, keepdims=True)

def gram_schmidt(vectors):
    basis = [vectors[:,0]]
    for v in vectors[:,1:].T:
        w = v - np.sum(np.array([np.dot(v,b)*b  for b in basis]),axis=0)
        if np.linalg.norm(w) > 1e-10:  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis).T


def hessian2normalmodes(hessian, coords, mass, cm=False, mass_weighted=False):
    # generate orthogonal space where translational and rotational vectors are seperated out
    T = translational_normalmodes(mass)
    R = rotational_normalmodes(coords,mass)
    D = gram_schmidt(np.concatenate((T,R,np.eye(9)),axis=1))


    # translate hessian to mass-weighted internal coordinates and diagonalize
    mass_weighted_hessian = hessian/np.sqrt(np.outer(np.repeat(mass,3),np.repeat(mass,3)))
    hess_int = D.T @ mass_weighted_hessian @ D
    lambd, L = np.linalg.eig(hess_int)
    
    # resort to increasing order of energies
    sorted_indices = np.argsort(lambd)
    lambd = lambd[sorted_indices] # square of normal mode energies
    L = L[:,sorted_indices] # normal modes in mass-weighted internal coordinates
    
    # translate eigenvalues to energies
    fac = 4.3597482e-18/(4*np.pi**2)/(0.529177249e-10)**2/1.660539068e-27 # sqrt(Hartree/u)/Bohr to eV
    freqs = np.sign(lambd)*np.sqrt(abs(lambd)*fac)*4.135667e-15
    if cm == True:
        freqs *= 8065.56    

    # translate eigenvectors to non-mass-weighted cartesian normal-modes and renorm
    M = np.sqrt(np.diag(1/np.repeat(mass,3)))
    normal_modes = M @ D @ L
    normal_modes *= 1/np.linalg.norm(normal_modes, axis=0, keepdims=True)
    
    # reshape normalmodes and remove translational and vibrational modes
    normal_modes = np.reshape(normal_modes, (3, len(freqs) // 3, len(freqs)),order="F")
    freqs = freqs[6:]
    normal_modes = normal_modes[:,:,6:]
    
    if mass_weighted:
        normal_modes *= np.sqrt(mass[None,:,None])
        normal_modes *= 1/np.sqrt(np.sum(normal_modes**2,axis=(0,1))) # renorm
        
    return freqs, normal_modes
    