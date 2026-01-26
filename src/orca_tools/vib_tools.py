import numpy as np
import math
from .orca_molecule import Molecule
from .orca_utils import bohr2ang, atomic_mass, hartree, hbar, q_e

def huang_rhys(initial_state: Molecule, final_state: Molecule,
               freqs: np.ndarray = None ,mwnm: np.ndarray = None) -> np.ndarray:
    '''Calculate Huang-Rhys factors between initial state and final state molecule.

    :param initial_state: Initial structure of molecule.
    :type initial_state: Molecule
    :param final_state: Final structure of molecule.
    :type final_state: Molecule
    :param freqs: Frequencies to be used for projection. If None, the frequencies of the final state will be used. Defaults to None.
    :type freqs: np.ndarray, optional
    :param mwnm: Mass-weighted normal modes to be used for projection. If None, the normal modes of the final state will be used. Defaults to None.
    :type mwnm: np.ndarray, optional
    :return: Array of Huang-Rhys factors.
    :rtype: np.ndarray
    
    By default, the normal modes of the final state will be used to obtain the Huang-Rhys factors.
    If this is not possible, the normal modes of the initial state will be used.
    '''    

    # S = omega / 2 hbar * mwdiff**2
    mwdiff = (final_state.coords_masscentered-initial_state.coords_masscentered)*np.sqrt(final_state.mass[:,None])

    # fac = (1E-10 m/Angstrom)^2 * 1.66E-27 kg/u / hbar(in eV) / hbar(in SI)
    fac = 1e-20*atomic_mass*q_e/hbar**2/2
    if type(mwnm) != type(None) and type(freqs) != type(None):
        proj = np.sum(mwdiff[None,:,:]*mwnm,axis=(1,2))
        S = fac*freqs*proj**2
    elif type(final_state.mwnm) != type(None): 
        proj = np.sum(mwdiff[None,:,:]*final_state.mwnm,axis=(1,2))
        S = fac*final_state.freqs*proj**2
    elif type(initial_state.mwnm) != type(None):
        proj = np.sum(mwdiff[None,:,:]*initial_state.mwnm,axis=(1,2))
        S = fac*initial_state.freqs*proj**2
    else:
        return None

    return S
    
def parab_shift(initial_state: Molecule, final_state: Molecule,
                freqs: np.ndarray = None ,mwnm: np.ndarray = None,
                renorm_to_hwhm: bool=False) -> np.ndarray:
    '''Calculate shift of relaxed structures between initial and final state in mass-weighted normal coordinates.

    :param initial_state: Initial structure of molecule.
    :type initial_state: Molecule
    :param final_state: Final structure of molecule.
    :type final_state: Molecule
    :param freqs: Frequencies to be used for projection. If None, the frequencies of the final state will be used. Defaults to None.
    :type freqs: np.ndarray, optional
    :param mwnm: Mass-weighted normal modes to be used for projection. If None, the normal modes of the final state will be used. Defaults to None.
    :type mwnm: np.ndarray, optional
    :param renorm_to_hwhm: Shift will be renormalized to HWHM of vibrational ground state gaussian distribution, with a renormalized value of 2 corresponding to the classical turning point. Defaults to False.
    :type renorm_to_hwhm: bool, optional
    :return: 1d array containing shift for each normal mode in units of sqrt(atomic_mass)*Angstrom, if `renorm_to_hwhm = False`.
    :rtype: np.ndarray

    By default, the normal modes of the final state will be used to obtain the Huang-Rhys factors.
    If this is not possible, the normal modes of the initial state will be used.
    '''    

    mwdiff = (final_state.coords_masscentered-initial_state.coords_masscentered)*np.sqrt(final_state.mass[:,None])
    if type(mwnm) != type(None) and type(freqs) != type(None):
        proj = np.sum(mwdiff[None,:,:]*mwnm,axis=(1,2))
    elif type(final_state.mwnm) != type(None): 
        proj = np.sum(mwdiff[None,:,:]*final_state.mwnm,axis=(1,2))
        freqs = final_state.freqs
    elif type(initial_state.mwnm) != type(None):
        proj = np.sum(mwdiff[None,:,:]*initial_state.mwnm,axis=(1,2))
        freqs = initial_state.freqs
    else:
        return None
    if renorm_to_hwhm:
        proj *= np.sqrt(2*atomic_mass*q_e*freqs)/hbar*1e-10
    return proj


def relaxation_energy(initial_state: Molecule, final_state: Molecule) -> float:
    '''Calculate total vibrational relaxation energy between ground and final state.

    :param initial_state: Initial structure of molecule.
    :type initial_state: Molecule
    :param final_state: Final structure of molecule.
    :type fina_statel: Molecule
    :return: Total relaxation energy in eV
    :rtype: float
    
    By default, the normal modes of the final state will be used to obtain the Huang-Rhys factors.
    If this is not possible, the normal modes of the initial state will be used.
    '''    
    if type(final_state.freqs) != type(None): 
        return np.sum(huang_rhys(initial_state,final_state)*final_state.freqs)
    elif type(initial_state.freqs) != type(None):
        return np.sum(huang_rhys(initial_state,final_state)*initial_state.freqs)
    else:
        return None

        
### ----------------------------------------------------------------------------

def FC_max_n(S: float, threshold: float=1e-2) -> int:
    '''Determine cut-off for vibrational level for given Huang-Rhys factor and threshold intensity ratio.

    :param S: Huang-Rhys factor.
    :type S: float
    :param threshold: Threshold for determining the cut-off, defaults to 1e-2
    :type threshold: float, optional
    :return: Level for highest vibrational level to be considered.
    :rtype: int

    This function determines the highest vibrational level with intensity ratio above `threshold`.
    The intensity ratio is compared to the vibrational level with highest intensity.
    '''    
    def poisson(S: float,n: int) -> float:
        return np.exp(-S)*S**n/math.factorial(n)
    
    amax = poisson(S,0)
    n=1
    a = poisson(S,1)
    while amax * threshold < a:
        n += 1 
        a = poisson(S,n)
        amax = max(amax,a)
    return n

def FC_spectrum_mode(huang_rhys: float,freq: float, dx: float=0.25e-3, threshold: float=1e-3) -> np.ndarray:
    '''Create Franck-Condon spectrum for single vibrational mode with Dirac delta peaks.

    :param huang_rhys: Huang-Rhys factor of vibrational mode.
    :type huang_rhys: float
    :param freq: Energy in eV of vibrational mode.
    :type freq: float
    :param dx: Point density in eV for spectrum, defaults to 0.25e-3.
    :type dx: float, optional
    :param threshold: Cut-off value for vibrational level, defaults to 1e-3 (see `vibtools.FC_max_n`).
    :type threshold: float, optional
    :return: Array containing spectrum, with first value being at zero energy, aka the elastic peak.
    :rtype: np.ndarray
    '''        
    n = FC_max_n(huang_rhys,threshold)
    
    length = int(np.round((n-1)*freq/dx)+1)
    a = np.zeros((length))
    for i in range(n):
        a[int(np.round(i*freq/dx))]=np.exp(-huang_rhys)*huang_rhys**i/math.factorial(i)
    return a

def FC_spectrum(huang_rhys: np.ndarray, freqs: np.ndarray, dx: float=0.25e-3, threshold: float=1e-3) -> tuple[np.ndarray,np.ndarray]:
    '''Create full Franck-Condon spectrum for multiple vibrational modes ("progression of progression") with Dirac delta peaks.

    :param huang_rhys: Huang-Rhys factors of vibrational modes.
    :type huang_rhys: np.ndarray
    :param freqs: Energies of vibrational modes in eV.
    :type freqs: np.ndarray
    :param dx: Point density in eV for spectrum, defaults to 0.25e-3.
    :type dx: float, optional
    :param threshold: Cut-off value for vibrational level, defaults to 1e-3 (see `vibtools.FC_max_n`).
    :type threshold: float, optional
    :return: Intensity and energy of Franck-Condon spectrum.
    :rtype: tuple[np.ndarray,np.ndarray]
    '''    
    assert freqs.shape == huang_rhys.shape, "Freqs and Huang-Rhys factors do not match in shape"
    spec = np.ones((1,))
    for S,hw in zip(huang_rhys,freqs):
        if S > threshold and hw > 0:
            spec = np.convolve(spec,FC_spectrum_mode(S,hw,dx,threshold))
    spec = crop_spec(spec,threshold)
    return spec, np.linspace(0,(len(spec)-1)*dx,len(spec))

def crop_spec(spec: np.ndarray, threshold: float=1e-3) -> np.ndarray:
    '''Crop end of spectrum, with all discarded values being smaller than `max(spec)*threshold`.

    :param spec: Input array
    :type spec: np.ndarray
    :param threshold: Threshold for determining the cut-off, defaults to 1e-3
    :type threshold: float, optional
    :return: Cropped array.
    :rtype: np.ndarray
    '''    
    return np.resize(spec, (np.where(spec > np.max(spec)*threshold)[0][-1]+1,))

### ----------------------------------------------------------------------------

def lorentzian(gamma: float, dx: float=0.25e-3, threshold: float=1e-3) -> tuple[np.ndarray,float,np.ndarray]:
    '''create lorentzian lineshape for convolution

    :param gamma: Half width at half maximum of lorentzian lineshape
    :type gamma: float
    :param dx: Point density in eV for array, defaults to 0.25e-3. Must match point density of the Franck-Condon spectrum to be convolved with.
    :type dx: float, optional
    :param threshold: Cut-off value to determine length of array, defaults to 1e-3.
    :type threshold: float, optional
    :return: Lorentzian lineshape, offset in eV and x values in eV.
    :rtype: tuple[np.ndarray,float,np.ndarray]
    '''    
    offset = np.sqrt(abs(gamma**2/threshold))
    lorx = np.linspace(-offset,offset,int(2*offset/dx)+1)
    return gamma/np.pi/(lorx**2+gamma**2), offset, lorx


def convolve_spec(spec: np.ndarray, specx: np.ndarray, lineshape: np.ndarray, offset: float, normalize=True) -> tuple[np.ndarray,np.ndarray]:
    '''Convolve spectrum with given lineshape.

    :param spec: y values of spectrum
    :type spec: np.ndarray
    :param specx: x values of spectrum
    :type specx: np.ndarray
    :param lineshape: lineshape with same x spacing like `spec`.
    :type lineshape: np.ndarray
    :param offset: x-offset of lineshape (e.g. see `vib_tools.lorentzian`).
    :type offset: float
    :param normalize: Normalize spectrum such that maximum value is 1. Defaults to True
    :type normalize: bool, optional
    :return: x and y arrays of convoluted spectrum.
    :rtype: tuple[np.ndarray,np.ndarray]
    '''    
    conv = np.convolve(spec,lineshape)
    conv_x = np.linspace(-offset+specx[0],specx[-1]+offset,len(lineshape)+len(spec)-1)
    if normalize:
        conv *= 1/np.max(conv)
    return conv_x, conv


### ----------------------------------------------------------------------------
#        Solve Hessian Matrix to obtain vibrational normal modes
### ----------------------------------------------------------------------------

def hessian2normalmodes(hessian: np.ndarray, coords: np.ndarray, mass: np.ndarray, cm: bool=False, mass_weighted: bool=False) -> tuple[np.ndarray,np.ndarray]:
    '''Calculate vibrational normal modes and their energies form Hessian matrix.

    :param hessian: Hessian matrix in units of Hartree/bohr_radius**2
    :type hessian: np.ndarray
    :param coords: Atomic coordinates in Angstrom
    :type coords: np.ndarray
    :param mass: Atomic mass in units of atomic mass.
    :type mass: np.ndarray
    :param cm: Return energies of normal modes in cm**-1 instead of eV, defaults to False
    :type cm: bool, optional
    :param mass_weighted: Return mass-weighted normal modes, defaults to False
    :type mass_weighted: bool, optional
    :return: Energies (1d) and vibrational normal modes (3d) with shape (#mode,#atoms,3).
    :rtype: tuple[np.ndarray,np.ndarray]
    '''    

    # generate orthogonal space where translational and rotational vectors are seperated out
    T = translational_normalmodes(mass)
    R = rotational_normalmodes(coords,mass)
    D = gram_schmidt(np.concatenate((T,R,np.eye(3*len(mass))),axis=0))


    # translate hessian to mass-weighted internal coordinates and diagonalize
    mass_weighted_hessian = hessian/np.sqrt(np.outer(np.repeat(mass,3),np.repeat(mass,3)))
    hess_int = D.T @ mass_weighted_hessian @ D
    lambd, L = np.linalg.eig(hess_int)
    
    # resort to increasing order of energies
    sorted_indices = np.argsort(lambd)
    lambd = lambd[sorted_indices] # square of normal mode energies
    L = L[:,sorted_indices] # normal modes in mass-weighted internal coordinates
    
    # translate eigenvalues to energies
    fac = hartree/(bohr2ang*1e-10)**2/atomic_mass # sqrt(Hartree/u)/Bohr to eV
    freqs = np.sign(lambd)*np.sqrt(abs(lambd)*fac)*hbar/q_e
    if cm == True:
        freqs *= 8065.547   

    # translate eigenvectors to non-mass-weighted cartesian normal-modes and renorm
    M = np.sqrt(np.diag(1/np.repeat(mass,3)))
    normal_modes = M @ D @ L
    normal_modes *= 1/np.linalg.norm(normal_modes, axis=0, keepdims=True)
    
    # reshape normalmodes and remove translational and vibrational modes
    normal_modes = np.reshape(normal_modes, (3, len(freqs) // 3, len(freqs)),order="F")
    normal_modes = np.swapaxes(normal_modes,0,2)
    freqs = freqs[6:]
    normal_modes = normal_modes[6:,:,:]
    
    if mass_weighted:
        normal_modes *= np.sqrt(mass[None,:,None])
        normal_modes *= 1/np.sqrt(np.sum(normal_modes**2,axis=(1,2))[:,None,None]) # renorm
        
    return freqs, normal_modes
    
def rotational_normalmodes(coords: np.ndarray, mass: np.ndarray) -> np.ndarray:
    '''Calculate rotational normal modes.

    :param coords: Atomic coordinates in Angstrom
    :type coords: np.ndarray
    :param mass: Atomic mass in units of atomic mass.
    :type mass: np.ndarray
    :return: Rotational normal modes.
    :rtype: np.ndarray
    '''

    # make sure center of mass is at coordinate origin    
    center_of_mass = np.sum(coords*mass[:,None], axis=0)/np.sum(mass)
    coords -= center_of_mass[None,:]

    # make inertia tensor and solve for eigenvectors
    def inertia_tensor(coords: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        '''Create tensor of inertia.'''
        x,y,z = coords.T
        Ixx = np.sum(mass * (y**2 + z**2))
        Iyy = np.sum(mass * (x**2 + z**2))
        Izz = np.sum(mass * (x**2 + y**2))
        Ixy = -np.sum(mass * x * y)
        Iyz = -np.sum(mass * y * z)
        Ixz = -np.sum(mass * x * z)
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    inertia, X = np.linalg.eig(inertia_tensor(coords,mass))    
    
    # build rotational normal modes from eigenvectors
    P = np.matmul(X,coords.T)
    R = np.zeros((3,3,len(mass)))
    R[0,:,:] = (P[1,:][None,:]*X[:,2][:,None]-P[2,:][None,:]*X[:,1][:,None])*np.sqrt(mass[None,:])
    R[1,:,:] = (P[2,:][None,:]*X[:,0][:,None]-P[0,:][None,:]*X[:,2][:,None])*np.sqrt(mass[None,:])
    R[2,:,:] = (P[0,:][None,:]*X[:,1][:,None]-P[1,:][None,:]*X[:,0][:,None])*np.sqrt(mass[None,:])
    R = R.reshape((3,-1),order='F')

    return R / np.linalg.norm(R, axis=1, keepdims=True)

# ----------------------------------------------------------------------------

def translational_normalmodes(mass: np.ndarray) -> np.ndarray:
    '''Create translational normalmodes

    :param mass: Atomic mass in units of atomic mass.
    :type mass: np.ndarray
    :return: Translational normal modes
    :rtype: np.ndarray
    '''    
    T = np.concatenate([np.eye(3)]*len(mass),axis=0)*np.sqrt(np.repeat(mass,3))[:,None]
    return T.T / np.linalg.norm(T.T, axis=1, keepdims=True)

# ----------------------------------------------------------------------------

def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    '''Construct orthonormal basis using the Gram-Schmidt algorithm.

    :param vectors: Array with shape `(m,n)`, with `m` being number of vectors and `n` their length.
    :type vectors: np.ndarray
    :return: Orthonormal basis set of vectors with normalized length.
    :rtype: np.ndarray
    '''    
    basis = [vectors[0,:]]
    for v in vectors[1:,:]:
        w = v - np.sum(np.array([np.dot(v,b)*b  for b in basis]),axis=0)
        if np.linalg.norm(w) > 1e-10:  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)