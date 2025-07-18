import numpy as np
import os
import copy

try: 
    from scipy.signal import convolve2d
except ModuleNotFoundError:
    print('WARNING: Could not load module `scipy.signal`. It is only required for `method="Bardeen"` in `Cube.sim_STS()` and `Cube.extrapolate_wavefunction()`.')

from .orca_utils import number_to_element, element_to_number, pad_lin_extrapolate, chunked
from .orca_utils import default_masses, bohr2ang, hbar, m_e, q_e # some physical constants

# --------------------------------------------------------------------------------------------------------------

class Cube:
    '''Class to load Gaussian formatted cubes and simulate constant height STS maps.'''

    def __init__(self,input: str | os.PathLike, deleteCubeFile=False, readData=True):
        '''Initialize instance of `Cube` class

        :param input: Either path to existing cube file, or string containing header of Gaussian cube.
        :type input: str | os.PathLike
        :param deleteCubeFile: Delete cube file after loading. To be used for temporary cube files. Default: False
        :type deleteCubeFile: bool, optional
        :param readData: If set to False read only header from Cube file, defaults to True
        :type readData: bool, optional
        :raises ValueError: Inputstr is neither valid file nor valid header.
        '''        

        self.filename = None
        '''Filename of cube.'''
        self.data = None
        '''Array containing data of cube'''
        self.natoms = None
        '''Number of atoms in molecule'''
        self.atoms = None
        '''List of dictionaries containing information for all atoms'''

        self.origin = None
        '''Origin of cube volume'''
        self.NVal = None
        '''Number of values per voxel'''
        self.n1 = None
        '''Number of points in first axis'''
        self.n2 = None
        '''Number of points in second axis'''
        self.n3 = None
        '''Number of points in third axis'''
        self.vec1 = None
        '''Length and direction of first axis'''
        self.vec2 = None
        '''Length and direction of first axis'''
        self.vec3 = None
        '''Length and direction of first axis'''

        self.nMO = 1
        '''Number of Orbitals in cube file'''
        self.isMO = False
        '''Does cube contain Molecular orbitals?'''
        self.vecMO = None
        '''List of molecular orbitals'''
        self.header = None
        '''Header lines of cube file'''

        try: # assume input is path to cube file
            self.filename = input
            header, pointer_position = self.load_cube_header(input)
            self.parse_cube_header(header)
            if readData:
                self.data = self.load_cube_data(input,pointer_position)
            if deleteCubeFile:
                os.remove(input)
        
        except AssertionError: # assume input is string containing a cube header
            try:
                self.parse_cube_header(input)
            except:
                raise ValueError('Inputstr is neither valid file nor valid header.')

    # -----------------------------------------------------------------------------------------------------

    @staticmethod
    def load_cube_header(filename: str | os.PathLike) -> tuple[str, int]:
        '''Load header from cube file.

        :param filename: Path to cube file.
        :type filename: str | os.PathLike
        :return: Tuple containing header as string and file position where header ends.
        :rtype: tuple[str, int]
        '''        
        assert os.path.exists(filename), "File does not exists."
        with open(filename, 'br') as f:
            header = ""
            for i in range(6):
                header += f.readline().decode("utf-8")
            nAtoms = int(header.split("\n")[2].split()[0])
            for i in range(abs(nAtoms)):
                header += f.readline().decode("utf-8")
            if nAtoms < 0:
                header += f.readline().decode('utf-8')

            return header, f.tell()
    
    def parse_cube_header(self, header: str):
        '''Parse header of cube to class attributes.

        :param header: String containing header of cube file.
        :type header: str
        '''

        def parse_cubeheader_line(line:str, factor=1.0):
            '''Helper function to parse single lines in cube file'''
            line = line.split()
            try:
                return abs(int(line[0])), np.array([float(x) for x in line[1:4]])*factor, int(line[4])
            except:
                return abs(int(line[0])), np.array([float(x) for x in line[1:4]])*factor, 1
            
        lines = header.removesuffix('\n').split('\n')

        self.header = [lines[0].removesuffix("\n"),lines[1].removesuffix("\n")] # first two lines of cube are comments

        # cube dimensions
        self.natoms, self.origin, self.NVal = parse_cubeheader_line(lines[2],factor=bohr2ang)
        self.n1, self.vec1, _ = parse_cubeheader_line(lines[3],factor=bohr2ang)
        self.n2, self.vec2, _ = parse_cubeheader_line(lines[4],factor=bohr2ang)
        self.n3, self.vec3, _ = parse_cubeheader_line(lines[5],factor=bohr2ang)

        # atoms in molecule
        self.atoms = []
        for i in range(self.natoms):
            line = lines[6+i].split()
            atom={}
            element, charge, coords = int(line[0]), float(line[1]), np.array([float(x) for x in line[2:]])*bohr2ang
            atom["element"] = number_to_element(element)
            atom["nuclearcharge"] = charge
            atom["coords"] = coords
            atom['mass']=default_masses[int(atom['nuclearcharge'])]
            self.atoms.append(atom)

        # if cube contains molecular orbital(s), parse the additional line stating how many and which MOs are in the cube
        if len(lines) == self.natoms+7:
            self.isMO = True
            self.nMO, self.vecMO = abs(int(lines[self.natoms+6].split()[0])), np.array([float(x) for x in lines[self.natoms+6].split()[1:]])
         
    def load_cube_data(self, filename: str | os.PathLike, pointer_position: int) -> np.ndarray:
        '''Load data from cube file into array.

        :param filename: Path to cube file
        :type filename: str | os.PathLike
        :param pointer_position: File position where data starts. See also `cube_utils.load_cube_header()`.
        :type pointer_position: int
        :return: Cube data.
        :rtype: np.ndarray
        '''        
        data = []
        with open(filename, 'br') as f:
            f.seek(pointer_position)
            for line in f:
                line = line.split()
                data += [float(val) for val in line]

        return self.reshape_data(np.array(data))

    def reshape_data(self, data: np.ndarray) -> np.ndarray:
        '''Reshape 1d data to n-dimensional array, depending on cubesize, number of orbitals and number of values per voxel.

        :param data: 1d array of cube data.
        :type data: np.ndarray
        :return: nd array of cube data.
        :rtype: np.ndarray
        '''        
        if self.NVal == 1:
            if self.nMO > 1:
                return data.reshape(self.shape+(self.nMO,))
            else:
                return data.reshape(self.shape)
        else:
            if self.nMO > 1:
                return data.reshape(self.shape+(self.nMO,self.NVal))
            else:
                return data.reshape(self.shape+(self.NVal,))

    # -----------------------------------------------------------------------------------------------------

    @property
    def parameters(self) -> str:
        '''Reconstruct parameter string from cube header.

        :return: String containing cube dimensions in atomic units.
        :rtype: str
        '''        
        if self.isMO:
            sign = -1
        else:
            sign = 1
        ns = np.stack([self.natoms*sign,self.n1,self.n2,self.n3])
        vecs = np.stack([self.origin,self.vec1,self.vec2,self.vec3])
        header = ""
        for i in range(4):
            header += f'{ns[i]:>5}'
            for j in range(3):
                header += '{:>12}'.format(f'{vecs[i,j]/bohr2ang:.6f}')
            header += "\n"
        return header

    @property
    def coords(self) -> np.array:
        '''Cartesiaon coordinates of atoms in Angstrom.

        :return: Array of shape (nAtoms,3).
        :rtype: np.array
        '''        
        return np.array([atom["coords"] for atom in self.atoms])

    @coords.setter    
    def coords(self, new_coords):
        '''Set coordinates of atoms.

        :param new_coords: 2d array of dimension (number_of_atoms,3).
        :type new_coords: np.ndarray
        '''        
        for i, icoords in enumerate(new_coords):
            self.atoms[i]["coords"] = icoords

    @property
    def element(self) -> list:
        '''Element names of atoms. For element number, use property `element_num`.
        '''
        return [atom["element"] for atom in self.atoms]

    @property
    def element_num(self) -> np.ndarray:
        '''Element numbers of atoms. For element names, use property `element`
        '''
        return element_to_number(self.element)

    @property
    def center_of_mass(self) -> np.ndarray:
        '''Center of mass of molecule.'''
        masses = default_masses[self.element_num]
        return np.sum(self.coords*masses[:,None], axis=0)/np.sum(masses)

    @property
    def shape(self) -> tuple[int,int,int]:
        '''Shape of cube dataset'''
        return (self.n1,self.n2,self.n3)

    @property
    def vecs(self) -> np.ndarray:
        '''2d array containing vec1, vec2, vec3.'''
        return np.stack((self.vec1, self.vec2, self.vec3))

    # -----------------------------------------------------------------------------------------------------

    def _cartesian_axis(self,dimension: int) -> np.ndarray:
        '''Linspace of cartesian axis, if cube dimension aligns with it.'''        
        axis_name=['x','y','z'][dimension]
        assert abs(self.vecs[dimension,dimension]) == np.linalg.norm(self.vecs[dimension,:]), f'Cube axis {dimension} does not align with {axis_name} axis.'
        return np.linspace(self.origin[dimension],self.origin[dimension]+(self.shape[dimension]-1)*self.vecs[dimension,dimension],self.shape[dimension])

    @property
    def x(self) -> np.ndarray:
        '''Array containing values of x-axis.'''
        return self._cartesian_axis(0)

    @property
    def y(self) -> np.ndarray:
        '''Array containing values of y-axis.'''
        return self._cartesian_axis(1)

    @property
    def z(self) -> np.ndarray:
        '''Array containing values of z-axis.'''
        return self._cartesian_axis(2)

    def meshgrid(self, pad: int=0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        '''Create meshgrid of cube

        :param pad: Pad cube with `pad` points in all axes, defaults to 0
        :type pad: int, optional
        :return: Three 3d arrays.
        :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
        '''        
        if pad == 0:
            return np.meshgrid(self.x,self.y,self.z, indexing='ij')
        else:
            x = np.pad(self.x,pad,pad_lin_extrapolate)
            y = np.pad(self.y,pad,pad_lin_extrapolate)
            z = np.pad(self.z,pad,pad_lin_extrapolate)
            return np.meshgrid(x,y,z,indexing='ij')
        
    # -----------------------------------------------------------------------------------------------------

    def _internal_axis(self,dimension: int) -> np.ndarray:
        '''Linspace of internal axis.'''        
        origin = np.dot(self.origin-self.center_of_mass,self.vecs[dimension,:])/np.linalg.norm(self.vecs[dimension,:])
        return origin+np.linalg.norm(self.vecs[dimension,:])*np.arange(self.shape[dimension])

    @property
    def x_int(self) -> np.ndarray:
        '''Array containing values of first internal axis.'''
        return self._internal_axis(0)

    @property
    def y_int(self) -> np.ndarray:
        '''Array containing values of second internal axis.'''
        return self._internal_axis(1)

    @property
    def z_int(self) -> np.ndarray:
        '''Array containing values of third internal axis.'''
        return self._internal_axis(2)
    
    def internal_meshgrid(self, pad: int=0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        '''Create meshgrid of cube using internal coordinates.

        :param pad: Pad cube with `pad` points in all axes, defaults to 0
        :type pad: int, optional
        :return: Three 3d arrays.
        :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
        '''        
        if pad == 0:
            return np.meshgrid(self.x_int,self.y_int,self.z_int, indexing='ij')
        else:
            x = np.pad(self.x_int,pad,pad_lin_extrapolate)
            y = np.pad(self.y_int,pad,pad_lin_extrapolate)
            z = np.pad(self.z_int,pad,pad_lin_extrapolate)
            return np.meshgrid(x,y,z,indexing='ij')

    def _ax_is_perp(self, axis: int) -> bool:
        '''Is `axis` perpendicular to the other two axes?

        :param axis: Axis number 0, 1, or 2.
        :type axis: int
        :return: True if axis is perpendicular.
        :rtype: bool
        '''        
        t = [0,1,2]
        t.remove(axis)
        return not np.linalg.norm(np.cross(np.cross(self.vecs[t[0]],self.vecs[t[1]]),self.vecs[axis]))

    # -----------------------------------------------------------------------------------------------------

    def sim_STS(self, tip_height: float=7., plane: float=1.5, method='TH',
                workfunction: float=5.4, roi: float=None, center_mass=True, MO: int=None,
                pad: int=0) -> np.ndarray:
        '''Simulate constant height STS image, using Bardeen or Tersoff-Hamann (TH) formalism.

        :param tip_height: Height of tip center in Angstrom, defaults to 7.
        :type tip_height: float, optional
        :param plane: Plane height in Angstrom for wave function cut off, defaults to 1.5
        :type plane: float, optional
        :param method: Method to calculate STS map. Options: Tersoff-Hamann (`TH`, default) or Bardeen (`Bardeen`).
        :type method: str, optional
        :param workfunction: Vacuum barrier height in eV, defaults to 5.4
        :type workfunction: float, optional
        :param roi: Bardeen only: Range of interest around tip from -roi to +roi in Angstrom. If omitted, roi will be set to `tip_height`.
        :type roi: float, optional
        :param center_mass: `tip_height` and `plane` are relative to molecules center of mass (default), otherwise they are absolute values.
        :type center_mass: bool, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: 2d array with constant height STS image
        :rtype: np.ndarray

        This function returns simply the square of `self.extrapolate_wavefunction()`
        '''        
        return self.extrapolate_wavefunction(tip_height, plane, method, workfunction, roi, center_mass, MO, pad)**2

    def extrapolate_wavefunction(self, tip_height: float=7., plane: float=1.5, method='TH',
                workfunction: float=5.4, roi: float=None, center_mass=True, MO: int=None,
                pad: int=0) -> np.ndarray:
        '''Extrapolate wavefunction to `tip_height`, using Bardeen or Tersoff-Hamann (TH) formalism, if `tip_height > plane`.

        :param tip_height: Height of tip center in Angstrom, defaults to 7.
        :type tip_height: float, optional
        :param plane: Plane height in Angstrom for wave function cut off, defaults to 1.5
        :type plane: float, optional
        :param method: Method to calculate STS map. Options: Tersoff-Hamann (`TH`, default) or Bardeen (`Bardeen`).
        :type method: str, optional
        :param workfunction: Vacuum barrier height in eV, defaults to 5.4
        :type workfunction: float, optional
        :param roi: Bardeen only: Range of interest around tip from -roi to +roi in Angstrom. If omitted, roi will be set to `tip_height`.
        :type roi: float, optional
        :param center_mass: `tip_height` and `plane` are relative to molecules center of mass (default), otherwise they are absolute values.
        :type center_mass: bool, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: 2d array with wavefunction extrapolated to `tip_height`.
        :rtype: np.ndarray
        '''        
        
        if center_mass:
            tip_height += self.center_of_mass[2]
            plane += self.center_of_mass[2]
        
        if tip_height <= plane:
            return self.integration_plane(plane, MO=MO, pad=pad)

        if method == "TH":
            return self.extrapolate_WF(tip_height, plane, workfunction, MO=MO, pad=pad)
        
        elif method == "Bardeen":
            return self.convolve_WF(tip_height-plane, plane, workfunction, MO=MO, pad=pad, roi=roi)



    def extrapolate_WF(self, z: float=7., cut_off: float=1.5, workfunction: float=5.4, MO: int=None, pad: int=0) -> np.ndarray:
        '''Cut off wave function at height `cut_off` and extrapolate into vacuum to height `z`.

        :param z: Height in Angstrom to which wave function will be extrapolated, defaults to 7.
        :type z: float, optional
        :param cut_off: Height in Angstrom at which the wavefunction will be cut off, defaults to 1.5
        :type cut_off: float, optional
        :param workfunction: Vacuum barrier height in eV, defaults to 5.4
        :type workfunction: float, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: _description_
        :rtype: np.ndarray
        '''        
        morb_plane = self.integration_plane(cut_off, MO=MO, pad=pad)

        # =====================================================================
        # The following code snipped was adapted from:
        # https://github.com/nanotech-empa/cp2k-spm-tools/blob/main/cp2k_spm_tools/cp2k_grid_orbitals.py
        #
        fourier = np.fft.rfft2(morb_plane)
        kx_arr = 2*np.pi*np.fft.fftfreq(morb_plane.shape[0], np.linalg.norm(self.vec1))
        ky_arr = 2*np.pi*np.fft.rfftfreq(morb_plane.shape[1], np.linalg.norm(self.vec2))
        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr,  indexing='ij')

        fac = 2*m_e*q_e/hbar**2*1e-20
        kappa = np.sqrt(kx_grid**2 + ky_grid**2 + fac*workfunction)

        dz = z - cut_off
        return np.fft.irfft2(fourier*np.exp(-kappa*dz), morb_plane.shape)
        #
        # =====================================================================

    def integration_plane(self, z: float=1.5, n: int=1, MO: int=None, pad: int=0) -> np.ndarray:
        '''Return xy slice of cube at height plane or subset of cube if `n>1`.

        :param z: Height of plane in Angstrom, defaults to 1.5
        :type z: float, optional
        :param n: Number of slices to return, defaults to 1
        :type n: int, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: 2d (n=1) or 3d (n>1) array
        :rtype: np.ndarray
        '''        
        assert self._ax_is_perp(2), "z is not perpendicular to xy plane."
        plane_idx = np.searchsorted(self.z, z)

        assert self.z[plane_idx]-z <= np.linalg.norm(self.vec3), 'Height of integration plane not within z-range of cube data.' 

        if self.data.ndim == 3 and MO == 0: MO=None
        padding = ((pad,pad),(pad,pad))
        if n==2: padding += ((0,0),)
        return np.pad(np.squeeze(self.data[:,:,plane_idx:plane_idx+n,MO]), padding)

    def convolve_WF(self, tip_height: float=7., int_plane: float=1.5, workfunction: float=5.4, roi: float=None, MO: int=None, pad: int=0) -> np.ndarray:
        '''Use Bardeen's formalism to simulate STM probability.

        :param tip_height: Height in Angstrom of tip center above center of molecule, defaults to 7.
        :type tip_height: float, optional
        :param int_plane: Height in Angstrom of integration plane above center of molecule, defaults to 1.5
        :type int_plane: float, optional
        :param workfunction: Vacuum barrier height in eV, defaults to 5.4
        :type workfunction: float, optional
        :param roi: Range of interest around tip from -roi to +roi in Angstrom. If omitted, roi will be set to `tip_height`.
        :type roi: float, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: 2d array containing convoluted wavefunction.
        :rtype: np.ndarray

        This function convolves slice of cube data at height `int_plane` with s-wave centered at `tip_height`.
        '''        
        psi = self.integration_plane(int_plane, n=2, MO=MO, pad=pad)
        dpsi = (psi[:,:,1]-psi[:,:,0])/np.linalg.norm(self.vec3)
        tip, dtip = self.tip_wavefuncs(tip_height,roi,workfunction)
        
        try:
            return convolve2d(psi[:,:,0],dtip,mode='same') - convolve2d(dpsi,tip,mode='same')
        except NameError:
            print('Bardeen method not available without scipy module, output consists of zeros.')
            return np.zeros_like(psi[...,0])
    

    def tip_wavefuncs(self, tip_height: float=7., roi: float=None, workfunction: float=5.4) -> tuple[np.ndarray,np.ndarray]:
        '''Create slice of s-wave function (e.g. for STM tip) and its derivative in z.

        :param tip_height: Distance to center of s-wave, defaults to 7.
        :type tip_height: float, optional
        :param roi: Range of slice in x and y direction will be from -roi to +roi. If omitted, roi will be set to `tip_height`.
        :type roi: float, optional
        :param workfunction: Workfunction of s-wave in eV, defaults to 5.4
        :type workfunction: float, optional
        :return: Tuple of 2d arrays containing wave function and its derivative in z.
        :rtype: tuple[np.ndarray,np.ndarray]
        '''      

        assert np.dot(self.vec1,self.vec2) == 0, 'x and y axis must be perpendicular.'

        if roi == None:
            roi = tip_height
        
        # x-axis
        d1 = np.linalg.norm(self.vec1)
        tn1 = int(np.ceil(roi/d1))
        x = np.linspace(-tn1*d1,tn1*d1,2*tn1+1)

        # y-axis
        d2 = np.linalg.norm(self.vec2)
        tn2 = int(np.ceil(roi/d2))
        y = np.linspace(-tn2*d2,tn2*d2,2*tn2+1)

        # z-axis
        d3 = np.linalg.norm(self.vec3)

        X, Y = np.meshgrid(x,y,indexing='ij')
        r = np.sqrt(X**2+Y**2+tip_height**2)
        dr = np.sqrt(X**2+Y**2+(tip_height-d3)**2)

        kap = np.sqrt(2*m_e*q_e*workfunction/hbar**2)*1e-10 # decay constant, based on workfunction

        tip = np.exp(-kap*r)/(kap*r)
        dtip = np.exp(-kap*dr)/(kap*dr)
        dtip = (dtip-tip)/d3

        return tip, dtip
            
    # -----------------------------------------------------------------------------------------------------

    def write_cube(self, filename: str, path: str | os.PathLike="./", header: str=None) -> os.PathLike:
        '''Write cube to file.

        :param filename: Name of cube file to be created. 
        :type filename: str
        :param path: Path to directory where cube file is to be stored, defaults to "./"
        :type path: str | os.PathLike, optional
        :param header: Up to two header lines (seperated by r'\n') to be added to cube file. If not defined `self.header` will be used.
        :type header: str, optional
        :return: Path to file created.
        :rtype: os.PathLike
        '''

        def two_line_header(header):
            '''Make sure header consists of exactly two lines.'''
            header += '\n\n'
            return '\n'.join(header.split('\n')[:2])+'\n'

        def atoms_string(atoms: np.ndarray) -> str:
            '''Create line for given atom dictionary.'''
            out = [""] * len(atoms)
            for a, atom in enumerate(atoms):
                line = [""] * 5
                line[0] = '{:>5}'.format(f'{element_to_number(atom["element"])}')
                line[1] = '{:>12}'.format(f'{atom["nuclearcharge"]:.6f}')
                for i in range(3):
                    line[2+i] = '{:>12}'.format(f'{atom["coords"][i]/bohr2ang:.6f}')

                out[a] = "".join(line)
            return '\n'.join(out)+'\n'

        def data_string(data: np.ndarray) -> str:
            '''Compile all cube data into one string with Gaussian formatting.'''
            out = [''] * data.shape[0]*data.shape[1]
            for ix in range(data.shape[0]):
                for iy in range(data.shape[1]):
                    out[ix*data.shape[1]+iy] = "\n".join(["".join(['{:>14}'.format(f'{a:.5E}') for a in row])
                                                               for row in list(chunked(data[ix,iy,:],6))])
            return '\n'.join(out)+'\n'
        

        # Make sure filename ends with .cub or .cube
        if not filename.endswith('.cub') and not filename.endswith('.cube'):
            filename+='.cub'
        
        # create file and write...
        filepath = os.path.join(path,filename)
        with open(filepath,"wb") as f:    
            
            # two header lines
            if type(header) is type(None):
                f.write(bytes('\n'.join(self.header)+'\n','utf-8'))
            else:
                f.write(bytes(two_line_header(header),'utf-8'))

            # cube dimensions
            f.write(bytes(self.parameters,'utf-8'))

            # list all atoms
            f.write(bytes(atoms_string(self.atoms),'utf-8'))

            # special line for molecular orbitals
            if self.isMO:
                f.write(bytes(''.join(['{:>5}'.format(int(a)) for a in [self.nMO]+list(self.vecMO)])+'\n','utf-8'))

            # actual data
            f.write(bytes(data_string(self.data),'utf-8'))

        return filepath
    
    # -----------------------------------------------------------------------------------------------------

    def copy(self) -> 'Cube':
        '''Create copy of this instance. See also `self.deepcopy()`.

        :return: Copy of instance
        :rtype: Cube
        '''        
        return copy.copy(self)
    
    def deepcopy(self) -> 'Cube':
        '''Create deepcopy of this instance. See also `self.copy()`.

        :return: Copy of instance
        :rtype: Cube
        '''        
        return copy.deepcopy(self)