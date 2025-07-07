import numpy as np
import os
from .orca_parser import HessMolecule, OutMolecule, JSONMolecule
from .orca_utils import dict2json, gbw2json, json2gbw, gbw2cube, write_orcaplot_inputfile, cube_dims, element_to_number
from .orca_utils import q_e, hbar, atomic_mass # some constants

try:
    import ujson as json
except ModuleNotFoundError:
    import json

class Molecule:
    '''Class to parse and analyze DFT calculations from ORCA.'''
    
    def __init__(self, filenames: str | os.PathLike | list[str], orca_path="", parse_gbw=False):
        '''Initialize instance of `Molecule` class.

        :param filenames: Path or list of paths to files to be loaded.
        :type filenames: str | os.PathLike | list[str]
        :param orca_path: Path to directory containing ORCA binaries, defaults to "". Only required if ORCA is not added to PATH
        :type orca_path: str, optional
        :param parse_gbw: Unpack gbw/nto binary and load parse input. Requires installation of ORCA, defaults to False
        :type parse_gbw: bool, optional
        '''        
        try:
            filename = filenames.pop(0)
        except AttributeError:
            filename = filenames
            filenames = []
        
        self.filename = filename
        '''First filename with which Molecule instance was initiated.'''
        self.basename = self._get_basename(filename)
        '''Basename of `self.filename`'''

        self.atoms = None
        '''List of dictionaries containing information for all atoms'''
        
        self.input_str = None
        '''Content of input file for ORCA calculation.'''
        self.energy = None
        '''Total Energy (eV)'''
        self.multiplicity = None
        '''Multiplicity'''
        self.charge = None
        '''Charge. Positive values indicates removal of electrons.'''
        self.UKS = None
        '''Boolean: Is unrestricted?'''
        self.success = None
        '''Boolean: Did ORCA terminate normally?'''

        self.occupation = None
        '''Array containing occupation of MOs. Array is 1d if `UKS=False` and 2d otherwise.'''
        self.MO_energies = None
        '''Array containing energies of MOs. Array is 1d if `UKS=False` and 2d otherwise.'''

        self.TDstates = None
        '''List of dict containing excited states.'''
        self.NTOs = None
        '''List of NTO states.'''
        self.tdip = None
        '''Transition Dipole Moment from ESD calculation.'''
        self.dtdip = None
        '''Transition Dipole Derivative from ESD calculation.'''

        self.freqs = None
        '''Energies of vibrational normal modes (eV)'''
        self.normal_modes = None
        '''Vibrational normal modes.'''
        self.hess = None
        '''Hessian Matrix for vibrational analysis.'''
        self.ddipole = None
        '''Electric dipole derivative with respect to normal modes'''
        self.ir = None
        '''Table of IR intensities.'''
        
        self.gbw = None
        '''Path to gbw/nto file.'''
        self.json = None
        '''Dictionary containing content of gbw/nto file.'''

        self.orca_path = orca_path
        '''Path to directory containing ORCA binaries.'''

        # ----------------------------------------------------------

        self.__read_file(self.filename, parse_gbw=parse_gbw)
        self.add_source(filenames, parse_gbw=parse_gbw)

    @staticmethod
    def _get_basename(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]

    def add_source(self,filenames: str | os.PathLike | list[str], parse_gbw=False):
        '''Add additional source to existing instance of `Molecule` class.

        :param filenames: Path or list of paths to files to be loaded.
        :type filenames: str | os.PathLike | list[str]
        :param parse_gbw: Unpack gbw/nto binary and load parse input. Requires installation of ORCA, defaults to False.
        :type parse_gbw: bool, optional
        '''        
        if isinstance(filenames,str):
            self.__read_file(filenames, parse_gbw=parse_gbw)
        else:
            for filename in filenames:
                self.__read_file(filename, parse_gbw=parse_gbw)
    
    def __read_file(self, filename: str | os.PathLike, parse_gbw=False):
        '''Add content of source to instance `Molecule` class.

        :param filename: Path to file to be loaded.
        :type filename: str | os.PathLike
        :param parse_gbw: Unpack gbw/nto binary and load parse input. Requires installation of ORCA, defaults to False.
        :type parse_gbw: bool, optional
        '''        

        assert os.path.exists(filename), "File not found!"

        if filename.endswith('hess'):
            self.__add_attributes(HessMolecule(filename))

        elif filename.endswith('.gbw') or filename.endswith('.nto'):
            self.gbw = filename
            if parse_gbw:
                self.load_gbw()
                
        else:
            self.__add_attributes(OutMolecule(filename))
        

    def __add_attributes(self,parsed_mol: OutMolecule | HessMolecule | JSONMolecule):
        '''Add attributes from `parsed_mol` to this instance of `Molecule`.

        :param parsed_mol: Object containing data in attributes.
        :type parsed_mol: OutMolecule | HessMolecule | JSONMolecule
        '''        
        if parsed_mol == None:
            return None
        for attr in parsed_mol.__dict__.keys():
            if hasattr(self,attr):
                if type(getattr(self,attr)) == type(None):
                    setattr(self,attr, getattr(parsed_mol,attr))

    # -------------------------------------------------
    
    def load_gbw(self, parse=True, cleanup=True):
        '''Unpack gbw/nto binary to JSON using ORCA and load as dictionary.

        :param parse: Parse the JSON dict into attributes. Defaults to True
        :type parse: bool, optional
        :param cleanup: Remove JSON file afterwards, defaults to True
        :type cleanup: bool, optional
        '''        
        assert type(self.gbw) is not type(None), "Now gbw file defined"
        
        jsonfile = gbw2json(self.gbw)
        with open(jsonfile, 'r') as file:
            self.json = json.load(file)

        if cleanup:
            os.remove(jsonfile)

        if parse:
            self.__add_attributes(JSONMolecule(content=self.json))

    def save_to_gbw(self, json_dict: dict=None, json_filename: str = "", cleanup=True) -> str:
        '''Store dictionary to JSON file and convert to gbw using ORCA binaries.

        :param json_dict: Dictionary to be stored. If not defined `self.json` will be used.
        :type json_dict: dict, optional
        :param json_filename: Name of JSON file to be created. If not defined `self.basename+'_json2gbw.json'` will be used
        :type json_filename: str, optional
        :param cleanup: Remove JSON file afterwards, defaults to True
        :type cleanup: bool, optional
        :return: Path to created gbw file.
        :rtype: str
        '''        

        if json_dict is None:
            json_dict = self.json
        
        assert json_dict != None, "No JSON template available!"

        if len(json_filename) == 0:
            json_filename = self.basename + "_json2gbw.json"
        
        with open(json_filename, 'w') as f:
            json.dump(json_dict, f, indent=4)

        gbwfile = json2gbw(json_filename,self.orca_path)

        if cleanup:
            os.remove(json_filename)

        return gbwfile    

    # -------------------------------------------------
            
    @property
    def mass(self) -> np.ndarray:
        '''Mass of atoms in atomic units.'''        
        return np.array([atom["mass"] for atom in self.atoms])

    @property
    def nuclearcharge(self) -> np.ndarray:
        '''Nuclear charge of atoms.'''
        return np.array([atom["nuclearcharge"] for atom in self.atoms])
        
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
    def coords(self) -> np.ndarray:
        '''Coordinates of atoms'''
        return np.array([atom["coords"] for atom in self.atoms])#.transpose()
    
    @coords.setter
    def coords(self, new_coords: np.ndarray):
        '''Set coordinates of atoms.

        :param new_coords: 2d array of dimension (number_of_atoms,3).
        :type new_coords: np.ndarray
        '''        
        for i, atom in enumerate(self.atoms):
            atom["coords"] = new_coords[i,:]

    @property
    def center_of_mass(self) -> np.ndarray:
        '''Center of mass of molecule.'''
        return np.sum(self.coords*self.mass[:,None], axis=0)/np.sum(self.mass)

    @property
    def coords_masscentered(self) -> np.ndarray:
        '''Coordinates of atoms, relative to the center of mass.'''
        return self.coords - self.center_of_mass[None,:]
    
    @property
    def mwnm(self) -> np.ndarray:
        '''Mass weighted vibrational normal modes.

        :return: 3d array of dimensions (number_of_normal_modes, number_of_atoms, 3)
        :rtype: np.ndarray
        '''        
        try:
            mwc = self.normal_modes*np.sqrt(self.mass[None,:,None])
            renorm = np.sqrt(np.sum(mwc**2,axis=(1,2)))
            return mwc/renorm[:,None,None]
        except:
            return None
    
    @property
    def num_electrons(self) -> int:
        '''Number of electrons in molecule.'''
        if type(self.occupation) != type(None):
            return int(np.sum(self.occupation))
        else:
            return None
    
    @property
    def num_alpha(self) -> int:
        '''Number of spin up electrons.'''
        if type(self.occupation) == type(None):
            return None
        elif self.UKS:
            return int(np.sum(self.occupation[0,:]))
        else:
            return int(np.sum(self.occupation[:])/2)
    
    @property
    def num_beta(self) -> int:
        '''Number of spin down electrons.'''
        if type(self.occupation) == type(None):
            return None
        elif self.UKS:
            return int(np.sum(self.occupation[1,:]))
        else:
            return int(np.sum(self.occupation[:])/2)
    
    @property
    def homo(self) -> int:
        '''Number of highest occupied molecular orbital (RKS only).'''
        assert np.ndim(self.occupation) == 1, "Molecule is not restricted!"
        return int(np.sum(self.occupation == 2.0)-1)

    @property
    def lumo(self) -> int:
        '''Number of highest unoccupied molecular orbital (RKS only).'''
        return self.homo+1
    
    def __sxmo(self,x: int, s: int) -> int:
        '''Number of highest or lowest molecular orbital for spin `s`.

        :param x: Return highest occupied (x==1) or lowest unoccupied (x==0) orbital
        :type x: int
        :param s: Spin up (s==0) or down (s==1)
        :type s: int
        :return: Number of molecular orbital
        :rtype: int
        '''        
        assert np.ndim(self.occupation) == 2, "Molecule is not unrestricted!"
        return int(np.sum(self.occupation[s,:] == 1.0)-x)
    
    @property
    def somo(self) -> int:
        '''Return highest singly occupied orbital (UKS only).'''
        return self.somo_a
        
    @property
    def sumo(self) -> int:
        '''Return lowest singly unoccupied orbital (UKS only).'''
        return self.sumo_b

    @property
    def somo_a(self) -> int:
        '''Return highest singly occupied alpha orbital (UKS only).'''        
        return self.__sxmo(1,0)
        
    @property
    def sumo_a(self) -> int:
        '''Return lowest singly unoccupied alpha orbital (UKS only).'''
        return self.__sxmo(0,0)

    @property
    def somo_b(self) -> int:
        '''Return highest singly occupied beta orbital (UKS only).'''        
        return self.__sxmo(1,1)
        
    @property
    def sumo_b(self) -> int:
        '''Return lowest singly unoccupied beta orbital (UKS only).'''
        return self.__sxmo(0,1)

    def get_ntos(self, state: int , threshold: float = 0) -> list[tuple[int, int, float]]:
        '''List of NTOs for `state` with occupation larger than `threshold`

        :param state: Number of excited state.
        :type state: int
        :param threshold: Consider only NTOs with occupation larger than `threshold`, defaults to 0
        :type threshold: float, optional

        :return: List of tuples containing NTO number, spin and occupation.
        :rtype: list[tuple[int, int, float]]
        '''        
        assert type(self.NTOs) is not type(None), "No NTOs found for this object."
        return [(nto['MO'], nto['Spin'], nto['Occupation']) for nto in self.NTOs[state]['NTOs']
                if nto['Occupation'] > threshold]
    
    # -------------------------------------------------

    def dislocated_coords(self, mode: int, amplitude=1.0, mass_centered = False) -> np.ndarray:
        '''Coordinates of atoms after dislocation by vibrational normal mode.

        :param mode: Number of vibrational normal mode.
        :type mode: int
        :param amplitude: Dislocation amplitude. Defaults to 1.0, which is HWHM of vibrational ground state.
        :type amplitude: float, optional
        :param mass_centered: Use mass centered coordinates. Defaults to False
        :type mass_centered: bool, optional
        :return: Dislocated coordinates.
        :rtype: np.ndarray
        '''        
        # factor = sqrt(hbar**2 / m_e / 2 m(u) hw(eV) / 1.66E-27 kg/u ) * 1E10 Ang/m
        factor = np.sqrt(hbar**2/q_e/2/self.mass/atomic_mass/abs(self.freqs[mode]))*1e10
        if mass_centered:
            return self.coords_masscentered + amplitude*factor[:,None]*self.mwnm[mode,:,:]
        else:
            return self.coords + amplitude*factor[:,None]*self.mwnm[mode,:,:]
    
    def write_to_xyz(self, filename: str='', coords: np.ndarray=None, infostr=""):
        '''Write coordinates into *.xyz file.

        :param filename: Name of xyz file to be created.  If not specified a filename will be created automatically.
        :type filename: str, optional
        :param coords: Coordinates do be used. If not defined `self.coords` will be used.
        :type coords: np.ndarray, optional
        :param infostr: Info string to be printed in the file (will be reduced to single line)
        :type infostr: str, optional
        :return: Name of file.
        :rtype: str
        '''
        if type(coords) == type(None):
            coords = self.coords
        
        content = f"{len(self.atoms)}\n"
        content += infostr.replace("\n", "; ").replace("\r", "")+'\n'
        for atom, atom_coords in enumerate(coords):
            content += f'{self.element[atom]:<2}'
            for xyz in atom_coords:
                content += '{:>23}'.format(f'{xyz:.12e}')
            content += f'\n'
        content += '\n'

        if filename == "":
            filename = self.basename + '.xyz'

        with open(filename, "wb") as file:
            file.write(bytes(content,"utf-8"))
            
    def dislocate_to_xyz(self, mode: int, amplitude=1.0, filename="", mass_centered = False) -> str:
        '''Write coordinates dislocated by vibrational normal mode to *.xyz file.

        :param mode: Number of vibrational normal mode
        :type mode: int
        :param amplitude:  Dislocation amplitude. Defaults to 1.0, which is HWHM of vibrational ground state.
        :type amplitude: float, optional
        :param filename: Name of xyz file to be created. If not specified a filename will be created automatically.
        :type filename: str, optional
        :param mass_centered: Use mass centered coordinates. Defaults to False
        :type mass_centered: bool, optional
        :return: Name of file.
        :rtype: str
        '''        
        coords = self.dislocated_coords(mode,amplitude,mass_centered)
        infostr=f'Dislocation mode={mode} a0={amplitude}'
        if filename == "":
            filename = self.basename + f"_m{mode}_a{amplitude}.xyz"
        self.write_to_xyz(filename, coords, infostr)

    def coords_to_json(self, coords: np.ndarray=None) -> dict:
        '''Update atom coordinates in `self.json` and return copy of JSON dictionary.

        :param coords: New coordinates to write into JSON dictionary. If omitted, the coordinates will be not changed.
        :type coords: np.ndarray, optional
        :return: Copy of JSON dictionary stored in `self.json`.
        :rtype: dict
        '''        
        assert self.json != None, "No JSON template available!"
        
        coords_json = json.loads(json.dumps(self.json))

        if type(coords) == type(None):
            return coords_json
        
        for i, atom_coords in enumerate(coords):
            coords_json["Molecule"]["Atoms"][i]["Coords"][0]=atom_coords[0]
            coords_json["Molecule"]["Atoms"][i]["Coords"][1]=atom_coords[1]
            coords_json["Molecule"]["Atoms"][i]["Coords"][2]=atom_coords[2]
        
        return coords_json    
    
    def make_cube(self, MO: int, spin=0, input: str | os.PathLike | dict = None,
                  orcaplotinput: str | os.PathLike = None,
                    cubename: str=None, plottype=1, fileformat=7, orcaplotinput_filename='./orcaplotinput',
                    cubedims: str=None,
                      coords: np.ndarray=None, z_range: float | tuple[float,float]=None, boundary=5., spacing=3.,
                  cleanup = True) -> str:
        '''Create cube file of molecular orbital using `orca_plot` binary.

        :param MO: Molecular orbital.
        :type MO: int
        :param spin: Consider spin up (spin=0, default) or spin down (spin=1).
        :type spin: int, optional
        :param input: String or path like object to *.gbw or *.JSON file, or JSON dictionary to be converted to *.gbw file. If not specified `self.gbw` will be used.
        :type input: str | os.PathLike | dict, optional
        :param orcaplotinput: input file to feed into `orca_plot` binary. If omitted (`None`) a new file will be created using `orca_utils.write_orcaplot_inputfile()`.
        :type orcaplotinput: str | os.PathLike, optional
        :param cubename: Name of cube file to be created. If omitted name will be automatically created. (Only used if `orcaplotinput=None`)
        :type cubename: str, optional
        :param plottype: Input value for `orca_plot` binary. plottype=1 (default) creates cube for molecular orbitals. (Only used if `orcaplotinput=None`)
        :type plottype: int, optional
        :param fileformat: Input value for `orca_plot` binary. fileformat=7 (default) creates Gaussian formatted cube. (Only used if `orcaplotinput=None`)
        :type fileformat: int, optional
        :param orcaplotinput_filename: Name of orcaplotinput file to be created, defaults to './orcaplotinput'. (Only used if `orcaplotinput=None`)
        :type orcaplotinput_filename: str, optional
        :param cubedims: String containing cube dimensions to be used in orcaplotinput file. See also `orca_utils.cube_dims()`. (Only used if `orcaplotinput=None`)
        :type cubedims: str, optional
        :param coords: Atomic coordinates to be used for creating `cubedims` string. If omitted `self.coords` is used.
        :type coords: np.ndarray, optional
        :param z_range: Z range of cube to be created. Valid input is either tuple[z_min,z_max] or single float (z_height) which will be expanded to tuple[z_height,z_height+bohr_radius/spacing]. If omitted `boundary` values will be used.
        :type z_range: float | tuple[float,float], optional
        :param boundary: Boundary around atoms to determine cube size in x,y and z dimensions. The latter can be overwritten by the `z_range` argument. Default is 5 Angstrom.
        :type boundary: float, optional
        :param spacing: Number of voxels per bohr_radius. Default is spacing=3, corresponding to a voxel distance of about 0.176 Angstrom.
        :type spacing: float, optional
        :param cleanup: Remove temporary files afterwards, e.g. *.JSON and *.gbw (if input was dictionary) or orcaplotinput file in case it was created.
        :type cleanup: bool, optional
        :return: String containing path to created cubefile.
        :rtype: str
        '''        
        delete_JSONfile = False
        delete_gbwfile = False
        delete_orcaplotinput = False

        if input == None:
            input = self.gbw
        elif isinstance(input,dict):
            if type(coords) is type(None):
                coords = np.array([atom["Coords"] for atom in input["Molecule"]["Atoms"]])
            input = dict2json(input,'./tmpjsonfile.JSON')
            delete_JSONfile = cleanup
            
        if os.path.isfile(input):
            if os.path.splitext(input)[-1].lower() == ".json":
                gbwfile = json2gbw(input,self.orca_path)
                delete_gbwfile = cleanup
                if delete_JSONfile: os.remove(input)
            elif os.path.splitext(input)[-1].lower() in ['.gbw','.nto']:
                gbwfile = input
        
        if orcaplotinput == None:
            if cubedims==None:
                if type(coords) is type(None): coords = self.coords
                cubedims = cube_dims(coords, z_range=z_range, boundary=boundary, spacing=spacing)

            if not isinstance(cubename,str):
                cubename = f'{self.basename}_MO{MO}{"ab"[spin]}.cub'

            orcaplotinput, cubename = write_orcaplot_inputfile(cubedims, MO, spin=spin,
                                                            cubename=cubename, input_filename=orcaplotinput_filename,
                                                            plottype=plottype, fileformat=fileformat)
            delete_orcaplotinput = cleanup

        gbw2cube(gbwfile, orcaplotinput, orca_path=self.orca_path)
        
        if delete_orcaplotinput:
            os.remove(orcaplotinput)
        if delete_gbwfile:
            os.remove(gbwfile)

        return cubename
        
    def __str__(self):
        return self.filename
    
    def __dir__(self):
        all_attrs = super().__dir__()
        return [attr for attr in all_attrs if not attr.startswith("_")]


