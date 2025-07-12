import subprocess
import os
import numpy as np
try:
    import ujson
except ModuleNotFoundError:
    import json as ujson
# -----------------------------------------
# some useful constants

bohr2ang = 0.529177249 # bohr radius in Angstrom
'''Conversion factor from atomic units (Bohr radius) to Angstrom'''
hbar = 1.054571e-34 # Js
'''Reduced Planck constant in Joule seconds'''
m_e = 9.109383e-31 # kg
'''Electron mass in kg'''
q_e = 1.602176e-19 # C
'''Elementary charge in Coulomb'''
atomic_mass = 1.660539068e-27 # kg
'''Atomic mass in kg'''
hartree = 4.3597482e-18 # J
'''Hartree energy in Joule'''

# -----------------------------------------

def dict2json(input: dict, json_file: str) -> str:
    '''Save dictionary into JSON file.

    :param input: JSON dictionary
    :type input: dict
    :param json_file: Filename into which to write the dictionary.
    :type json_file: str
    :return: Filename of JSON file.
    :rtype: str
    '''    
    with open(json_file, 'w') as f:
        ujson.dump(input, f, indent=4)
    
    return json_file

def json2gbw(json_file: str | os.PathLike, orca_path: str | os.PathLike="") -> str:
    '''Create gbw file from JSON file, using ORCA binary `orca_2json`.

    :param json_file: Path to JSON file
    :type json_file: str | os.PathLike
    :param orca_path: Path to ORCA binaries. Only required if ORCA is not added to PATH 
    :type orca_path: str | os.PathLike, optional
    :return: Path to created gbw file.
    :rtype: str
    '''    
    if os.name == 'posix':
        ext = ''
    elif os.name == 'nt':
        ext = '.exe'

    subprocess.run([os.path.join(orca_path,"orca_2json"+ext), json_file, "-gbw"],
                    stdout=subprocess.DEVNULL)

    return os.path.splitext(json_file)[0]+"_copy.gbw"
    

def gbw2json(gbw_file: str | os.PathLike, orca_path: str | os.PathLike="", delete_bibtex=True) -> str:
    '''Unpack gbw/nto file to JSON, using ORCA binary `orca_2json`.

    :param gbw_file: Path to gbw/nto file
    :type gbw_file: str | os.PathLike
    :param orca_path: Path to ORCA binaries. Only required if ORCA is not added to PATH 
    :type orca_path: str | os.PathLike, optional
    :param delete_bibtex: Remove automatically created *.bibtex file.
    :type delete_bibtex: bool, optional
    :return: Path to created JSON file.
    :rtype: str
    '''    
    assert os.path.isfile(gbw_file), "Error: *.gbw file not found!"
    
    if os.name == 'posix':
        ext = ''
    elif os.name == 'nt':
        ext = '.exe'

    subprocess.run([orca_path+"orca_2json"+ext, gbw_file], stdout=subprocess.DEVNULL)
    suffix = gbw_file.split(".")[-1]

    if delete_bibtex:
        bibtexfile = gbw_file.removesuffix(suffix)+"JSON.bibtex"
        os.remove(bibtexfile)

    return gbw_file.removesuffix(suffix)+"json"


def gbw2cube(gbw_file: str | os.PathLike, orcaplotinput: str | os.PathLike, orca_path: str | os.PathLike=''):
    '''Create cube from gbw/nto file, using ORCA binary `orca_plot`.

    :param gbw_file: Path to *.gbw or *.nto file
    :type gbw_file: str | os.PathLike
    :param orcaplotinput: Path to orcaplotinput file to feed into `orca_plot` binary (see `orca_utils.write_orcaplot_inputfile()`)
    :type orcaplotinput: str | os.PathLike
    :param orca_path: Path to ORCA binaries. Only required if ORCA is not added to PATH 
    :type orca_path: str | os.PathLike, optional
    '''    

    assert os.path.exists(gbw_file), "Error: *.gbw file not found!"
    assert os.path.exists(orcaplotinput), "Error: orcaplotinput file not found!"

    if os.name == 'posix':
        ext = ''
    elif os.name == 'nt':
        ext = '.exe'
    subprocess.run([os.path.join(orca_path,"orca_plot"+ext), gbw_file, orcaplotinput],
                    stdout=subprocess.DEVNULL)
            
def write_orcaplot_inputfile(cubedims: str, MO: int, spin: int=0,
                            cubename: str=None, input_filename: str | os.PathLike='./orcaplotinput',
                            plottype: int=1, fileformat: int=7) -> tuple[str, str]:
    '''Create input file to be used with ORCA binary `orca_plot`.

    :param cubedims: String containing cube dimensions to be used in orcaplotinput file. See also `orca_utils.cube_dims()`.
    :type cubedims: str
    :param MO: Molecular orbital
    :type MO: int
    :param spin: Consider spin up (spin=0, default) or spin down (spin=1).
    :type spin: int, optional
    :param cubename: Name of cube file to be created. If omitted name will be automatically created.
    :type cubename: str, optional
    :param input_filename: Name of orcaplotinput file to be created, defaults to './orcaplotinput'.
    :type input_filename: str | os.PathLike, optional
    :param plottype: Input value for `orca_plot` binary. plottype=1 (default) creates cube for molecular orbitals.
    :type plottype: int, optional
    :param fileformat: Input value for `orca_plot` binary. fileformat=7 (default) creates Gaussian formatted cube.
    :type fileformat: int, optional
    :return: Tuple containing `input_filename` and `cubename`.
    :rtype: tuple[str, str]
    '''    

    if not isinstance(cubename,str):
        cubename = f'MO{MO}{"ab"[spin]}.cub'

    content = f"{plottype}\n" # plottype
    content += f"{fileformat}\n" # Format
    content += f"{MO} {spin}\n" # MO / spin

    content += "0\n" # state density
    content += "input\n" # input
    content += cubename+"\n" # output

    content += "1\n" # ncont
    content += "0\n" # icont
    content += "0\n" # skeleton
    content += "0\n" # atoms
    content += "0\n" # usecol

    content += cubedims

    content += "0 0 0\n" # at1 at2 at3
    content += "0 0 0\n" # v1 v1 v1
    content += "0 0 0\n" # v2 v2 v2
    content += "0 0 0\n" # v3 v3 v3

    with open(input_filename,"wb") as f:
        f.write(bytes(content,"utf-8"))
        
    return input_filename, cubename

def cube_dims(coords: np.ndarray, z_range: float | tuple[float,float]=None, boundary: float=5., spacing: float=3.) -> str:
    '''Create string containing cube dimensions to be used in orcaplotinput file. To be used with `orca_utils.write_orcaplot_inputfile()` or `Molecule.make_cube()`.

    :param coords: Atomic coordinates to determain range of cube.
    :type coords: np.ndarray
    :param z_range: Z range of cube to be created. Valid input is either tuple[z_min,z_max] or single float (z_height) which will be expanded to tuple[z_height,z_height+bohr_radius/spacing]. If omitted `boundary` values will be used.
    :type z_range: float | tuple[float,float], optional
    :param boundary: Boundary around atoms to determine cube size in x,y and z dimensions. The latter can be overwritten by the `z_range` argument. Default is 5 Angstrom.
    :type boundary: float, optional
    :param spacing: Number of voxels per bohr_radius. Default is spacing=3, corresponding to a voxel distance of about 0.176 Angstrom.
    :type spacing: float, optional
    :return: String containing cube dimensions.
    :rtype: str
    '''

    # find outer most atoms in x,y and z direction (in units of bohr_radius)    
    mins, maxs = np.min(coords, axis=0)/bohr2ang, np.max(coords, axis=0)/bohr2ang

    boundary*=1/bohr2ang

    def minmax(val1: float, val2: float, boundary: float, spacing: float):
        '''Find min and max values for given boundary while keeping the given spacing.'''
        vmin, vmax = min(val1,val2), max(val1,val2)
        b = (np.round((vmax-vmin+2*boundary)*spacing)/spacing-(vmax-vmin))/2
        return vmin-b, vmax+b

    # pad cube dimensions by adding +- boundery in x and y
    x_range = minmax(mins[0],maxs[0],boundary=boundary,spacing=spacing)
    y_range = minmax(mins[1],maxs[1],boundary=boundary,spacing=spacing)

    # determine z_min and z_max
    try:
        z_range = (z_range[0]/bohr2ang,z_range[1]/bohr2ang) # tuple given with min and max values
    except (TypeError, IndexError):
        if type(z_range) == type(None):
            # nothing defined, use boundary padding like x,y
            z_range = minmax(mins[2],maxs[2],boundary=boundary,spacing=spacing)
        else:
            # single value given
            z_range = (z_range/bohr2ang,z_range/bohr2ang+1./spacing)

    # number of points for each dimension
    nx=int(np.round((x_range[1]-x_range[0])*spacing)+1)
    ny=int(np.round((y_range[1]-y_range[0])*spacing)+1)
    nz=int(np.round((z_range[1]-z_range[0])*spacing)+1)

    # write string to be used in orcaplot inputfile
    cubedims_str = ""
    cubedims_str += f"{nx} {ny} {nz}\n"
    cubedims_str += f"{x_range[0]:.6f} {x_range[1]:.6f}\n"
    cubedims_str += f"{y_range[0]:.6f} {y_range[1]:.6f}\n"
    cubedims_str += f"{z_range[0]:.6f} {z_range[1]:.6f}\n"

    return cubedims_str

# ------------------------------------------------------------------------

# array to be used for default masses of elements
default_masses = np.array([
		np.nan,	1.008,	4.0026,	6.94,	9.0122,	10.81,
		12.011,	14.007,	15.999,	18.998,	20.18,	22.99,
		24.305,	26.982,	28.085,	30.974,	32.06,	35.45,
		39.95,	39.098,	40.078,	44.956,	47.867,	50.942,
		51.996,	54.938,	55.845,	58.933,	58.693,	63.546,
		65.38,	69.723,	72.63,	74.922,	78.971,	79.904,
		83.798,	85.468,	87.62,	88.906,	91.222,	92.906,
		95.95,	np.nan,	101.07,	102.91,	106.42,	107.87,
		112.41,	114.82,	118.71,	121.76,	127.6,	126.9,
		131.29,	132.91,	137.33,	138.91,	140.12,	140.91,
		144.24,	np.nan,	150.36,	151.96,	157.25,	158.93,
		162.5,	164.93,	167.26,	168.93,	173.05,	174.97,
		178.49,	180.95,	183.84,	186.21,	190.23,	192.22,
		195.08,	196.97,	200.59,	204.38,	207.2,	208.98,
		np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,
		232.04,	231.04,	238.03,	np.nan,	np.nan,	np.nan,
		np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,
		np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,
		np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,
		np.nan,	np.nan,	np.nan,	np.nan,	np.nan])

# list of element names
_element_symbols = [
'',     'H',	'He',	'Li',	'Be',	'B',
'C',	'N',	'O',	'F',	'Ne',	'Na',
'Mg',	'Al',	'Si',	'P',	'S',	'Cl',
'Ar',	'K',	'Ca',	'Sc',	'Ti',	'V',
'Cr',	'Mn',	'Fe',	'Co',	'Ni',	'Cu',
'Zn',	'Ga',	'Ge',	'As',	'Se',	'Br',
'Kr',	'Rb',	'Sr',	'Y',	'Zr',	'Nb',
'Mo',	'Tc',	'Ru',	'Rh',	'Pd',	'Ag',
'Cd',	'In',	'Sn',	'Sb',	'Te',	'I',
'Xe',	'Cs',	'Ba',	'La',	'Ce',	'Pr',
'Nd',	'Pm',	'Sm',	'Eu',	'Gd',	'Tb',
'Dy',	'Ho',	'Er',	'Tm',	'Yb',	'Lu',
'Hf',	'Ta',	'W',	'Re',	'Os',	'Ir',
'Pt',	'Au',	'Hg',	'Tl',	'Pb',	'Bi',
'Po',	'At',	'Rn',	'Fr',	'Ra',	'Ac',
'Th',	'Pa',	'U',	'Np',	'Pu',	'Am',
'Cm',	'Bk',	'Cf',	'Es',	'Fm',	'Md',
'No',	'Lr',	'Rf',	'Db',	'Sg',	'Bh',
'Hs',	'Mt',	'Ds',	'Rg',	'Cn',	'Nh',
'Fl',	'Mc',	'Lv',	'Ts',	'Og']

def element_to_number(symbol: str | list[str]) -> int | np.ndarray:
    '''Return element number(s) for given element name (abbreviated) or list of names.

    :param symbol: Element name (e.g. 'H' or 'Fe') or list of element names.
    :type symbol: str | list[str]
    :return: Element number or array containing element numbers.
    :rtype: int | np.ndarray
    '''    
    try:
        return _element_symbols.index(symbol)
    except ValueError:
        try:
            return np.array([_element_symbols.index(s) for s in symbol], dtype=np.dtype(int))
        except ValueError:
            return None

def number_to_element(num: int | list[int]) -> str | list[str]:
    '''Return element name (abbreviated) for given element number or list of numbers.

    :param num: Element number or list/array of numbers.
    :type num: int | list[int]
    :return: String or list of strings containing abbreviated element names (e.g. 'H' or 'Fe')
    :rtype: str | list[str]
    '''    
    try:
        return _element_symbols[num]
    except ValueError:
        try:
            return [_element_symbols[n] for n in num]
        except ValueError:
            return None
        
# -----------------------------------------------------------------------------------------

def pad_lin_extrapolate(vector: np.ndarray, pad_width: tuple, iaxis: int, kwargs):
    '''Function for `np.pad()` to extrapolate linearly (see numpy docs).
    EXAMPLE: `x_padded = np.pad(x, pad_width, pad_lin_extrapolate)`
    '''        
    dd = vector[pad_width[0]+1]-vector[pad_width[0]]
    vector[:pad_width[0]] = vector[pad_width[0]]-(pad_width[0]-np.arange(pad_width[0]))*dd
    vector[-pad_width[1]:] = vector[-pad_width[1]-1]+(np.arange(pad_width[1])+1)*dd

# -----------------------------------------------------------------------------------------

def chunked(a: list | np.ndarray, n: int) -> list:
    '''Chunk list into several lists of length n.

    :param a: Input list or 1d array
    :type a: list | np.ndarray
    :param n: Size of chunks.
    :type n: int
    :return: List of lists containing individual chunks.
    :rtype: list

    If the length of `a` is not multiple of `n`, the last chunk will be shorter than `n`.
    '''
    assert n>0, "Chunk size `n` must be larger than 0."
    chunks = [None]*int(np.ceil(len(a)/n))
    for i in range(len(a) // n):
        chunks[i] = a[i*n:(i+1)*n]
    if len(a) % n:
        chunks[-1] = a[n*(len(a) // n):]
    return chunks

