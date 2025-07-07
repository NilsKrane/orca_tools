import os
import numpy as np
from .cube_utils import Cube
from .orca_utils import element_to_number
from .orca_utils import bohr2ang

def save_cubin(filename: str | os.PathLike, cubes: list[Cube], x: list | np.ndarray=None, headerline: str="")  -> str:
    '''Store data from list of cubes in a binary *.cubin file.

    The header will be created from first cube in list, for all other cubes the header data will not be stored.
    
    :param filename: Path to cubin file to be created.
    :type filename: str | os.PathLike
    :param cubes: List of `Cube` class objects.
    :type cubes: list[Cube]
    :param x: List or array of values to be stored with the cubes.
    :type x: list | np.ndarray, optional
    :param headerline: Header to be written in second line of cubin file, defaults to ""
    :type headerline: str, optional
    :return: Path of created file.
    :rtype: str
    '''    
    cube = cubes[0]
    assert cube.nMO == 1, "Error: More than one molecular orbital in cube!"

    # make sure x is np.array or can translated into one
    if type(x) == type(None):
        x = np.zeros((len(cubes)))
    else:
        x = np.array(x)
        assert len(cubes) == len(x), "List of cubes and x-values are not of same length."
    
    # create header
    header = f"Number of Cubes = {len(cubes)}\n"
    header += headerline.replace("\n"," ").replace("\r","")+"\n"
    header += cube.parameters
    for atom in cube.atoms:
        header += f'{element_to_number(atom["element"]):>5}'
        header += '{:>12}'.format(f'{atom["nuclearcharge"]:.6f}')
        for i in range(3):
            header += '{:>12}'.format(f'{atom["coords"][i]/bohr2ang:.6f}')
        header += f'\n'
    header += f'{int(1):>5}{int(cube.vecMO[0]):>5}\n'

    # write header
    with open(filename, "wb") as b:
        b.write(bytes(header,"utf-8"))
        
    # write x and data into file
    with open(filename, "ab") as b:
        x.tofile(b)
        for cube in cubes:
            cube.data.flatten().tofile(b)
        
    return filename

# -------------------------------------------------------------------------------------------------

def load_cubin(binfile: str | os.PathLike) -> tuple[list[Cube], np.ndarray]:
    '''Load cubin file.

    :param binfile: Path to cubin file.
    :type binfile: str | os.PathLike
    :return: List of `Cube` objects and 1d array of parameter values.
    :rtype: tuple[list[Cube], np.ndarray]
    '''    

    # load header and find number of cubes in file
    header, pointer_pos = Cube.load_cube_header(binfile)
    try:
        nCubes = int(header.split("\n")[0].split("Number of Cubes =")[1])
    except:
        print("Error: Could not parse first line. Number of cubes expected.")
        return None

    # read parameter array
    with open(binfile, "br") as b:
        b.seek(pointer_pos)
        buffer = b.read(nCubes*8)
        pointer_pos = b.tell()
    x = np.frombuffer(buffer)

    # create empty cubes from header and fill with data from file.
    cubes = [None]*nCubes
    for i in range(nCubes):
        cube = Cube(header)
        with open(binfile, "br") as b:
            b.seek(pointer_pos)
            buffer = b.read(int(cube.n1*cube.n2*cube.n3)*8)
            pointer_pos = b.tell()
        cube.data = cube.reshape_data(np.frombuffer(buffer))
        cubes[i] = cube
        
    return cubes, x
