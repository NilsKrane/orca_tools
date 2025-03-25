import numpy as np
import os
from .cube_utils import Cube

def save_cubin(filename: str, cubes: list, x=None, headerline=""):
    cube = cubes[0]

    assert cube.nMO == 1, "Error: More than one MO in cube!"
        
    if isinstance(x,np.ndarray):
        assert len(cubes) == len(x), "List of cubes and x-values are not of same length."
    elif type(x) == type(None):
        x = np.zeros((len(cubes)))
    else:
        raise ValueError("x-values must be ndarray or None")
    
    assert isinstance(headerline,str), "Parameter 'headerline' has to be a string variable."
    
    header = f"Number of Cubes = {len(cubes)}\n"
    header += headerline.replace("\n","")+"\n"
    header += cube.parameters
    for atom in cube.atoms:
        header += f'{atom["element"]:>5}'
        header += '{:>12}'.format(f'{atom["charge"]:.6f}')
        for i in range(3):
            header += '{:>12}'.format(f'{atom["coords"][i]/0.529177249:.6f}')
        header += f'\n'
    header += f'{int(1):>5}{int(cube.vecMO[0]):>5}\n'

    with open(filename, "wb") as b:
        b.write(bytes(header,"utf-8"))
        
    with open(filename, "ab") as b:
        x.tofile(b)
        for cube in cubes:
            cube.data.flatten().tofile(b)
        
    return filename

def load_cubin(binfile: str) -> (list, np.array):
    header, pointer_pos = Cube.load_cube_header(binfile)

    try:
        nCubes = int(header.split("\n")[0].split("Number of Cubes =")[1])
    except:
        print("Error: Could not parse first line. Number of cubes expected.")
        return None

    with open(binfile, "br") as b:
        b.seek(pointer_pos)
        buffer = b.read(nCubes*8)
        pointer_pos = b.tell()
    x = np.frombuffer(buffer)

    cubes = []
    for i in range(nCubes):
        cube = Cube(header)
        with open(binfile, "br") as b:
            b.seek(pointer_pos)
            buffer = b.read(int(cube.n1*cube.n2*cube.n3)*8)
            pointer_pos = b.tell()
        cube.save_cube_data(np.frombuffer(buffer))
        cubes.append(cube)
    return cubes, x
