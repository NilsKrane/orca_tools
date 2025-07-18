#!python3

import numpy as np
import os
import orca_tools as ot

# import matplotlib.pyplot as plt <-- will be called if plot_IETS_maps == True

# ================================================================================================
#                                  USER INPUT REQUIRED
# ================================================================================================
 
ORCA_PATH = '' # Path to orca dictionary if not added to PATH variable. Use empty string '' otherwise

# ------------------------------------------------------------------
# output and gbw file of relaxed/optimized molecule

mol_path = './'
mol_gbw = os.path.join(mol_path, "mol.gbw")
mol_out = os.path.join(mol_path, "mol.out")
mol = ot.Molecule([mol_out,mol_gbw], orca_path=ORCA_PATH)

# ------------------------------------------------------------------
# Directory with vibrational dislocation
# e.g. $vib_path/<mode_dir>/<dislocation_dir>/ <-- containing .out and .gbw files

vib_path = os.path.join(mol_path, 'vibs')

# ------------------------------------------------------------------
# which MOs/Spins to look at

MO = mol.somo # orbital number, e.g. mol.somo (UKS) or mol.homo (RKS)
spin = 0 # spin=0 for spin up and spin=1 for spin down
tip_height = 7. # for const height STS maps, in Angstrom above center of molecule.

# ------------------------------------------------------------------
# OPTIONAL parameters:

z_range = 1.5 # cut_off height for Tersoff-Hamann in Angstrom
boundary=5. # boundaries for cubes in Angstrom
tmp_cubefile = 'tmpvib.cub' # temporary name for cube files
del_temp_cubes = True # delete temporary cubes


# ------------------------------------------------------------------
# Output:

plot_IETS_maps = True
save_IETS_plots = f'{mol.basename}_IETS_maps_MO{MO}_s{spin}_z{tip_height:0.1f}Ang.png' # use empty string to prevent saving the plot to png file.

# ================================================================================================
#                                  SOME HELPER FUNCTIONS
# ================================================================================================

def isDirValid(dir: os.PathLike) -> bool:
    '''Checks if directory contains .gbw and .out file

    :param dir: Directory
    :type dir: os.PathLike
    :return: True if .out and .gbw file were found
    :rtype: bool
    '''        
    file_list = [f.path for f in os.scandir(dir) if f.is_file()]
    endings = [os.path.splitext(f)[1] for f in file_list]
    return '.out' in endings and '.gbw' in endings

# ------------------------------------------------------------------

def validSubDirs(dir: os.PathLike) -> list[os.PathLike]:
    '''List all subdirectories, which contain an .out and a .gbw file.

    :param dir: Directory
    :type dir: os.PathLike
    :return: List of subdirectories containing .out and .gbw file.
    :rtype: list[os.PathLike]
    '''    
    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    return [subdir for subdir in subdirs if isDirValid(subdir)]

# ------------------------------------------------------------------

def getFiles(dir: os.PathLike) -> list[os.PathLike]:
    '''Get list containing paths to .out and .gbw file.

    :param dir: Directory
    :type dir: os.PathLike
    :return: [out_file, gbw_file]
    :rtype: list[os.PathLike]
    '''    
    file_list = [f.path for f in os.scandir(dir) if f.is_file()]
    out = [f for f in file_list if f.endswith('.out')]
    gbw = [f for f in file_list if f.endswith('.gbw')]
    return [out[0],gbw[0]]

# ------------------------------------------------------------------

def getCommentVal(molecule: ot.Molecule, keyword: str) -> str:
    '''Return Value from comment in input of Molecule instance.

    :param molecule: Instance of orca_tools.Molecule
    :type molecule: ot.Molecule
    :param keyword: Keyword to find, e.g. `#VibMode`
    :type keyword: str
    :return: First string following keyword, until first whitespace.
    :rtype: int
    '''
    try:
        return [line.split()[-1] for line in molecule.input_str.split('\n')
                if line.startswith(keyword)][0]
    except IndexError:
        return ''

# ------------------------------------------------------------------

def phase_shift(cubes: list[ot.Cube]) -> int:
    '''Get phase shift (aka sign) of two similar cubes.

    :param cubes: List containing two orca_tool.Cubes instances
    :type cubes: list[ot.Cube]
    :return: 1 if they are in phase, -1 if sign is flipped.
    :rtype: int
    '''    
    maxpos = np.unravel_index(cubes[0].data.argmax(), cubes[0].data.shape)
    return np.sign(cubes[0].data[maxpos])*np.sign(cubes[1].data[maxpos])

# ================================================================================================
#                                  DO THE SCIENCE
# ================================================================================================
# in the following the code iterates through all directories ($mode_dir) of $vib_path
# for each valid (see below) $mode_dir a cube file with the perturbed wave function will be created
# in the end an constant height IETS map for each mode will be calculated.

# define cube dimensions (size, number of points) once and keep the same for all cubes.
cubedims=ot.cube_dims(mol.coords, z_range=z_range+mol.center_of_mass[2], boundary=boundary)
cube_kwargs = {'spin':spin, 'cubedims':cubedims, 'cubename':tmp_cubefile} # arguments to create cube

# create cube for zero-phonon mode (elastic tunneling) and add to list of cubes
vcubes= [ot.Cube(mol.make_cube(MO,**cube_kwargs), deleteCubeFile=del_temp_cubes)]
vfreqs = [0] # zero phonon line has energy zero

# ------------------------------------------------------------------
# parse through all directories found in $vib_path and consider only those which...
# a) have at least two subdirectories
# b) exactly two of those subdirectories contain at least one .out and one .gbw file

mode_directories = [f.path for f in os.scandir(vib_path) if f.is_dir()] # $vib_path/
mode_directories.sort()

print(f'Found {len(mode_directories)} directories in {vib_path}, start creating cubes:',flush=True)

for mode_dir in mode_directories: # $vib_path/<mode_dir>/

    disloc_directories = validSubDirs(mode_dir) # list of valid subdirectories (see above)
    if len(disloc_directories) != 2:
        continue # skip this subdirectory
    
    a0, modes, mcubes = [], [], []
    for disloc_dir in disloc_directories: # $vib_path/<mode_dir>/<dislocation_dir>/

        disloc_mol = ot.Molecule(getFiles(disloc_dir), orca_path=ORCA_PATH) # load .out and .gbw file
        
        # get dislocation amplitude, vibrational mode and create a cube file
        a0.append(float(getCommentVal(disloc_mol,'#VibAmp')))
        modes.append(int(getCommentVal(disloc_mol,'#VibMode')))
        mcubes.append(ot.Cube(disloc_mol.make_cube(MO, **cube_kwargs), deleteCubeFile=del_temp_cubes))
    
    if modes[0] == modes[1]: # make sure they are of the same mode

        mcubes[0].data -= mcubes[1].data*phase_shift(mcubes) # substract wave functions with correct sign
        mcubes[0].data *= 1/(a0[0]-a0[1]) # renormalize with dislocation amplitude
        
        vcubes.append(mcubes[0])
        vfreqs.append(mol.freqs[modes[0]])
        print(modes[0],end="..",flush=True)

vfreqs = np.array(vfreqs)
print("done!")

# ------------------------------------------------------------------
# Simulate constant height STS maps for each mode.
stack = np.array([cube.sim_STS(tip_height) for cube in vcubes])


# ================================================================================================
#                                  PLOT THE MAPS
# ================================================================================================
if plot_IETS_maps:
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # some specs for plotting

    size = 2 # size of subfigures in inches

    nmaps = len(stack) # number of modes to plot, use len(stack) for all modes

    ncol = 4 # number of columns for figure grid
    nrow = int(np.ceil(nmaps/ncol)) #  number of rows
    
    vmax = np.max(stack[1:,...]) # max value for color scale, e.g. display all vib maps with same colorscale

    # ------------------------------------------------------------------

    # clean up old plot. Use this to keep number of figures in memory to a minimum
    figname = "IETS_maps_plot"
    figures = plt.get_fignums()
    if figname in figures: plt.close(figname)

    # create new plot
    fig = plt.figure(num=figname, figsize=(ncol*size,nrow*(size+0.2)))
    axes = [None]*nmaps
    for i in range(nmaps): # for each map...

        # create subplot
        row = int(np.floor(i/ncol))
        col = i % ncol
        axes[i] = plt.subplot2grid((nrow,ncol),(row,col))

        if i == 0: # if elastic map, do not use colorscale vmax
            axes[i].imshow(stack[i,...],cmap='gray',vmin=0)
            title = "elastic"
        else:
            axes[i].imshow(stack[i,...],cmap='gray',vmin=0,vmax=vmax)
            title = f'#{i-1}: {vfreqs[i]*1000:0.1f}meV'

        axes[i].tick_params(labelbottom=False, labelleft=False, size = 0)
        axes[i].set_title(title, fontsize='small')
    
    if save_IETS_plots:
        plt.savefig(save_IETS_plots, format="png", dpi=300)  # Save as PNG
    plt.show()

