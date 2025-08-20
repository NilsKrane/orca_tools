#!python3

import numpy as np
import os
import orca_tools as ot
# import matplotlib.pyplot as plt <-- will be called later, if do_plots == True


# ================================================================================================
#                                  USER INPUT REQUIRED
# ================================================================================================
#
ORCA_PATH = r'' # Path to orca dictionary if not added to PATH variable. Use empty string '' otherwise
#
mol_path = './' # path to where .out and .gbw file can be found
mol = ot.Molecule(mol_path, orca_path=ORCA_PATH)
#
MOs = [mol.lumo] # orbital numbers, e.g. mol.homo or mol.lumo+1
spin = 0 # spin=0 for spin up and spin=1 for spin down
#
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# reload previously calculated cubes from cubin files. Empty list or False if none.
load_cubin_files = []
# adjust input <MOs> above if used to save plots (see below).
#
# ------------------------------------------------------------------
#### if load_cubin_files is not defined create cubes of perturbed wavefunctions:
####
#### Directory with vibrational dislocation
#### e.g. $vib_path/<mode_dir>/<dislocation_dir>/ <-- containing .out and .gbw files
vib_path = os.path.join(mol_path, 'vibs')
####
#### use empty list to prevent saving to cubin file.
store_cubin_files = [f'{mol.basename}_MO{MO}_s{spin}_vibs.cubin' for MO in MOs]
####
#### OPTIONAL parameters:
z_range = 1.5 # create cube data only at this height above center of molecule in Angstrom
boundary = 5. # boundaries for cubes in Angstrom
tmp_cubefile = 'tmpvib.cub' # temporary name for cube files
del_temp_cubes = True # delete temporary cubes again
deviation_threshold = 0.05 # deviation tolerance before mode will be listed as "devation"
####
####
# ================================================================================================
#
# simulation of STS maps:
cut_off_height = z_range # cut-off for Tersoff-Hamann in Angstrom
tip_height = 7. # for const height STS maps, in Angstrom above center of molecule.
#
# ------------------------------------------------------------------
#
# Plotting of results:
do_plots = True # will import matplotlib.pyplot
#
# plot maximal number of modes:
max_mode = None
# None:     plot all maps
# integer:  plot first <max_mode> maps in stack
# float:    plot all modes with energy smaller than <max_mode> in eV
#
# use empty string to prevent saving the plot to png file.
save_IETS_maps = f'{mol.basename}_IETS_maps_MO{"_".join(map(str,MOs))}_s{spin}_z{tip_height:0.1f}Ang.png'
#
#
# ================================================================================================
#                                  END OF USER INPUT
# ================================================================================================

if do_plots:
    import matplotlib.pyplot as plt

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

# ------------------------------------------------------------------

def cube_mapping(cubes: list[ot.Cube]) -> list[tuple[int,int]]:
    '''Create list of tuples for matching cubes.

    :param cubes: List containing two lists of cubes, for alpha and beta spin, respectively.
    :type cubes: list[ot.Cube]
    :return: List of tuple with indices for the matching cubes.
    :rtype: list[tuple[int,int]]
    '''    
    data = np.array([[cube.data for cube in cubes[0]],[cube.data for cube in cubes[1]]])
    data /= np.sqrt(np.sum(data**2,axis=(2,3,4),keepdims=1))
    outer = np.abs(np.einsum('iklm,jklm->ij',data[0],data[1]))
    tuple_list = list(zip(np.arange(outer.shape[0]),np.argmax(outer,axis=0)))
    return tuple_list

# ------------------------------------------------------------------

def tuple_list_complete(tuple_list: list[tuple]) -> bool:
    '''Check if tuple list is complete and no cube was doubly assigned.

    :param tuple_list: List of tuples.
    :type tuple_list: list[tuple]
    :return: Boolean if tuples are complete.
    :rtype: bool
    '''    
    set1, set2 = [set(sub) for sub in zip(*tuple_list)]
    return set1 == set2

# ================================================================================================
#                                  DO THE SCIENCE
# ================================================================================================
# either load perturbed wave function from stored cubin files or calculate from vibrational analysis
# in the end an constant height IETS map for each mode will be calculated

if load_cubin_files: # Load perturbed wavefunctions from already created cubin files.
    
    vcubes = []
    for file in load_cubin_files:
        cubes, freqs = ot.load_cubin(file)
        print(f'Loaded cubin file {file}.')
        vcubes.append(cubes)
    print('')
    vcubes = zip(*vcubes)
    vfreqs = freqs # all cubin files should contain the same frequencies

# ================================================================================================

else:  # if pertrubed wavefunctions not provided from cubin files:
    # in the following the code iterates through all directories ($mode_dir) of $vib_path
    # for each valid (see below) $mode_dir and MO a cube file with the perturbed wave function will be created
    # these perturbed wavefunctions will additionally be stored in cubin files

    # define cube dimensions (size, number of points) once and keep the same for all cubes.
    cubedims=ot.cube_dims(mol.coords, z_range=z_range+mol.center_of_mass[2], boundary=boundary)
    cube_kwargs = {'spin':spin, 'cubedims':cubedims, 'cubename':tmp_cubefile} # arguments to create cube

    # init empty output lists
    venergy=[[0,0]] # define energy of zero-phonon mode as zero
    vfreqs = [0] # zero phonon line has energy zero
    vcubes= [[]]
    for MO in MOs: # create cube for zero-phonon mode (elastic tunneling) and add to list of cubes
        vcubes[0].append(ot.Cube(mol.make_cube(MO,**cube_kwargs), deleteCubeFile=del_temp_cubes))
    
    # ------------------------------------------------------------------
    # parse through all directories found in $vib_path and consider only those which...
    # have exactly two subdirectories containing at least one .out and one .gbw file
    mode_directories = [f.path for f in os.scandir(vib_path) if f.is_dir()] # $vib_path/
    mode_directories.sort()
    valid_mode_directories = []
    for mode_dir in mode_directories:
        disloc_directories = validSubDirs(mode_dir) 
        if len(disloc_directories) == 2: 
            valid_mode_directories.append(disloc_directories)

    print(f'Found {len(valid_mode_directories)} valid directories in {vib_path}, start creating cubes:',flush=True)

    # ------------------------------------------------------------------
    # iterate through all mode directories
    mode_counter = 0
    for disloc_directories in valid_mode_directories: # $vib_path/<mode_dir>/

        a0, modes, mcubes, menergy = [], [], [[],[]], []

        for k, disloc_dir in enumerate(disloc_directories): # $vib_path/<mode_dir>/<dislocation_dir>/

            disloc_mol = ot.Molecule(disloc_dir, orca_path=ORCA_PATH) # load .out and .gbw file
            
            # get dislocation amplitude, vibrational mode and energy
            a0.append(float(getCommentVal(disloc_mol,'#VibAmp')))
            modes.append(int(getCommentVal(disloc_mol,'#VibMode')))
            menergy.append((disloc_mol.energy-mol.energy)*4/a0[k]**2)
            
            # create cube file for each molecular orbital
            for MO in MOs:
                mcubes[k].append(ot.Cube(disloc_mol.make_cube(MO,**cube_kwargs), deleteCubeFile=del_temp_cubes))

        # --------------------------------------
        if modes[0] == modes[1]: # make sure they are of the same mode
            
            # print output
            mode_counter += 1
            print(f'{modes[0]:03d}',end="..",flush=True)
            if mode_counter > 0 and mode_counter % 10 == 0:
                print('\n',end='',flush=True)
            
            # if more than one molecular orbital given, assume they might be degenerate
            # use cube_mapping to match the orbitals, as the order might have changed
            if len(MOs)>1:
                tuple_list = cube_mapping(mcubes)
                if not tuple_list_complete(tuple_list):
                    print(f'incomplete',end="..",flush=True)
            else:
                tuple_list = [(0,0)] # if only one orbital given

            # create cube for perturbed wavefunction
            for i,j in tuple_list:
                mcubes[0][i].data -= mcubes[1][j].data*phase_shift([mcubes[0][i],mcubes[1][j]]) # substract wave functions with correct sign
                mcubes[0][i].data *= 1/(a0[0]-a0[1]) # renormalize with dislocation amplitude

            # append to output lists
            vcubes.append(mcubes[0])
            vfreqs.append(mol.freqs[modes[0]])
            venergy.append(menergy)

    vfreqs = np.array(vfreqs)
    venergy = np.array(venergy).T
    print("done!\n")

    # ------------------------------------------------------------------
    # Sanity check:
    # does total energy of dislocated structures correspond to vibrational energy?
    print('Checking dislocation energy',end='... ')
    
    deviations = np.mean(np.abs(venergy[:,1:] - vfreqs[None,1:]),axis=0)/vfreqs[1:]
    if len(deviations > deviation_threshold):
        print('anharmonic/deviating modes found:')
        [print(f'    Mode {k:03d} ({freq*1e3:.1f} meV): {deviations[k]*100:.1f}%')
         for k, freq in enumerate(vfreqs[1:]) if deviations[k] > deviation_threshold];
    else:
        print('all fine.')
    print('\n',flush=True)
    
    # plot dislocation energy over vibrational mode energy
    if do_plots:
        # clean up old plot. Use this to keep number of figures in memory to a minimum
        figname = "Energy_vs_Frequency"
        figures = plt.get_fignums()
        if figname in figures: plt.close(figname)

        # create new plot
        fig = plt.figure(num=figname, figsize=(3,3))
        plt.plot(vfreqs*1e3, vfreqs*1e3, '--', color='black')
        plt.plot(vfreqs*1e3, venergy[0]*1e3, '+', color='red')
        plt.plot(vfreqs*1e3, venergy[1]*1e3, '+', color='blue')

        plt.xlabel('Vibrational Freq (meV)')
        plt.ylabel('Excitation Energy (meV)')
        plt.show()
    
    # ------------------------------------------------------------------
    # Store perturbed wavefunctions in cubin file. Can be reimported by:
    #   vcubes, vfreq = ot.load_cubin(<path>)
    if store_cubin_files:
        for i, MO in enumerate(MOs):
            store_cubin_file = store_cubin_files[i]
            cubes = [cubes[i] for cubes in vcubes]
            ot.save_cubin(store_cubin_file,cubes,vfreqs,headerline='Perturbed wavefunctions dPsi/dQ')
        print(f'Stored cubes in cubefiles:\n{store_cubin_files}\n')


# ================================================================================================
#                                  Simulate STS Maps
# ================================================================================================
#
# Simulate constant height STS maps for each mode.
print(f'Simulating STS maps for tip height: {tip_height}Å.\n')
stack = []
for cubes in vcubes:
    STS_maps = [cube.sim_STS(tip_height,plane=cut_off_height) for cube in cubes]
    stack.append(np.sum(np.array(STS_maps),axis=0))
stack = np.array(stack)

# ------------------------------------------------------------------
# Print modes with highest intensity.
intensities = np.max(stack[1:],axis=(1,2))/np.max(stack[0])
modes = np.arange(len(intensities))
sorted_modes = sorted(zip(intensities,modes,vfreqs[1:]),reverse=True)
print(f'Modes with highest relative intensity:')
[print(f'    Mode {k:03d} ({f*1e3:5.1f} meV): {i:6.1%}')
  for i,k,f in sorted_modes if i > 0.01];
print('\n',flush=True)


# ------------------------------------------------------------------
# Plot the Maps
if do_plots:
    print('Plotting IETS maps...')
    # ------------------------------------------------------------------
    # some specs for plotting

    size = 2 # size of subfigures in inches

    # number of modes to plot
    if isinstance(max_mode,type(None)):
        nmaps = len(stack)
    elif isinstance(max_mode,int):
        nmaps = min(max_mode,len(stack))
    elif isinstance(max_mode,float):
        nmaps = min(np.searchsorted(mol.freqs,max_mode),len(stack))

    ncol = 4 # number of columns for figure grid
    nrow = int(np.ceil(nmaps/ncol)) #  number of rows
    
    vmax = np.max(stack[1:,...]) # max value for color scale, e.g. display all vib maps with same colorscale
    vmax_elastic = np.max(stack[0])
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
            title = "elastic\nrel. Int.: 1.0"
        else:
            axes[i].imshow(stack[i,...],cmap='gray',vmin=0,vmax=vmax)
            rel_int = np.max(stack[i])/vmax_elastic
            title = f'#{i-1}: {vfreqs[i]*1000:0.1f}meV\nrel. Int.: {rel_int:.3f}'

        axes[i].tick_params(labelbottom=False, labelleft=False, size = 0)
        axes[i].set_title(title, fontsize='small')
    
    if save_IETS_maps:
        plt.savefig(save_IETS_maps, format="png", dpi=300)  # Save as PNG
        print(f'Saved maps as: {save_IETS_maps}')
    plt.show()

