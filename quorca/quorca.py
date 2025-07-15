#!python3

import os
import argparse
import numpy as np
import subprocess
import orca_tools as ot

# ==============================================================
# System specific parameter
ORCA_PATH = '' # Path to orca dictionary if not added to PATH variable. Use empty string '' otherwise
PYTHON_CALL = 'python3' # How to run this file
# Some default values
NPROCS = 5
MAXCORE = 2000
# SLURM
SLURM_ENV = 'source ~/miniconda3/etc/profile.d/conda.sh\nconda activate env01\n'
# ==============================================================


# ------------------------------------------------------------------------ 

def ext() -> str:
    '''Get OS specific file extension for ORCA binaries.

    :return: Return '.exe' if OS is windows and '' otherwise.
    :rtype: str
    '''    
    if os.name == 'posix':
        return ''
    elif os.name == 'nt':
        return '.exe'
    
# ------------------------------------------------------------------------ 

def odd_number_of_electrons(xyz: str | os.PathLike, charge: int = 0) -> bool:
    '''Calculate number of electrons for molecule defined in `*.xyz` file.

    :param xyz: Path to `*.xyz` file.
    :type xyz: str | os.PathLike
    :param charge: Charge of molecule with positive (negative) value corresponding to removal (addition) of electron, defaults to 0.
    :type charge: int, optional
    :return: True if number of electrons is odd, False otherwise.
    :rtype: bool
    '''    
    with open(xyz, 'r') as f:
        content = f.read()
    
    content = content.split('\n')
    nAtoms = int(content[0])
    
    electrons = -charge
    for atom in content[2:nAtoms+2]:
        electrons+=ot.element_to_number(atom.split()[0])
    return bool(electrons % 2)

# --------------------------------------------------------------------

def find_xyz_file(xyz: str | os.PathLike='') -> str:
    '''Check if given xyz file exists and look for alternative otherwise.

    :param xyz: Path to xyz file.
    :type xyz: str | os.PathLike
    :return: Path to xyz file if one could be found.
    :rtype: str

    This function checks if the given xyz file exists. If none is found it will first look for `input.xyz` and, if that does not exist, take the first `*.xyz` file.
    '''    

    if os.path.isfile(xyz):
        return xyz
    
    elif os.path.isfile('input.xyz'):
        print('No valid xyz file defined. Found file input.xyz and will proceed.')
        return 'input.xyz'
        
    else:
        xyzs = [f for f in os.listdir() if f.endswith('.xyz')]
        if len(xyzs):
            print(f'No valid xyz file defined. Found file {xyzs[0]} and will proceed.')
            return xyzs[0]
        else:
            return None
    
# ------------------------------------------------------------------------ 

def make_input_dict(args: argparse.Namespace) -> dict:
    '''Create dictionary containing all relevant input values to create an input file for ORCA.

    :param args: Namespace object from argparse.
    :type args: argparse.Namespace
    :return: Dictionary
    :rtype: dict
    '''    

    # ----- KEYWORDS -----
    if args.scf.upper() == 'TIGHT':
        args.scf = 'TIGHTSCF'
    elif args.scf.upper() == 'VERY' or args.scf.upper() == 'VERYTIGHT':
        args.scf = 'VERYTIGHTSCF'

    if args.opt == None:
        args.opt = ''
    elif args.opt.upper() == 'TIGHT':
        args.opt = 'TIGHTOPT'
    elif args.opt.upper() == 'VERY' or args.opt.upper() == 'VERYTIGHT':
        args.opt = 'VERYTIGHTOPT'

    inp_dict = {'!':[]}
    if args.unrestricted: inp_dict['!'].append('UKS')
    inp_dict['!'].append(args.functional.upper())
    inp_dict['!'].append(args.basisset)
    
    if args.scf: args.keywords.add(args.scf.upper())
    if args.opt: args.keywords.add(args.opt.upper())
    if args.freq: args.keywords.add('FREQ')
    inp_dict['!'] += list(args.keywords)
    
    # ----- COORDS -----
    args.xyz = find_xyz_file(args.xyz)
    if args.xyz == None:
        print('No xyz file found. Abort!')
        quit()

    if args.multiplicity == 0:
        args.multiplicity = 1 + int(odd_number_of_electrons(args.xyz, args.charge))
    elif args.multiplicity % 2 == odd_number_of_electrons(args.xyz, args.charge):
        print('Multiplicity not valid for given number of electrons. Abort!')
        quit()

    inp_dict['*xyzfile'] = [args.charge, args.multiplicity, args.xyz]
    
    # ----- SINGLE VALUE PARAMETER -----
    if not len(args.basename):
        args.basename = '.'.join(args.filename.split('.')[:-1])
    inp_dict['base'] = f'\"{args.basename}\"'

    inp_dict['maxcore'] = args.maxcore
    
    # ----- PARAMETER BLOCKS -----
        # -- pal --
    inp_dict['pal'] = {'nprocs':args.nprocs}
    
        # -- scf --
    if len(args.gbw_input):
      #  inp_dict['scf']={'Guess':'MORead','MOInp':f'\"{args.gbw_input}\"'}
        add_subkeys(inp_dict,'scf',{'Guess':'MORead','MOInp':f'\"{args.gbw_input}\"'})
    if args.multiplicity == 1 and args.unrestricted:
        add_subkeys(inp_dict,'scf',{'GuessMix':60})

        # -- output --
    if not args.omit_MO_basis:
        #inp_dict['output'] = {'Print[P_Basis]':2,'Print[P_MOs]':1}
        add_subkeys(inp_dict,'output',{'Print[P_Basis]':2,'Print[P_MOs]':1})
        
        # -- tddft --
    if args.tddft:
        add_subkeys(inp_dict,'tddft',{'nroots':10,'maxdim':10,'TDA':True,'doNTO':False,'maxcore':args.maxcore})
        if args.opt or args.freq:
            add_subkeys(inp_dict,'tddft',{'iroot':1,'followiroot':True})
        
    return inp_dict

# ------------------------------------------------------------------------ 

def write_input_file(inp_dict: dict, filename: str | os.PathLike) -> str:
    '''Write input dictionary to file.

    :param inp_dict: Input dictionary created by `make_input_dict()` or `make_input_dict()`.
    :type inp_dict: dict
    :param filename: Filename of input file to be created.
    :type filename: str
    :return: Path to create inputfile
    :rtype: str
    '''    
    inp_dict = inp_dict.copy()
    
    def key2line(dic: dict, key: str) -> str:
    #    '''Translate input entries to strings.'''
    #    try: 
    #        if not len(dic[key]):
    #            return '' # empty set or dictionary will be ignored
    #    except TypeError:
    #        pass # value might be a number
    #    out = f'%{key}'
        out=''
        if isinstance(dic[key],dict):
            if len(dic[key]):
                out = f'%{key}\n\t'+'\n\t'.join([subkey+' '+str(inp_dict[key][subkey]) for subkey in inp_dict[key].keys()])+'\nend'
        elif isinstance(dic[key],set):
            if len(dic[key]):
                out = f'%{key}\n\t'+'\n\t'.join([str(val) for val in inp_dict[key]])+'\nend'
        else:
            out += f'%{key} {dic[key]}'
        return out

    # create input list with all blocks    
    inputs = ['!'+' '.join(inp_dict.pop('!'))]
    xyz_line = '*xyzfile '+' '.join([str(x) for x in inp_dict.pop('*xyzfile')])+'\n'
    
    if '#' in inp_dict:
        inputs.append('#'+'\n#'.join(inp_dict.pop('#')))

    inputs += [key2line(inp_dict,key) for key in inp_dict.keys()]
    
    # assemble complete input string
    out_str = '\n\n'.join([x for x in inputs if len(x)])+'\n\n'+xyz_line+'\n'
    
    # write to file
    filename = filename.removesuffix('.inp')+'.inp'
    with open(filename, 'wb') as f:
        f.write(bytes(out_str,'utf-8'))
    
    return filename

# ------------------------------------------------------------------------ 

def write_slurm_file(args: argparse.Namespace, filename: str='') -> str:
    '''Create bash script to be used as slurm input.

    :param args: Namespace object from argparse.
    :type args: argparse.Namespace
    :param filename: Filename of file to be created.
    :type filename: str, optional
    :return: Path to created script.
    :rtype: str
    '''

    out_str = ''
    out_str += '#!/bin/bash\n'
    out_str += f'#SBATCH --job-name={args.slurm}\n'
    out_str += f'#SBATCH --nodes=1\n'
    out_str += f'#SBATCH --ntasks={args.nprocs}\n'
    out_str += f'#SBATCH --cpus-per-task=1\n'
    out_str += f'#SBATCH --mem={int(args.maxcore*args.nprocs/1024*1.15)}GB\n'
    out_str += f'#SBATCH --time={args.walltime}\n'
    out_str += '\n'
    out_str += 'echo "Running on $SLURM_CPUS_ON_NODE CPU cores"\n'

    if SLURM_ENV:
        out_str += '\n'
        out_str += SLURM_ENV

    out_str += '\n'
    out_str += f'{rebuilt_quorca_cmd(args)}\n'
    
    if not len(filename): filename = args.basename+".sh"
    with open(filename, 'bw') as f:
        f.write(bytes(out_str,'utf-8'))
    
    return filename

# ------------------------------------------------------------------------ 

def slurm_args_from_input_dict(args: argparse.Namespace, input_dict: dict):
    '''Parse `input_dict` and add arguments, required for slurm script, to `args`.

    :param args: Namespace object from argparse.
    :type args: argparse.Namespace
    :param inp_dict: Input dictionary created by `make_input_dict()` or `make_input_dict()`.
    :type input_dict: dict
    '''    
    args.nprocs = int(input_dict['pal']['nprocs'])
    args.maxcore = int(input_dict['maxcore'])
    
# ------------------------------------------------------------------------ 

def rebuilt_quorca_cmd(args: argparse.Namespace) -> str:
    '''Reconstruct initial quorca input commad to run calculations, e.g. from slurm script.

    :param args: Namespace object from argparse.
    :type args: argparse.Namespace
    :return: quorca input command.
    :rtype: str

    This function recreates all flags required to run calculation via quorca using an existing input or output file.
    '''    
    cmd_str = f'{PYTHON_CALL} {os.path.abspath(__file__)}'
    if args.input:
        cmd_str += f' -i {args.input}'
    if args.out:
        cmd_str += f' -o {args.out}'
    if args.vibs:
        cmd_str += f' -v {args.vibs}'
    if args.maxfreq:
        cmd_str += f' --maxfreq {args.maxfreq}'
    if args.minfreq:
        cmd_str += f' --minfreq {args.minfreq}'
    if args.maxmode:
        cmd_str += f' --maxmode {args.maxmode}'
    if args.minmode:
        cmd_str += f' --minmode {args.minmode}'
    return cmd_str


# ------------------------------------------------------------------------ 

def parse_inp(input: str | os.PathLike) -> dict:
    '''Parse input string or file into input dictionary.

    :param input: Path to input file or string containing content of input file.
    :type input: str | os.PathLike
    :return: Input dictionary.
    :rtype: dict
    '''

    # load content of file if input is path to file.
    if os.path.isfile(input):
        with open(input,'r') as f:
            content = f.read()
    elif isinstance(input,str):
        content = input
    else:
        return None
    
    if not len(content):
        return None
        
    lines = list(filter(None,content.split("\n"))) # remove empty lines
   
    inp_dict = {}
    
    inp_dict['!'] = [keyword for line in lines if line[0] == '!' for keyword in line[1:].split()]
    inp_dict['*xyzfile'] = [[int(line.split()[1]),int(line.split()[2]),line.split()[3]]
                            for line in lines if line.split()[0] == '*xyzfile'][0]
    
    lines = [line for line in lines if line[0] not in ['!','*']] # remove keyword line and xzy line
    
    key = ''
    for line in lines:
        if line[0] == '#': # commment line
            add_subkeys(inp_dict, '#', {line[1:]})
        
        elif line[0] == '%': # start of key value block
            key = line.split()[0][1:].lower()
            if len(line.split()) == 2: # single value key, like '%maxcore 2000' or '%base "mol"'
                inp_dict[key] = line.split()[1]
                key = ''
            elif len(line.split()) == 1: # start of key value block, init set/dict
                inp_dict[key] = {}
        elif line.lower() == 'end': # end of key value block
            key = ''
        elif len(line.split()) < 3 and len(key): # line has 1 or 2 arguments and is within a key value block
            try:
                add_subkeys(inp_dict, key, {line.split()[0].lower():line.split()[1]})
            except IndexError:
                add_subkeys(inp_dict, key, {line.split()[0].lower()})
            
    return inp_dict

# ------------------------------------------------------------------------ 

def basename_from_input(filename: str | os.PathLike, cwd: str | os.PathLike='') -> str:
    '''Get basename of ORCA calculations from input file.

    :param filename: Path to input file
    :type filename: str | os.PathLike
    :param cwd: Directory containing input file, defaults to ''
    :type cwd: str | os.PathLike, optional
    :return: basename
    :rtype: str
    '''

    assert os.path.isfile(os.path.join(cwd,filename)), 'Can not find input file'
    
    with open(os.path.join(cwd,filename), 'r') as f:
        content = f.read()
        
    if content.find('%base') < 0:
        return filename.removesuffix('.inp')
    else:
        return content.split('%base')[-1].split()[0].replace('"','').replace("'",'')

# ------------------------------------------------------------------------ 

def basename_from_output(filename: str | os.PathLike, cwd='') -> str:
    '''Get basename from ORCA output file, by parsing the input section.

    :param filename: Path to output file
    :type filename: str | os.PathLike
    :param cwd: Directory containing output file, defaults to ''
    :type cwd: str, optional
    :return: basename
    :rtype: str
    '''    

    assert os.path.isfile(os.path.join(cwd,filename)), 'Cannot find output file'
    
    with open(os.path.join(cwd,filename), 'r') as f:
        content = f.read()
        
    if content.find('%base') < 0:
        marker = 'INPUT FILE\n================================================================================\nNAME ='
        return content.split(marker)[-1].split()[0].removesuffix('inp')
    else:
        return content.split('%base')[-1].split()[0].replace('"','').replace("'",'')

# ------------------------------------------------------------------------ 

def input_from_out(filename: str | os.PathLike) -> str:
    '''Get input string from output file.

    :param filename: Path to output file
    :type filename: str | os.PathLike
    :return: Input string
    :rtype: str
    '''    
    assert os.path.isfile(filename), 'Output file not found.'
    
    with open(filename,'r') as f:
        content = f.read()

    try:
        input_lines = content.split("INPUT FILE\n"+"="*80+"\n")[1].split("****END OF INPUT****")[0].split("\n")[1:-1]
        input_lines = [line[6:] for line in input_lines]
        return "\n".join(input_lines)
    except IndexError:
        return ''

# ------------------------------------------------------------------------ 

def remove_subkeys(dic: dict, key: str, subkeys: list):
    '''Remove subkey entry from input dictionary if present.

    :param dic: Input dictionary created by `make_input_dict()` or `make_input_dict()`.
    :type dic: dict
    :param key: Search for subkey in `dic[key]`
    :type key: str
    :param subkeys: List of subkeys to be removed from `dic[key]`.
    :type subkeys: list
    '''    
    
    try: # check if key is in dic and get correct spelling (case-sensitve)
        key = [k for k in dic if k.lower() == key.lower()][0]
    except IndexError:
        return 0
    
    if isinstance(dic[key],dict):
        for subkey in subkeys:
            try: # check if subkey is in dic[key] and get correct spelling (case-sensitve)
                subkey = [k for k in dic[key] if k.lower() == subkey.lower()][0]
                dic[key].pop(subkey, None)
            except IndexError:
                pass

    elif isinstance(dic[key],list):
        dic[key] = [k for k in dic[key] if k.lower() not in list(map(str.lower, subkeys))]
    
    elif isinstance(dic[key],set):
        for element in subkeys:
            try: # check if subkey is in dic[key] and get correct spelling (case-sensitve)
                element = [e for e in dic[key] if e.lower() == element.lower()][0]
                dic[key].discard(element)
            except IndexError:
                pass


# ------------------------------------------------------------------------ 

def add_subkeys(dic: dict, key: str, subdict: dict | set):
    '''Add set or dictionary entries to `dic[key]`.

    :param dic: Input dictionary created by `make_input_dict()` or `make_input_dict()`.
    :type dic: dict
    :param key: Search for subkey in `dic[key]`
    :type key: str
    :param subdict: Set or dictionary with values to be added to `dic[key]`.
    :type subdict: dict | set
    '''    
    if not len(subdict): # subdict empty, do nothing
        return 0

    try: # check if key is in dic and get correct spelling (case-sensitve)
        key = [k for k in dic if k.lower() == key.lower()][0]            
    except IndexError:
        dic[key] = subdict # key did not exist in dic, create new key-value pair
    
    else:
        if isinstance(subdict, dict):
            for subkey in subdict:
                try:
                    oldkey = [k for k in dic[key] if k.lower() == subkey.lower()][0]
                    dic[key][oldkey] = subdict[subkey]
                except IndexError:
                    dic[key].update({subkey:subdict[subkey]})                

        elif isinstance(subdict,set):
            if len(dic[key]): # a set already exists
                dic[key].update(subdict)
            else: # dic[key] is empty, create new set
                dic[key] = subdict

# ------------------------------------------------------------------------ 

def run_orca(inputfile: str | os.PathLike, outputfile: str | os.PathLike='', cwd: str | os.PathLike='./', quiet: bool=False) -> str:
    '''Run ORCA calculation using `inputfile`.

    :param inputfile: Path to input file for ORCA calculation.
    :type inputfile: str | os.PathLike
    :param outputfile: Path to file to write ORCA output. If not defined: write to $base.out
    :type outputfile: str | os.PathLike, optional
    :param cwd: Path to directory in which the calculation will run, defaults to './'
    :type cwd: str | os.PathLike, optional
    :param quiet: Supress printing to stdout, defaults to False
    :type quiet: bool, optional
    :return: Path to output file.
    :rtype: str
    '''        
    if not outputfile:
        outputfile = basename_from_input(inputfile,cwd)+'.out'  
    outputfile = os.path.join(cwd,outputfile)
    
    if not cwd: # make sure cwd is not empty for subprocess.run function
        cwd = './'
    
    if not quiet:
        print('Start DFT calculation...', flush=True)
    with open(outputfile,'w') as outfile:
            subprocess.run([os.path.join(ORCA_PATH,'orca'+ext()), inputfile], cwd=cwd, stdout=outfile)
    
    mol = ot.Molecule(outputfile)
            
    if mol.success:
        if not quiet:
            print('... finished successfully.', flush=True)
        return outputfile
    else:
        if not quiet:
            print('...something went wrong.', flush=True)
        return None

# ------------------------------------------------------------------------ 

def run_vibrational_analysis(args: argparse.Namespace, gbw_file: str | os.PathLike=''):
    '''Run vibrational analysis using ORCA. 

    :param args: Namespace object from argparse.
    :type args: argparse.Namespace
    :param gbw_file: Path to gbw file to be used for initial guess. 
    :type gbw_file: str | os.PathLike, optional
    '''

    if type(args.out) == type(None):
        print(f'No output file provided.')
        return 0

    mol = ot.Molecule([args.out])
    
    if not mol.success:
        print(f'Calculation stored in {args.out} was not successful.')
        return 0
    elif type(mol.freqs) == type(None):
        print(f'Provided outfile {args.out} does not include vibrational normalmodes and energies.')
        return 0

    # ------------ Which normal modes should be included? ------------
    if type(args.maxmode) == type(None):
        maxmode = len(mol.freqs)
    else:
        maxmode = min(len(mol.freqs),args.maxmode)
    
    if type(args.maxfreq) != type(None):
        maxmode = min(np.searchsorted(mol.freqs,args.maxfreq),maxmode)

    minmode = args.minmode
    if type(args.minfreq) != type(None):
        minmode = max(np.searchsorted(mol.freqs,args.minfreq),minmode)

    modes = range(minmode,maxmode)    


    # ------------ Create and Modify Input Dictionary ------------
    vib_dict = parse_inp(input_from_out(args.out))

    VPATH = "./vibs/"
    GBWFILENAME = 'input.gbw'
    XYZFILENAME = 'input.xyz'

    BASE = 'vibs'
    INPUTFILE = BASE+'.inp'
    OUTFILE = BASE+'.out'
    BIBFILE = BASE+'.bibtex'
            
    remove_subkeys(vib_dict,'!', ['FREQ', 'OPT', 'TIGHTOPT', 'VERYTIGHTOPT', 'NORMALOPT', 'LOOSEOPT', 'MOREAD'])
    remove_subkeys(vib_dict,'output', ['print[p_basis]','print[p_mos]'])
    vib_dict.pop('moinp',None)

    if os.path.isfile(gbw_file):
        add_subkeys(vib_dict,'scf',{'Guess':'MORead','MOInp':f'\"{GBWFILENAME}\"'})
        remove_subkeys(vib_dict,'scf',['GuessMix'])
    
    vib_dict['base'] = '\"'+BASE+'\"'
    vib_dict['*xyzfile'][2] = XYZFILENAME

    comments = list(vib_dict.pop('#',{}))

    # ------------ Start Calculations ------------

    print(f'Starting vibrational caclulations with {len(modes)} modes...', flush=True)

    if not os.path.exists(VPATH): 
            os.makedirs(VPATH)

    for mode in modes:
        mode_path = VPATH+f"mode{mode:03d}"
        if not os.path.exists(mode_path): 
            os.makedirs(mode_path) # create directory for mode
        
        for sign in range(2):
            a0 = [args.vibs,-args.vibs][sign] # dislocation in units of HWHM of gaussian distribution of zero-phonon mode
            sign_str = ["pos","neg"][sign]
            ipath = mode_path+f"/{sign_str}"
            if not os.path.exists(ipath): 
                os.makedirs(ipath) # create directory for dislocation
            
            print(f'Single Point of Mode {mode:03d}{["(+)","(-)"][sign]}', flush=True)    
            
            # --- prepare input files ---
            mol.dislocate_to_xyz(mode,a0,os.path.join(ipath,XYZFILENAME)) # create xyz file
            vib_dict['#'] = set(comments)
            add_subkeys(vib_dict, '#', {f'VibMode {mode}',f'VibAmp {a0}'}) # add comment to input file stating vibrational mode number and dislocation
            write_input_file(vib_dict,os.path.join(ipath,INPUTFILE)) # write input file
            if os.path.isfile(gbw_file): # copy gbw file for MOguess if provided
                subprocess.run(['cp',gbw_file, os.path.join(ipath,GBWFILENAME)], stdout=subprocess.DEVNULL)

            # --- run calculation ---
            run_orca(INPUTFILE, OUTFILE, cwd=ipath, quiet=True)

            # --- clean up ---
            subprocess.run(['rm', '-f', GBWFILENAME, INPUTFILE, BIBFILE],cwd=ipath,stdout=subprocess.DEVNULL)
            
    print(f'...done!', flush=True)
    
    return 1


# =======================================================================================
#                                  MAIN FUNCTION
# =======================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Quickly start ORCA calculations.'
        )
    
    # --------------------------- Define Arguments ---------------------------
    # possible input files
    inout = parser.add_mutually_exclusive_group(required=True)
    inout.add_argument('-i', '--input', type=str, nargs='?', default=False, const='', help='If raised without argument, a new input file will be created. Otherwise the argument will be used as input. Without this flag mol.inp or any *.inp in the directory will be used.')
    inout.add_argument('-o',  '--out', type=str, default=False, help='Use orca output file as input for vibrational analysis.')

    # handle filenames
    parser.add_argument('-f', '--filename', type=str, default = 'mol.inp', help='filename of created input file. Default: mol.inp')
    parser.add_argument('-x', '--xyz', type=str, default='', help='Name of xyz file, used for input. If omitted input.xyz or any *.xyz will be used.')

    # handle keywords for ORCA
    parser.add_argument('-F', '--functional', type=str, default='B3LYP', help='Default: B3LYP')
    parser.add_argument('-B', '--basisset', type=str, default='6-31G*', help='Default: 6-31G*')
    parser.add_argument('-U', '--unrestricted', action='store_true', help='Add UKS keyword. If omitted ORCA will use RKS if possible.')
    parser.add_argument('-M', '--multiplicity', type=int, default = 0, help='If omitted multiplicity will be 1 for even number of electrons and 2 otherwise.')
    parser.add_argument('-C', '--charge', type=int, default=0, help='Positive charge corresponds to removal of electrons.')
    parser.add_argument('-O', '--opt', type=str, nargs='?', default = None, const='OPT', help='If raised without argument, OPT will be added as keyword. "tight", "verytight" and "very" will be translated to "TIGHTOPT" or "VERYTIGHTOPT".')
    parser.add_argument('-S', '--scf', type=str, default = '', help='SCF conversion level. "tight", "verytight" and "very" will be translated to "TIGHTSCF" or "VERYTIGHTSCF".')
    parser.add_argument('-V', '--freq', action='store_true', help='Add FREQ keyword.')
    parser.add_argument('-K', '--keywords', type=str, default = '', help='Optional string containing further keywords.')
    
    # add additional input commands for ORCA
    parser.add_argument('-n', '--nprocs', type=int, default = NPROCS, help=f'Number of CPUs to be used.')
    parser.add_argument('-m', '--maxcore', type=int, default = MAXCORE, help='RAM used per CPU in MB.')
    parser.add_argument('-b', '--basename', type=str, default = '', help='Base name of output files')
    parser.add_argument('-p', '--omit_MO_basis', action='store_true', help='Omit printing Basis and MOs to output.')
    parser.add_argument('-g', '--gbw_input', type=str, default = '', help='Name of *.gbw file to use for MOInput flag.')
    parser.add_argument('-t', '--tddft', action='store_true', help='Add keyword block for TD-DFT calculations')

    # flags for vibrational analysis
    parser.add_argument('-v', '--vibs', type=float, nargs='?', default = False, const = 0.5, help='Do full vibrational analysis. Parameter defines dislocation amplitude in units of HWHM of zero-phonon mode. Default: 0.5')
    parser.add_argument(      '--maxfreq', type=float, help='Only consider frequencies up to this value in eV.')
    parser.add_argument(      '--maxmode', type=int, help='Only consider vibrational modes up to (and including) "maxmode".')
    parser.add_argument(      '--minfreq', type=float, help='Only consider frequencies above this value in eV.')
    parser.add_argument(      '--minmode', type=int, default=0, help='Only consider vibrational modes higher than "minmode".')
    
    # flags for slurm
    parser.add_argument(      '--slurm', type=str, nargs='?', default=False, const='Quorca', help='Create slurm file instead of running Orca. Optional argument defines job name for slurm')
    parser.add_argument(      '--walltime', type=str, default='72:00:00', help='Walltime for slurm. Default: 72:00:00')
    
    # --------- Clean up some arguements ---------
    args = parser.parse_args()
        
    args.keywords = set([w.upper() for w in args.keywords.split()])

    # --------- Create/Find Input File ---------
    if args.input == '': # no input provided, create input file
        args.input = write_input_file(make_input_dict(args),args.filename)
        if args.slurm:
            args.basename = basename_from_input(args.input)
            write_slurm_file(args)
        quit()

    elif args.input: # input provided, run orca or create slurm file from input
        if not os.path.isfile(args.input):
            print('Cannot find input file. Abort!')
            quit()

        if args.slurm:
            args.basename = basename_from_input(args.input)
            write_slurm_file(args)      
            quit()
        args.out = run_orca(args.input)

    # --------- Is there an output to continue whith? ---------------------
    if not os.path.isfile(args.out):
        print('Cannot find output file. Abort!')
        quit()

    # --------- Do Vibrational Analysis ---------
    if args.vibs:
        if args.slurm: # create slurm file to run vibrational analysis
            vib_dict = parse_inp(input_from_out(args.out))
            slurm_args_from_input_dict(args,vib_dict)
            write_slurm_file(args,filename='vibs.sh')
            quit()

        if args.input: # this is second step, add delimiter to stdout
            print('\n----------------------------------------\n', flush=True)

        # find gbw file and run vibrational analysis.
        gbw = basename_from_output(args.out)+'.gbw'
        run_vibrational_analysis(args,gbw)

    # --------- All done, go get a beer ---------
    quit()  
