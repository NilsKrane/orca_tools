import os
import numpy as np
try:
    import ujson
except ModuleNotFoundError:
    import json as ujson
    
from .orca_utils import element_to_number
from .orca_utils import default_masses, bohr2ang, hartree, q_e

class HessMolecule:
    '''Class to parse and load `*.hess` files from ORCA.'''

    def __init__(self,filename: str | os.PathLike):
        '''Initializing instance of HessMolecule class. 

        :param filename: Path to `*.hess` as created by ORCA.
        :type filename: str | os.PathLike
        '''        
        content_dict = self.load_content(filename)
        '''Dictionary created from hessian file.'''
        
        self.filename = filename
        '''Filename of hessian file.'''

        self.atoms = self.parse_atoms(content_dict)
        '''List of dictionaries containing information for all atoms'''

        freqs = self.parse_freqs(content_dict)
        mask = freqs == 0

        self.freqs = self.remove_zero_freqs(freqs,mask)
        '''Energies of vibrational normal modes (eV)'''
        self.normal_modes = self.remove_zero_freqs(self.parse_normal_modes(content_dict),mask)
        '''Vibrational normal modes.'''
        self.hess = self.parse_hessian(content_dict)
        '''Hessian Matrix for vibrational analysis.'''
        self.ddipole = self.remove_zero_freqs(self.parse_ddipole(content_dict),mask)
        '''Electric dipole derivative with respect to normal modes'''
        self.ir = self.remove_zero_freqs(self.parse_ir(content_dict),mask)
        '''Table of IR intensities.'''
    
    @staticmethod
    def load_content(filename: str | os.PathLike) -> dict:
        '''Create dictionary from `*.hess` file.

        :param filename: Path to `*.hess` as created by ORCA.
        :type filename: str | os.PathLike
        :return: Dictionary with content of `*.hess` file.
        :rtype: dict
        '''        
        with open(filename) as file:
            content = file.read()
        
        content_dict = {}
        content = content.split("$")
        if len(content) > 1:
            for section in content[1:]:
                section = section.split("\n")
                content_dict[section[0]] = "\n".join(
                    [line for line in section[1:] if len(line) != 0])
            
        return content_dict

    @staticmethod
    def parse_atoms(content_dict: dict) -> list[dict]:
        '''Parse atoms section in `content_dict` to obtain list of dictionaries for atoms.

        :param content_dict: Content dictionary as created by `HessMolecule.load_content()`
        :type content_dict: dict
        :return: List containing a dictionary for each atom.
        :rtype: list[dict]
        '''

        if 'atoms' not in content_dict: return None

        raw_atoms = content_dict["atoms"].split("\n")
        atoms = []
        for line in raw_atoms[1:]:
            line = line.split()
            atom = {}
            atom["element"]=line[0]
            atom["mass"]=float(line[1])
            atom["coords"]=np.array([float(x) for x in line[2:]])*bohr2ang
            atom['nuclearcharge']=float(element_to_number(atom['element']))
            atoms.append(atom)
        return atoms

    @staticmethod
    def parse_freqs(content_dict: dict) -> np.ndarray:
        '''Get vibrational energies of normal modes.

        :param content_dict: Content dictionary as created by `HessMolecule.load_content()`
        :type content_dict: dict
        :return: Array of vibrational energies in eV.
        :rtype: np.ndarray
        '''
        
        if 'vibrational_frequencies' not in content_dict: return None
        
        raw_freqs = content_dict["vibrational_frequencies"].split("\n")
        num_modes = int(raw_freqs[0])
        freqs = np.zeros((num_modes))
        for mode, line in enumerate(raw_freqs[1:]):
            freqs[mode] = float(line.split()[1])/8065.54429
        return freqs     
    
    @staticmethod
    def parse_normal_modes(content_dict: dict) -> np.ndarray:
        '''Get vibrational normal modes.

        :param content_dict: Content dictionary as created by `HessMolecule.load_content()`
        :type content_dict: dict
        :return: Array of vibrational normal modes with shape (#modes,#atoms,3).
        :rtype: np.ndarray
        '''
        
        if 'normal_modes' not in content_dict: return None

        raw_modes = content_dict["normal_modes"].split("\n")
        num_modes = int(raw_modes[0].split()[1])
        normal_modes = np.zeros((num_modes,num_modes // 3, 3))

        for line in raw_modes[1:]:
            if line == "#":
                break
            line = line.split()
            if line[-1].find(".") < 0:
                col = int(line[0])
                continue

            row = int(line[0])
            for linecol in range(0,len(line[1:])):
                normal_modes[col+linecol, row // 3, row % 3] = float(line[linecol+1])

        return normal_modes

    @staticmethod
    def parse_hessian(content_dict: dict) -> np.ndarray:
        '''Get Hessian matrix.

        :param content_dict: Content dictionary as created by `HessMolecule.load_content()`
        :type content_dict: dict
        :return: Array containing Hessian matrix.
        :rtype: np.ndarray
        '''
        
        if 'hessian' not in content_dict: return None

        raw_coords = content_dict["hessian"].split("\n")
        num_coords = int(raw_coords[0])
        hessian = np.zeros((num_coords,num_coords))

        for line in raw_coords[1:]:
            if line == "#":
                break
            line = line.split()
            if line[-1].find(".") < 0:
                col = int(line[0])
                continue

            row = int(line[0])
            for linecol in range(0,len(line[1:])):
                hessian[row,col+linecol]=float(line[linecol+1])

        hessian = np.reshape(hessian, (num_coords, num_coords),order="F")
        return hessian

    @staticmethod
    def parse_ddipole(content_dict: dict) -> np.ndarray:
        '''Parse dipole derivative.

        :param content_dict: Content dictionary as created by `HessMolecule.load_content()`
        :type content_dict: dict
        :return: Array containing dipole derivatives with shape (#modes,3).
        :rtype: np.ndarray
        '''        
        raw_dmu = content_dict["dipole_derivatives"].split("\n")
        num_modes = int(raw_dmu[0])
        dipole_derivatives = np.zeros((num_modes,3))
        for mode, line in enumerate(raw_dmu[1:]):

            if "#" in line:
                continue
            dipole_derivatives[mode,:] = [float(x) for x in line.split()]
        return dipole_derivatives

    @staticmethod
    def parse_ir(content_dict: dict) -> np.ndarray:
        '''Parse IR table.

        :param content_dict: Content dictionary as created by `HessMolecule.load_content()`
        :type content_dict: dict
        :return: Array of IR table (#modes,7), with columns being: wavenumber, eps, Int, T**2, Tx, Ty, Tz.
        :rtype: np.ndarray
        '''                
        raw_ir = content_dict["ir_spectrum"].split("\n")
        num_modes = int(raw_ir[0])
        ir_spectrum = np.zeros((num_modes,6))
        for mode, line in enumerate(raw_ir[1:]):
            ir_spectrum[mode,:] = [float(x) for x in line.split()]
        ir_spectrum = np.insert(ir_spectrum, 3, 0., axis=1)
        ir_spectrum[:,3] = np.sum(ir_spectrum[:,4:]**2,axis=1)
        return ir_spectrum
    
    @staticmethod
    def remove_zero_freqs(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
        '''Use mask to remove zero frequency (i.e. rotational and translational) modes.

        :param a: Array with first dimension being the modes.
        :type a: np.ndarray
        :param mask: Mask with boolean values (`True` corresponds to zero frequency mode)
        :type mask: np.ndarray
        :return: Shortened array without zero frequency modes.
        :rtype: np.ndarray
        '''        
        return np.delete(a, mask, axis=0)

# -------------------------------------------------

class OutMolecule:
    '''Class to parse and load output files from ORCA.'''
    def __init__(self,filename: str):
        '''Initializing instance of OutMolecule class. 

        :param filename: Path to output file as created by ORCA.
        :type filename: str | os.PathLike
        '''            

        self.filename = filename
        '''Filename of output file.'''

        content = self.load_content(filename)
        
        self.atoms = self.parse_atoms(content)
        '''List of dictionaries containing information for all atoms'''
        self.input_str = self.parse_input(content)
        '''Content of input file for ORCA calculation.'''
        self.energy = self.parse_energy(content)
        '''Total Energy (eV)'''
        self.charge, self.multiplicity = self.parse_charge_multiplicity(self.input_str)
        self.UKS = self.parse_UKS(content)
        '''Boolean: Is unrestricted?'''
        self.success = self.parse_success(content)
        '''Boolean: Did ORCA terminate normally?'''

        self.occupation, self.MO_energies = self.parse_occupation_energies(content, self.UKS)

        
        self.TDstates = self.parse_TDstates(content)
        '''List of dict containing excited states.'''
        self.NTOs = self.parse_NTOs(content)
        '''List of NTO states.'''
        self.tdip, self.dtdip = self.parse_dtdip(content)


        self.freqs = None
        '''Energies of vibrational normal modes (eV)'''
        self.normal_modes = None
        '''Vibrational normal modes.'''
        self.hess = None
        '''Hessian Matrix for vibrational analysis.'''
        self.ddipole = None
        '''Electric dipole derivative with respect to normal modes'''
        self.ir = None
        '''Table of IR intensities with shape (#modes,7) and columns being: wavenumber, eps, Int, T**2, Tx, Ty, Tz'''

        freqs = self.parse_freqs(content)
        if type(freqs) is not type(None):
            mask = freqs == 0
            self.freqs = self.remove_zero_freqs(freqs,mask)
            self.normal_modes = self.remove_zero_freqs(self.parse_normal_modes(content),mask)
       
    @staticmethod
    def load_content(filename: str) -> str:
        '''Load content from file into string.

        :param filename: Path to output file.
        :type filename: str
        :return: Content of file.
        :rtype: str
        '''        
        with open(filename) as file:
            content = file.read()
        return content
        
    @staticmethod
    def parse_atoms(content: str) -> list[dict]:
        '''Parse atoms section in content to obtain list of dictionaries for atoms.

        :param content: Content of output file.
        :type content: str
        :return: List containing a dictionary for each atom.
        :rtype: list[dict]
        '''
        marker_start = "CARTESIAN COORDINATES (A.U.)\n"
        marker_end = "INTERNAL COORDINATES (ANGSTROEM)"
        if marker_start not in content or marker_end not in content:
            return None
        coords_raw = content.split(marker_start)[-1]
        coords_raw = coords_raw.split(marker_end)[0]
        atoms=[]
        for line in coords_raw.split("\n")[2:-3]:
            line = line.split()
            atom = {}
            atom["element"]=line[1]
            atom['nuclearcharge']=float(line[2])
            atom["mass"]=float(line[4])
            atom["coords"]=np.array([float(x) for x in line[5:]])*bohr2ang
            atoms.append(atom)
        return atoms

    @staticmethod
    def parse_success(content: str) -> bool:
        '''Check if ORCA calculation terminated normally.

        :param content: Content of output file.
        :type content: str
        :return: True if `****ORCA TERMINATED NORMALLY****` found in content.
        :rtype: bool
        '''        
        if content.find('****ORCA TERMINATED NORMALLY****'):
            return True
        else:
            return False

    @staticmethod
    def parse_input(content: str) -> str:
        '''Find input section in output file and reconstruct input string.

        :param content: Content of output file.
        :type content: str
        :return: Input string of calculation.
        :rtype: str
        '''        
        marker_start = "INPUT FILE\n"+"="*80+"\n"
        marker_end = "****END OF INPUT****"
        if marker_start not in content or marker_end not in content:
            return None
        input_lines = content.split(marker_start)[1].split(marker_end)[0].split("\n")[1:-1]
        input_lines = [line[6:] for line in input_lines]
        return "\n".join(input_lines)

    @staticmethod
    def parse_charge_multiplicity(input_str: str) -> tuple[int, int]:
        '''Parse input string for charge state and multiplicity of molecule.

        :param input_str: Input string of calculation.
        :type input_str: str
        :return: Tuple of integers for charge and multiplicty.
        :rtype: tuple[int, int]
        '''        
        if type(input_str) == type(None):
            return None, None
        for line in input_str.split("\n"):
            if len(line) > 3:
                if line[0] == "*":
                    line = line[1:].split()
                    return int(line[1]), int(line[2])
        return None, None
    
    @staticmethod
    def parse_energy(content: str) -> float:
        '''Get total SCF energy.

        :param content: Content of output file.
        :type content: str
        :return: Total SCF Energy in eV.
        :rtype: float
        '''        
        try:
            return float(content.split("TOTAL SCF ENERGY\n----------------\n\nTotal Energy")[-1].split("\n")[0].split()[-2])
        except:
            return None
            
    @staticmethod
    def parse_UKS(content: str) -> bool:
        '''Was ORCA calculation unrestricted?

        :param content: Content of output file.
        :type content: str
        :return: True if unrestricted.
        :rtype: bool
        '''        
        try:
            return content.split(" Hartree-Fock type      HFTyp           .... ")[-1].split()[0] == 'UHF'
        except:
            return None
    
    @classmethod
    def parse_occupation_energies(self, content: str, UKS: bool) -> tuple[np.ndarray,np.ndarray]:
        '''Get occupation and energies of molecular orbitals.

        :param content: Content of output file.
        :type content: str
        :param UKS: True if calculation was unrestricted.
        :type UKS: bool
        :return: Tuple of arrays containing occupation and energies of MOs.
        :rtype: tuple[np.ndarray,np.ndarray]
        '''        
        if '------------------\nMOLECULAR ORBITALS' in content:
            occ, en = self.__parse_occupation_energies_MO(content, UKS)
        elif "ORBITAL ENERGIES\n----------------\n" in content:
            occ, en = self.__parse_occupation_energies_OE(content, UKS)
        else:
            occ, en = None, None
        return occ, en

    @staticmethod
    def __parse_occupation_energies_OE(content: str, UKS: bool) -> tuple[np.ndarray,np.ndarray]:
        '''Get occupation and energies of molecular orbitals, in case the full MOs were not printed.

        :param content: Content of output file.
        :type content: str
        :param UKS: True if calculation was unrestricted.
        :type UKS: bool
        :return: Tuple of arrays containing occupation and energies of MOs (LUMOs might be truncated).
        :rtype: tuple[np.ndarray,np.ndarray]
        '''        
        def occupations_energies(content,spin=0):
            content = content.split("ORBITAL ENERGIES\n----------------\n")[-1]
            
            marker1 = '\n*Only the first 10 virtual orbitals were printed.'
            marker2 = '------------------'
            lines = content.split(marker2)[0].split(marker1)[0].split('\n\n')[spin].split('\n')[2:]

            occupations = np.zeros(len(lines))
            energies = np.zeros(len(lines))
            for line in lines:
                if len(line):
                    occupations[int(line.split()[0])] = float(line.split()[1]) 
                    energies[int(line.split()[0])] = float(line.split()[3])

            return occupations, energies

        if UKS:
            occ_alpha, en_alpha = occupations_energies(content)
            occ_beta, en_beta = occupations_energies(content,spin=1)
            occ = np.stack([occ_alpha[:occ_beta.shape[0]],occ_beta], axis=1)
            en = np.stack([en_alpha[:en_beta.shape[0]],en_beta], axis=1)
        else:
            occ, en = occupations_energies(content)

        return occ, en

    @staticmethod
    def __parse_occupation_energies_MO(content: str, UKS: bool) -> tuple[np.ndarray,np.ndarray]:
        '''Get occupation and energies of molecular orbitals, in case the full MOs were printed.

        :param content: Content of output file.
        :type content: str
        :param UKS: True if calculation was unrestricted.
        :type UKS: bool
        :return: Tuple of arrays containing occupation and energies of MOs.
        :rtype: tuple[np.ndarray,np.ndarray]
        '''        

        def occupations_energies(MO_block,num_orbs):
            occupation = np.zeros((num_orbs))
            energies = np.zeros((num_orbs))

            block_sep = '                  --------  --------  --------  --------  --------  --------\n'
            for block in MO_block.split(block_sep)[:-1]:
                block = block.split('\n')[:-1]
                for i, iorb in enumerate(block[-3].split()):
                    iorb = int(float(iorb))
                    occupation[iorb] = float(block[-1].split()[i])
                    energies[iorb] = float(block[-2].split()[i])
            return occupation, energies

        content = content.split('------------------\nMOLECULAR ORBITALS')
        num_orbs = int(content[-2].split(' Basis Dimension        Dim             ....')[-1].split('\n')[0])
        MO_blocks = content[-1].split('\n\n\n')[0].split('\n\n')
        
        if UKS:
            occ_alpha, en_alpha = occupations_energies(MO_blocks[0],num_orbs)
            occ_beta, en_beta = occupations_energies(MO_blocks[1],num_orbs)
            occ = np.stack([occ_alpha,occ_beta], axis=0)
            en = np.stack([en_alpha,en_beta], axis=0)
        else:
            occ, en = occupations_energies(MO_blocks[0],num_orbs)

        return occ, en

    @staticmethod
    def parse_freqs(content: str) -> np.ndarray:
        '''Get vibrational energies of normal modes.

        :param content: Content of output file.
        :type content: str
        :return: Array of vibrational energies in eV.
        :rtype: np.ndarray
        '''        
        if "VIBRATIONAL FREQUENCIES" not in content:
            return None

        raw_freqs = content.split("VIBRATIONAL FREQUENCIES")[1]
        raw_freqs = raw_freqs.split("NORMAL MODES")[0]

        freqs = []
        for mode, line in enumerate(raw_freqs.split("\n")[1:]):
            if "cm**-1" in line:
                freqs.append(float(line.split()[1])/8065.54429)
        return np.array(freqs)
    
    @staticmethod
    def parse_normal_modes(content: str) -> np.ndarray:
        '''Get vibrational normal modes.

        :param content: Content of output file.
        :type content: str
        :return: Array of vibrational normal modes with shape (#modes,#atoms,3).
        :rtype: np.ndarray
        '''

        if "NORMAL MODES" not in content:
            return None

        raw_modes = content.split("NORMAL MODES")[1]
        raw_modes = raw_modes.split("IR SPECTRUM")[0]

        num_modes = int(raw_modes.split(" "*18)[1].split("\n")[-2].split()[0])+1
        normal_modes = np.zeros((num_modes,num_modes // 3, 3))

        for line in raw_modes.split("\n")[7:-3]:
            line = line.split()
            if not len(line):
                continue
            if line[-1].find(".") < 0:
                col = int(line[0])
                continue

            row = int(line[0])
            for linecol in range(0,len(line[1:])):
                normal_modes[col+linecol, row // 3, row % 3] = float(line[linecol+1])

        return normal_modes
    
    @staticmethod
    def remove_zero_freqs(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
        '''Use mask to remove zero frequency (i.e. rotational and translational) modes.

        :param a: Array with first dimension being the modes.
        :type a: np.ndarray
        :param mask: Mask with boolean values (`True` corresponds to zero frequency mode)
        :type mask: np.ndarray
        :return: Shortened array without zero frequency modes.
        :rtype: np.ndarray
        '''
        return np.delete(a, mask, axis=0)

    @staticmethod
    def parse_ir(content: str) -> np.ndarray:
        '''Parse IR table.

        :param content: Content of output file.
        :type content: str
        :return: Array of IR table (#modes,7), with columns being: wavenumber, eps, Int, T**2, Tx, Ty, Tz.
        :rtype: np.ndarray
        '''                

        if "IR SPECTRUM" not in content:
            return None

        raw_modes = content.split("IR SPECTRUM\n-----------\n\n")[1]
        raw_modes = raw_modes.split("\n\n")[0]

        raw_modes = raw_modes.split("\n")[3:]
        num_modes = len(raw_modes)
        ir_spectrum = np.zeros((num_modes,7))

        for m, line in enumerate(raw_modes):
            line = line.replace("(","").replace(")","").split()
            ir_spectrum[m,:] = np.array([float(x) for x in line[1:]])

        return ir_spectrum
        

    @staticmethod
    def parse_TDstates(content: str) -> list[dict]:
        '''Parse excited states from TD-DFT calculations.

        :param content: Content of output file.
        :type content: str
        :return: List containing dictionary for each excited state.
        :rtype: list[dict]
        '''                

        if not content: return None

        try:            
            nroots = int(content.split('Number of roots to be determined               ...')[-1].split()[0])
        except ValueError:
            return None
        
        content = content.split("TD-DFT/TDA EXCITED STATES")
        if len(content) == 1:
            TDA = False
            content = content[0].split("TD-DFT EXCITED STATES")
            if len(content) == 1:
                return None
        else:
            TDA = True
        
        content = [c.split('\n\n')[0] for c in content[-1].split("STATE")[1:nroots+1]]

        TDstate_list = [None]*len(content)
        for i, block in enumerate(content):
            block = block.removesuffix("\n\n")
            state_dict = {}
            state_dict["State"] = int(block.split(":")[0])
            state_dict['Energy'] = block.split("E=")[1].split()[2]
            state_dict["Multiplicity"] = block.split("E=")[1].split()[10]

            MO_lines = block.split("E=")[1].split("\n")[1:]
            orbitals = [None] * len(MO_lines)
            for j, line in enumerate(MO_lines):
                line = line.split()
                mo_dict = {}
                mo_dict['initialMO'] = int(line[0][:-1])
                mo_dict['initialSpin'] = int("ab".find(line[0][-1]))
                mo_dict['finalMO'] = int(line[2][:-1])
                mo_dict['finalSpin'] = int("ab".find(line[2][-1]))
                mo_dict['weight'] = float(line[4])
                if TDA:
                    mo_dict['c'] = float(line[6][:-1])
                orbitals[j] = mo_dict
            state_dict["Orbitals"] = orbitals
            TDstate_list[i] = state_dict
        return TDstate_list

    @staticmethod
    def parse_NTOs(content: str) -> list[dict]:
        '''Parse NTO tables.

        :param content: Content of output file.
        :type content: str
        :return: List containing dictionary for each excited state.
        :rtype: list[dict]
        '''        

        content = content.split("NATURAL TRANSITION ORBITALS FOR STATE    1")
        if len(content) > 1:
            final_ntos = "    1"+content[-1]
            nto_strings = [nto_table.split("\n\n\n")[0] 
                           for nto_table in final_ntos.split("NATURAL TRANSITION ORBITALS FOR STATE")]

            nto_list = [None] * len(nto_strings)
            for i,nto in enumerate(nto_strings):
                nto_dict = {}
                nto_dict['State'] = int(nto.split()[0])
                nto_dict['Energy'] = nto_strings[0].split("E=")[1].split()[2]

                MO_lines = nto_strings[0].split("E=")[1].split("\n")[1:]
                orbitals = [None] * len(MO_lines)*2
                for j, line in enumerate(MO_lines):
                        line = line.split()
                        mo_dict = {}
                        mo_dict['MO'] = int(line[0][:-1])
                        mo_dict['Spin'] = int("ab".find(line[0][-1]))
                        mo_dict['Occupation'] = float(line[-1])
                        orbitals[2*j] = mo_dict

                        mo_dict = {}
                        mo_dict['MO'] = int(line[2][:-1])
                        mo_dict['Spin'] = int("ab".find(line[2][-1]))
                        mo_dict['Occupation'] = float(line[-1])
                        orbitals[2*j+1] = mo_dict

                nto_dict['NTOs'] = orbitals
                nto_list[i]=nto_dict
        else:
            nto_list = None
        
        return nto_list
    
    @staticmethod
    def parse_absorption_dipole(content: str) -> list[dict]:
        '''Parse absortion dipole table.

        :param content: Content of output file.
        :type content: str
        :return: List containing dictionary for each excited state.
        :rtype: list[dict]
        '''              
        content = content.split("ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS")
        if len(content) > 1:
            content = content[-1].split("\n\n")[0]
            content = content.split("-------\n")[-1]
            content = content.split("\n")

            transitions_list = [None]*len(content)
            for i, line in enumerate(content):
                state_dict = {}
                state_dict["Tag"] = line[:20].strip()
                line = line[20:].split()
                state_dict["Energy"] = float(line[0])
                state_dict["Wavelength"] = float(line[2])
                state_dict["fosc"] = float(line[3])
                state_dict["D2"] = float(line[4])
                state_dict["DX"] = float(line[5])
                state_dict["DY"] = float(line[6])
                state_dict["DZ"] = float(line[7])

                transitions_list[i] = state_dict
            return transitions_list
        else:
            return None

    @staticmethod
    def parse_dtdip(content: str) -> tuple[np.ndarray,np.ndarray]:
        '''Calculate Transition Dipole Derivatives from ESD calculation.

        :param content: Content of output file.
        :type content: str
        :return: Two arrays: Transition dipole reference of normal mode and transition dipole derivatives with shape (#modes,3)
        :rtype: tuple[np.ndarray,np.ndarray]
        '''          

        if content.find("Computing transition dipole derivatives") < 0:
            return None, None
        else:
            ref = content.split("Reference TDIP (x,y,z, a.u.):")[1].split("\n\n")[0].replace(")","").replace("(","").split(",")
            ref_tdip = np.array([float(a.strip().split()[0]) for a in ref])

            num_modes = int(content.split("will need at least")[-1].split("displacements")[0])

            lines = content.split("\nFinal check on derivatives:")[0].split("\t<<Displacing mode ")[1:]
            tdip = np.zeros((num_modes,3))
            tdip_r = np.zeros_like(tdip)
            tdip_r[:,:] = ref_tdip[None,:]
            scale = np.ones(num_modes)
            for line in lines:
                mode = int(line.split()[0])
                sign = line.split("(")[1].split(")")[0]
                itdip = [float(a.strip().split()[0]) for a in line.split("tdip = ")[1].replace(")","").replace("(","").split(",\n")]
                if sign == "+":
                    tdip[mode-1,:] = itdip
                elif sign == "-":
                    tdip_r[mode-1,:] = itdip
                    scale[mode-1] = 2.
            return ref_tdip, (tdip - tdip_r)/scale[:,None]
   

# -------------------------------------------------

class JSONMolecule:
    '''Class to parse and load `*.JSON` files created from .gbw or .nto files via orca_2json binary.'''

    def __init__(self,filename: str | os.PathLike=None, content: dict=None):
        '''Initializing instance of JSONMolecule class, either from JSON file or directly from dictionary.

        :param filename: Path to `*.JSON` file
        :type filename: str | os.PathLike, optional
        :param content: Dictionary with content from orca created JSON file.
        :type content: dict, optional
        '''        
        if type(filename) is not type(None):
            self.filename = filename
            '''Filename of JSON file.'''
            self.json = self.load_content(filename)
            '''JSON Dictionary'''
        elif isinstance(content,dict):
            self.json = content
        
        self.atoms = self.parse_atoms(self.json)
        '''List of dictionaries containing information for all atoms'''
        self.occupation = self.parse_occupation(self.json)
        '''Array containing occupation of MOs. Array is 1d if `UKS=False` and 2d otherwise.'''
        self.UKS = len(self.occupation.shape) == 2
        '''Boolean: Is unrestricted?'''
        self.MO_energies = self.parse_MO_energies(self.json,self.UKS)
        '''Array containing energies of MOs. Array is 1d if `UKS=False` and 2d otherwise.'''
        
        self.multiplicity = self.json['Molecule']['Multiplicity']
        '''Multiplicity'''
        self.charge = self.json['Molecule']['Charge']
        '''Charge. Positive values indicates removal of electrons.'''

    @staticmethod
    def load_content(filename: str) -> dict:
        '''Load content from JSON file into dictionary

        :param filename: Path to JSON file.
        :type filename: str
        :return: Content as dictionary.
        :rtype: dict
        '''        
        with open(filename, 'r') as file:
            content = ujson.load(file)
        return content
    
    @staticmethod
    def parse_atoms(content: dict) -> list[dict]:
        '''Parse atoms section in content to obtain list of dictionaries for atoms.

        :param content: Content dictionary loaded from JSON file.
        :type content: dict
        :return: List containing a dictionary for each atom.
        :rtype: list[dict]
        '''        
        atoms = []
        for json_atom in content['Molecule']['Atoms']:
            atom = {}
            atom["element"]=json_atom['ElementLabel']
            atom["nuclearcharge"]=float(json_atom['NuclearCharge'])
            atom["coords"]=np.array(json_atom['Coords'])
            atom['mass']=default_masses[int(atom['nuclearcharge'])]
            atoms.append(atom)
        return atoms

    @staticmethod
    def parse_occupation(content: dict) -> np.ndarray:
        '''Parse MO occupation numbers from JSON dictionary.

        :param content: Content dictionary loaded from JSON file.
        :type content: dict
        :return: 1d (restricted) or 2d (unrestricted) array with occupation numbers.
        :rtype: np.ndarray
        '''        
        n_orbs = len(content["Molecule"]['MolecularOrbitals']['MOs'])
        occ = np.zeros((n_orbs,))
        for i in range(n_orbs):
            occ[i] = content["Molecule"]["MolecularOrbitals"]['MOs'][i]['Occupancy']
        if occ[0] == 1:
            occ = occ.reshape(2,(n_orbs//2),order='C')
        return occ
    
    @staticmethod
    def parse_MO_energies(content: dict, UKS: bool) -> np.ndarray:
        '''Parse MO energies from JSON dictionary.

        :param content: Content dictionary loaded from JSON file.
        :type content: dict
        :return: 1d (restricted) or 2d (unrestricted) array with MO energies in eV.
        :rtype: np.ndarray
        '''                
        n_orbs = len(content["Molecule"]['MolecularOrbitals']['MOs'])
        en = np.zeros((n_orbs,))
        for i in range(n_orbs):
            en[i] = content["Molecule"]["MolecularOrbitals"]['MOs'][i]['OrbitalEnergy']
        if UKS:
            en = en.reshape(2,(n_orbs//2),order='C')
        return en*hartree/q_e # convert from Hartree to eV

