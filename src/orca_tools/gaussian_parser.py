import os
import numpy as np

from .orca_utils import number_to_element, default_masses
from .orca_utils import hartree, q_e

class GaussianOutMolecule:
    '''Class to parse and load output files from Gaussian.'''
    def __init__(self,filename: str):
        '''Initializing instance of OutMolecule class. 

        :param filename: Path to output file as created by Gaussian.
        :type filename: str | os.PathLike
        '''            

        self.filename = filename
        '''Filename of output file.'''

        content = self.load_content(filename)
        
        self.atoms = self.parse_atoms(content)
        '''List of dictionaries containing information for all atoms'''

        self.energy = self.parse_energy(content)
        '''Total Energy (eV)'''
        self.charge, self.multiplicity = self.parse_charge_multiplicity(content)

        self.success = self.parse_success(content)
        '''Boolean: Did Gaussian terminate normally?'''

        self.occupation, self.MO_energies = self.parse_occupation_energies(content)

        self.UKS = self.isUKS()
        '''Boolean: Is unrestricted?'''

        self.TDstates = self.parse_TDstates(content)
        '''List of dict containing excited states.'''

        self.freqs = None
        '''Energies of vibrational normal modes (eV)'''
        self.normal_modes = None
        '''Vibrational normal modes.'''

        self.freqs, self.normal_modes = self.parse_normal_modes(content)

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
    def parse_success(content: str) -> bool:
        '''Parse whether Gaussian terminated normally.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: True if Gaussian terminated normally, False otherwise.
        :rtype: bool
        '''        
        return ' Normal termination of Gaussian' in content.split('\n')[-2]

    @staticmethod
    def parse_energy(content: str) -> float:
        '''Parse total energy from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: Total energy in eV.
        :rtype: float
        '''        
        energy_au = float(content.split('SCF Done:')[1].split('\n')[0].split()[2])
        return energy_au*hartree/q_e

    @staticmethod
    def parse_atoms(content: str) -> list[dict]:
        '''Parse atoms section in content to obtain list of dictionaries for atoms.

        :param content: Content of output file.
        :type content: str
        :return: List containing a dictionary for each atom.
        :rtype: list[dict]
        '''

        coords, element = GaussianOutMolecule.parse_coords(content)
        try:
            mass = GaussianOutMolecule.parse_mass(content)
        except ValueError:
            mass = np.array([default_masses[e] for e in element])

        atoms = []
        for c, e, m in zip(coords,element,mass):
            atom = {}
            atom['nuclearcharge']=e
            atom["element"]=number_to_element(e)
            atom["mass"]=m
            atom["coords"]=c
            atoms.append(atom)
        
        return atoms

    @staticmethod
    def parse_coords(content: str) -> tuple[np.ndarray,np.ndarray]:
        """Parse coordinates and element numbers from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: Tuple containing coordinates (in Angstrom) and atomic numbers.
        :rtype: tuple[np.ndarray,np.ndarray]
        """

        block_start = ''' ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------\n'''

        block_end = '''\n ---------------------------------------------------------------------'''

        block = content.split(block_start)[1].split(block_end)[0].split('\n')

        coords = np.zeros((len(block),3))
        element = np.zeros(len(block),dtype=int)
        for i, atom in enumerate(block):
            coords[i,:] = [float(x) for x in atom.split()[-3:]]
            element[i] = int(atom.split()[1])

        return coords, element


    @staticmethod
    def parse_mass(content: str) -> np.ndarray:
        """Parse atomic masses from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: Array of atomic masses (in amu).
        :rtype: np.ndarray
        """

        block_start = ''' -------------------
 - Thermochemistry -
 -------------------\n'''
        block_end = ' Molecular mass:'

        try:
            block = content.split(block_start)[1].split(block_end)[0].split('\n')[1:-1]
        except IndexError:
            raise ValueError('Could not parse masses from Gaussian output file.')

        mass = np.zeros((len(block)))
        for i, atom in enumerate(block):
            mass[i] = float(atom.split()[-1])

        return mass
    
    @staticmethod
    def parse_charge_multiplicity(content: str) -> tuple[int,int]:
        """Parse charge and multiplicity from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: Tuple containing charge and multiplicity.
        :rtype: tuple[int,int]
        """        
    
        input = content.split(' ******************************************\n')[2].split('\n\n')[0]
        input = input.split('\n Charge = ')[1].split('\n')[0].split()
        
        return int(input[0]), int(input[-1])

    @staticmethod
    def parse_occupation_energies(content: str) -> tuple[np.ndarray, np.ndarray]:
        """Parse orbital occupation numbers and energies from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: Tuple containing arrays of occupation numbers and orbital energies (in eV).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        alpha_occ, alpha_energies = GaussianOutMolecule.__spin_occ_energy(content, 'Alpha')
        try:
            beta_occ, beta_energies = GaussianOutMolecule.__spin_occ_energy(content, 'Beta')
        except ValueError:
            occupation = 2*alpha_occ
            energies = alpha_energies
        else:
            occupation = np.vstack([alpha_occ, beta_occ])
            energies = np.vstack([alpha_energies, beta_energies])

        return occupation, energies

    @staticmethod
    def __spin_occ_energy(content: str, spin: str) -> tuple[np.ndarray, np.ndarray]:
        """Parse orbital occupation numbers and energies for a given spin from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :param spin: Spin type ('Alpha' or 'Beta').
        :type spin: str
        :return: Tuple containing arrays of occupation numbers and orbital energies (in eV).
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        block_start = 'Population analysis using the SCF Density.'
        block = content.split(block_start)[-1]

        if f'{spin}  occ. eigenvalues --' in block:
            occ_lines = [line.split('\n')[0] for line in  block.split(f'{spin}  occ. eigenvalues --')[1:]]
            virt_lines = [line.split('\n')[0] for line in  block.split(f'{spin} virt. eigenvalues --')[1:]]
            occ_energies = np.array([float(x) for line in occ_lines for x in line.split()])
            virt_energies = np.array([float(x) for line in virt_lines for x in line.split()])
            occupation = np.concatenate([np.ones_like(occ_energies), np.zeros_like(virt_energies)])
            energies = np.concatenate([occ_energies, virt_energies])

            return occupation, energies * 27.2114  # Convert Hartree to eV
        else:
            raise ValueError(f"{spin} orbitals not found.")

    def isUKS(self) -> bool:
        """Check if calculation is unrestricted.

        :return: True if unrestricted, False if restricted.
        :rtype: bool
        """        

        if self.occupation.ndim == 2:
            return True
        elif self.occupation.ndim == 1:
            return False
        return None

    @staticmethod
    def parse_normal_modes(content: str) -> tuple[np.ndarray,np.ndarray]:
        """Parse normal modes from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: Tuple containing frequencies (in eV) and normal modes.
        :rtype: tuple[np.ndarray,np.ndarray]
        """        

        block_start = ''' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering
 activities (A**4/AMU), depolarization ratios for plane and unpolarized
 incident light, reduced masses (AMU), force constants (mDyne/A),
 and normal coordinates:\n'''
        
        block_end = '\n\n'
        
        try:
            block = content.split(block_start)[1].split(block_end)[0].split('\n')
        except IndexError:
            return None, None
        
        block += ['               ']

        normal_modes = []
        freqs = []

        flag = False
        for line in block:
            if line.startswith('               ') and flag:
                flag = False
                [normal_modes.append(mode) for mode in line_modes];
            elif line.startswith(' Frequencies --'):
                freqs += [float(f) for f in line.split()[2:]]
                line_modes =  [[] for _ in line.split()[2:]]
            
            elif line.startswith('  Atom'):
                flag = True
            elif flag:
                line_coords = line.split()[2:]
                for i in range(len(line_modes)):
                    line_modes[i].append([float(x) for x in line_coords[i*3:(i+1)*3]])
            
        return np.array(freqs)/8065, np.array(normal_modes)

    @staticmethod
    def parse_TDstates(content: str) -> list[dict]:
        """Parse excited state from Gaussian output content.

        :param content: Content of Gaussian output file.
        :type content: str
        :return: List containing a dictionary for each excited state.
        :rtype: list[dict]
        """        

        # ---------------
        # Parse transition dipole moments
        # ---------------
        block_start = 'Ground to excited state transition electric dipole moments (Au):'
        block_end = 'Ground to excited state transition velocity dipole moments (Au):'

        try:
            block = content.split(block_start)[1].split(block_end)[0].split('\n')[2:-1]
        except IndexError:
            return None

        nstates = len(block)

        states = [None]*nstates
        for i in range(nstates):
            line = block[i].split()
            state = {
                'State': int(line[0]),
                'tdip': np.array([float(a) for a in line[1:4]]),
                'tdip2': float(line[4]),
                'fosc': float(line[5]),
            }
            states[i] = state

        # ---------------
        # Parse excitation energies and orbital contributions
        # ---------------
        block_start = ' Excitation energies and oscillator strengths:\n \n'
        block_end = ' SavETr:'
        blocks = content.split(block_start)[1].split(block_end)[0].split(' Excited State')[1:]

        for block in blocks:
            
            lines = block.split('\n')
            line = lines[0].split()
            
            state_idx = int(line[0].strip(':'))-1

            states[state_idx]['Energy'] = float(line[2])
            states[state_idx]['S2'] = float(line[7].strip('<S**2>='))

            orbitals = []
            for line in lines[1:]:
                if '->' in line:
                    mo_dict = {}
                    mo_dict['initialMO'] = int(line.split('->')[0].strip())
                    mo_dict['finalMO'] = int(line.split('->')[1].split()[0])
                    mo_dict['c'] = float(line.split()[2])
                    mo_dict['weight'] = mo_dict['c']**2
                    orbitals.append(mo_dict)
                else:
                    break

            states[state_idx]['Orbitals'] = orbitals

        return states