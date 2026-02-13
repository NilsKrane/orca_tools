import os
import numpy as np

from .orca_utils import element_to_number, default_masses

class xyzMolecule:
    '''Class to parse and load output files from Gaussian.'''
    def __init__(self,filename: str):
        '''Initializing instance of xyzMolecule class. 

        :param filename: Path to xyz file.
        :type filename: str | os.PathLike
        '''            

        self.filename = filename
        '''Filename of xyz file.'''

        content = self.load_content(filename)
        
        self.atoms = self.parse_atoms(content)
        '''List of dictionaries containing information for all atoms'''

        self.header = self.parse_header(content)
        '''Header line from xyz file'''

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

        lines = content.split('\n')
        num_atoms = int(lines[0].strip())

        atoms = []
        for line in lines[2:2 + num_atoms]:
            parts = line.split()
            atom = {
                'element': parts[0],
                'coords': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                'mass': default_masses[element_to_number(parts[0])],
                'nuclearcharge': parts[0],
            }
            atoms.append(atom)
                
        return atoms

    @staticmethod
    def parse_header(content: str) -> str:
        '''Grab second line from xyz file content containing comments.
        
        :param content: Content of output file.
        :type content: str
        :return: String containing comment in xyz file.
        :rtype: str 
        '''
        if isinstance(content,str):
            lines = content.split('\n')
            if len(lines) > 1:
                return lines[1]
        else:
            return None