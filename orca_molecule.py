import numpy as np
import os
import subprocess
try:
    import ujson as json
except ModuleNotFoundError:
    print("Module ujson not found, defaulting to slower json module")
    import json

#sys.path.append(os.path.join(os.environ['USERPROFILE'],'OneDrive - empa.ch/_calcs'))
class Molecule:
    def __init__(self,filenames, orca_path="", warning=True):

        try:
            filename = filenames.pop(0)
        except AttributeError:
            filename = filenames
            filenames = []
        
        self.filename = filename
        self.basename = None

        self.atoms = None
        
        self.input_str = None
        self.energy = None
        self.multiplicity = None
        self.charge = None
        self.UKS = None
        self.success = None
        
        self.occupation = None
        self.MO_energies = None

        self.TDstates = None
        self.NTOs = None
        self.tdip = None
        self.dtdip = None

        self.freqs = None
        self.normal_modes = None
        self.hess = None
        self.ddipole = None
        self.ir = None
        
        self.gbw = None
        self.json = None
        self.orca_path = orca_path
        
        self.__read_file(warning)
        self.add_source(filenames,warning)

    def add_source(self,filenames, warning=True, overwrite=False):
        if isinstance(filenames,str):
            self.__read_file(warning,filenames,overwrite)
        else:
            for filename in filenames:
                self.__read_file(warning,filename,overwrite)
    
    def __read_file(self, warning=True, filename="", overwrite=False):
        if len(filename) == 0:
            filename=self.filename
            setbasename = True
        else:
            setbasename = False
                    
        assert os.path.exists(filename), "File not found!"
        neg_freqs_found = 0

        if filename[-5:] == ".hess":
            neg_freqs_found = self.__read_file_hess(filename,overwrite)
            if setbasename: self.basename = self.filename.removesuffix(".hess")

        elif filename[-4:] == ".gbw":
            self.gbw = filename
            #self.__read_file_gbw(filename,overwrite)
            if setbasename: self.basename = self.filename.removesuffix(".gbw")
        
        elif filename[-4:] == ".nto":
            self.__read_file_gbw(filename,overwrite)
            if setbasename: self.basename = self.filename.removesuffix(".nto")
        
        else:
            neg_freqs_found = self.__read_file_out(filename,overwrite)
            if setbasename: self.basename = self.filename.removesuffix(".out")
        
        if warning and neg_freqs_found:
            print(f"WARNING: Imaginary frequencies found in file {filename}")
    
    # -------------------------------------------------

    def __read_file_hess(self, filename, overwrite=False):
        with open(filename) as file:
            content = file.read()

        def parse_atoms_hess(content_dict: dict) -> list:
            raw_atoms = content_dict["atoms"].split("\n")
            num_atoms = int(raw_atoms[0])
            atoms = []
            for line in raw_atoms[1:]:
                line = line.split()
                atom = {}
                atom["element"]=line[0]
                atom["mass"]=float(line[1])
                atom["coords"]=np.array([float(x) for x in line[2:]])*0.529177249
                atoms.append(atom)
            return atoms

        def parse_freqs_hess(content_dict: dict) -> np.array:
            raw_freqs = content_dict["vibrational_frequencies"].split("\n")
            num_modes = int(raw_freqs[0])
            freqs = np.zeros((num_modes))
            for mode, line in enumerate(raw_freqs[1:]):
                freqs[mode] = float(line.split()[1])/8065.54429
            return freqs     
        
        def parse_normal_modes_hess(content_dict: dict) -> np.array:
            raw_modes = content_dict["normal_modes"].split("\n")
            num_modes = int(raw_modes[0].split()[1])
            normal_modes = np.zeros((num_modes,num_modes))

            for line in raw_modes[1:]:
                if line == "#":
                    break
                line = line.split()
                if line[-1].find(".") < 0:
                    col = int(line[0])
                    continue

                row = int(line[0])
                for linecol in range(0,len(line[1:])):
                    normal_modes[row,col+linecol]=float(line[linecol+1])

            normal_modes = np.reshape(normal_modes, (3, num_modes // 3, num_modes),order="F")
            return normal_modes

        def parse_hessian_hess(content_dict: dict) -> np.array:
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

        def parse_ddipole_hess(content_dict: dict) -> np.array:
            raw_dmu = content_dict["dipole_derivatives"].split("\n")
            num_modes = int(raw_dmu[0])
            dipole_derivatives = np.zeros((3,num_modes))
            for mode, line in enumerate(raw_dmu[1:]):

                if "#" in line:
                    continue
                dipole_derivatives[:,mode] = [float(x) for x in line.split()]
            return dipole_derivatives

        def parse_ir_hess(content_dict: dict) -> np.array:
            raw_ir = content_dict["ir_spectrum"].split("\n")
            num_modes = int(raw_ir[0])
            ir_spectrum = np.zeros((6,num_modes))
            for mode, line in enumerate(raw_ir[1:]):
                ir_spectrum[:,mode] = [float(x) for x in line.split()]
            ir_spectrum = np.insert(ir_spectrum, 3, 0., axis=0)
            ir_spectrum[3,:] = np.sum(ir_spectrum[4:,:]**2,axis=0)
            return ir_spectrum
        
        content_dict = {}
        for section in content.split("$")[1:]:
            section = section.split("\n")
            content_dict[section[0]] = "\n".join([line for line in section[1:] if len(line) != 0])

        self.__update_atoms(parse_atoms_hess(content_dict),overwrite)
        
        freqs = parse_freqs_hess(content_dict)
        mask = freqs == 0
        if type(self.freqs) == type(None) or overwrite:
            self.freqs = self.__remove_zero_freqs(freqs,mask)
        if type(self.normal_modes) == type(None) or overwrite:
            self.normal_modes = self.__remove_zero_freqs(parse_normal_modes_hess(content_dict),mask)
        if type(self.hess) == type(None) or overwrite:
            self.hess = parse_hessian_hess(content_dict)
        if type(self.ddipole) == type(None) or overwrite:
            self.ddipole = self.__remove_zero_freqs(parse_ddipole_hess(content_dict),mask)
        if type(self.ir) == type(None) or overwrite:
            self.ir = self.__remove_zero_freqs(parse_ir_hess(content_dict),mask)
            
        if type(freqs) is not type(None):
            neg_freqs_found = np.sum(freqs < 0)

        return neg_freqs_found
    
    # -------------------------------------------------

    def __read_file_out(self, filename, overwrite = False):
        with open(filename) as file:
            content = file.read()

        def parse_success_out(content: str) -> bool:
            if content.find('****ORCA TERMINATED NORMALLY****'):
                return True
            else:
                return False

        def parse_input_out(content: str) -> str:
            input_lines = content.split("INPUT FILE\n"+"="*80+"\n")[1].split("****END OF INPUT****")[0].split("\n")[1:-1]
            input_lines = [line[6:] for line in input_lines]
            return "\n".join(input_lines)

        def parse_charge_multiplicity_out(input_str: str) -> (int, int):
            for line in input_str.split("\n"):
                if len(line) > 3:
                    if line[0] == "*":
                        line = line[1:].split()
                        return int(line[1]), int(line[2])
            return None, None
        
        def parse_energy_out(content: str) -> float:
            try:
                #return float(content.split("FINAL SINGLE POINT ENERGY")[-1].split("\n")[0])*27.2114
                return float(content.split("TOTAL SCF ENERGY\n----------------\n\nTotal Energy")[-1].split("\n")[0].split()[-2])
            except:
                return None
                
        def parse_UKS_out(content: str) -> bool:
            try:
                return content.split(" Hartree-Fock type      HFTyp           .... ")[-1].split()[0] == 'UHF'
            except:
                return None
        
        def parse_atoms_out(content: list) -> list:
            coords_raw = content.split("CARTESIAN COORDINATES (A.U.)\n")[-1]
            coords_raw = coords_raw.split("INTERNAL COORDINATES (ANGSTROEM)")[0]
            atoms=[]
            for line in coords_raw.split("\n")[2:-3]:
                line = line.split()
                atom = {}
                atom["element"]=line[1]
                atom["mass"]=float(line[4])
                atom["coords"]=np.array([float(x) for x in line[5:]])*0.529177249
                atoms.append(atom)
            return atoms

        def parse_occupation_energies_out(content: str, UKS: bool):

            def occupations_energies_ORBEN(content,spin=0):
                content = content.split("ORBITAL ENERGIES\n----------------\n")[-1]
                
                marker1 = '*Only the first 10 virtual orbitals were printed.'
                marker2 = '------------------'
                lines = content.split(marker2)[0].split(marker1)[0].split('\n\n')[spin].split('\n')[2:]

                occupations = np.zeros(len(lines))
                energies = np.zeros(len(lines))
                for line in lines:
                    if len(line):
                        occupations[int(line.split()[0])] = float(line.split()[1]) 
                        energies[int(line.split()[0])] = float(line.split()[3])

                return occupations, energies

            def occupations_energies_MO(MO_block,num_orbs):
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
            if len(content) == 1:
                occ, en = occupations_energies_ORBEN(content)
                if UKS:
                    occ1, en1 = occupations_energies(content,spin=1)
                    occ = np.stack([occ[:occ1.shape[0]],occ1], axis=1)
                    en = np.stack([en[:en1.shape[0]],en1], axis=1)

                return occ, en
            else:
                num_orbs = int(content[-2].split(' Basis Dimension        Dim             ....')[-1].split('\n')[0])
                MO_blocks = content[-1].split('\n\n\n')[0].split('\n\n')
                
                if UKS:
                    occ_alpha, en_alpha = occupations_energies_MO(MO_blocks[0],num_orbs)
                    occ_beta, en_beta = occupations_energies_MO(MO_blocks[1],num_orbs)
                    occ = np.stack([occ_alpha,occ_beta], axis=1)
                    en = np.stack([en_alpha,en_beta], axis=1)
                else:
                    occ, en = occupations_energies_MO(MO_blocks[0],num_orbs)

                return occ, en

        def parse_freqs_out(content: str) -> np.array:
            if "VIBRATIONAL FREQUENCIES" not in content:
                return None

            raw_freqs = content.split("VIBRATIONAL FREQUENCIES")[1]
            raw_freqs = raw_freqs.split("NORMAL MODES")[0]

            freqs = []
            for mode, line in enumerate(raw_freqs.split("\n")[1:]):
                if "cm**-1" in line:
                    freqs.append(float(line.split()[1])/8065.54429)
            return np.array(freqs)
        
        def parse_normal_modes_out(content: str) -> np.array:
            if "NORMAL MODES" not in content:
                return None

            raw_modes = content.split("NORMAL MODES")[1]
            raw_modes = raw_modes.split("IR SPECTRUM")[0]

            num_modes = int(raw_modes.split(" "*18)[1].split("\n")[-2].split()[0])+1
            normal_modes = np.zeros((num_modes,num_modes))

            for line in raw_modes.split("\n")[7:-3]:
                line = line.split()
                if not len(line):
                    continue
                if line[-1].find(".") < 0:
                    col = int(line[0])
                    continue

                row = int(line[0])
                for linecol in range(0,len(line[1:])):
                    normal_modes[row,col+linecol]=float(line[linecol+1])

            normal_modes = np.reshape(normal_modes, (3, num_modes // 3, num_modes),order="F")
            normal_modes.shape 
            return normal_modes

        def parse_ir_out(content: str) -> np.array:
            if "IR SPECTRUM" not in content:
                return None

            raw_modes = content.split("IR SPECTRUM\n-----------\n\n")[1]
            raw_modes = raw_modes.split("\n\n")[0]

            raw_modes = raw_modes.split("\n")[3:]
            num_modes = len(raw_modes)
            ir_spectrum = np.zeros((7,num_modes))

            for m, line in enumerate(raw_modes):
                line = line.replace("(","").replace(")","").split()
                ir_spectrum[:,m] = np.array([float(x) for x in line[1:]])

            return ir_spectrum
            
        def parse_NTOs_out(content: str) -> list:
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
        
        def parse_absorption_dipole_out(content: str) -> list:
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

        def parse_TDstates_out(content: str) -> list:
            if not content:
                return None
                
            content = content.split("TD-DFT/TDA EXCITED STATES")
            if len(content) == 1:
                TDA = False
                content = content[0].split("TD-DFT EXCITED STATES")
                if len(content) == 1:
                    return None
            else:
                TDA = True
            content = content[-1].split("\n\n\n")[0]
            content = content.split("STATE")[1:]

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
        
        def parse_dtdip_out(content: str):
            if content.find("Computing transition dipole derivatives") < 0:
                return None, None
            else:
                ref = content.split("Reference TDIP (x,y,z, a.u.):")[1].split("\n\n")[0].replace(")","").replace("(","").split(",")
                ref_tdip = np.array([float(a.strip().split()[0]) for a in ref])

                num_modes = int(content.split("will need at least")[-1].split("displacements")[0])

                lines = content.split("\nFinal check on derivatives:")[0].split("\t<<Displacing mode ")[1:]
                tdip = np.zeros((3,num_modes))
                tdip_r = np.zeros_like(tdip)
                tdip_r[:,:] = ref_tdip[:,None]
                scale = np.ones(num_modes)
                for line in lines:
                    mode = int(line.split()[0])
                    sign = line.split("(")[1].split(")")[0]
                    itdip = [float(a.strip().split()[0]) for a in line.split("tdip = ")[1].replace(")","").replace("(","").split(",\n")]
                    if sign == "+":
                        tdip[:,mode-1] = itdip
                    elif sign == "-":
                        tdip_r[:,mode-1] = itdip
                        scale[mode-1] = 2.
                return ref_tdip, (tdip - tdip_r)/scale
        
        self.success = parse_success_out(content)
        
        if self.energy == None or overwrite:
            self.energy = parse_energy_out(content)
        
        self.input_str = parse_input_out(content)
        charge, multiplicity = parse_charge_multiplicity_out(self.input_str)
        if self.charge == None or overwrite:
            self.charge = charge
        if self.multiplicity == None or overwrite:
            self.multiplicity = multiplicity
        if self.UKS == None or overwrite:
            self.UKS = parse_UKS_out(content)

        if type(self.occupation) == type(None) or overwrite:
            self.occupation, self.MO_energies = parse_occupation_energies_out(content,self.UKS)

        if self.TDstates == None or overwrite:
            TDstate_list = parse_TDstates_out(content)
            absorptionDipole_list = parse_absorption_dipole_out(content)
            
            if (type(TDstate_list) != type(None)) and (type(absorptionDipole_list) != type(None)):
                self.TDstates = [ b | a for a,b in zip(absorptionDipole_list,TDstate_list)]
        
        if self.NTOs == None or overwrite:
            self.NTOs = parse_NTOs_out(content)
        
        if self.dtdip == None or overwrite:
            self.tdip, self.dtdip = parse_dtdip_out(content)
        
        self.__update_atoms(parse_atoms_out(content),overwrite)

        freqs = parse_freqs_out(content)
        if type(freqs) is not type(None):
            mask = freqs == 0
            if type(self.freqs) == type(None) or overwrite:
                self.freqs = self.__remove_zero_freqs(freqs,mask)
            if type(self.normal_modes) == type(None) or overwrite:
                self.normal_modes = self.__remove_zero_freqs(parse_normal_modes_out(content),mask)
            if type(self.ir) == type(None) or overwrite:
                self.ir = parse_ir_out(content)
                
        if type(freqs) is not type(None):
            neg_freqs_found = np.sum(freqs < 0)
        else:
            neg_freqs_found = 0

        return neg_freqs_found
    # -------------------------------------------------
 
    def load_gbw(self,overwrite = False):
        assert type(self.gbw) is not type(None), "Now gbw file defined"
        self.__gbw2json()   

        def parse_atoms_json(json: dict) -> list:
            atoms = []
            for json_atom in json['Molecule']['Atoms']:
                atom = {}
                atom["element"]=json_atom['ElementLabel']
                atom["nuclearcharge"]=json_atom['NuclearCharge']
                atom["coords"]=np.array(json_atom['Coords'])
                atoms.append(atom)
            return atoms

        def parse_occupation_json(json: dict) -> np.array:
            n_orbs = len(json["Molecule"]['MolecularOrbitals']['MOs'])
            occ = np.zeros((n_orbs,))
            for i in range(n_orbs):
                occ[i] = json["Molecule"]["MolecularOrbitals"]['MOs'][i]['Occupancy']
            if occ[0] == 1:
                occ = occ.reshape((n_orbs//2),2,order='F')
            return occ
        
        def parse_MO_energies_json(json: dict, UKS: bool) -> np.array:
            n_orbs = len(json["Molecule"]['MolecularOrbitals']['MOs'])
            en = np.zeros((n_orbs,))
            for i in range(n_orbs):
                en[i] = json["Molecule"]["MolecularOrbitals"]['MOs'][i]['OrbitalEnergy']
            if UKS:
                en = en.reshape((n_orbs//2),2,order='F')
            return en*27.2114 # hartree to eV

        if self.multiplicity == None or overwrite:
            self.multiplicity = self.json['Molecule']['Multiplicity']
        if self.charge == None or overwrite:
            self.charge = self.json['Molecule']['Charge']

        if type(self.occupation) == type(None) or overwrite:
            self.occupation = parse_occupation_json(self.json)

        if self.UKS == None or overwrite:
            self.UKS = len(self.occupation.shape) == 2
        
        if type(self.MO_energies) == type(None) or overwrite:
            self.MO_energies = parse_MO_energies_json(self.json,self.UKS)
        
        self.__update_atoms(parse_atoms_json(self.json),overwrite)
            
    # -------------------------------------------------
    
    def __update_atoms(self,new_atoms,overwrite=False):
        if self.atoms == None:
            self.atoms = new_atoms
        elif self.atoms != None and overwrite:
            self.atoms = [self.atoms[i] | new_atoms[i] for i in range(len(new_atoms))]
        else:
            self.atoms = [new_atoms[i] | self.atoms[i] for i in range(len(new_atoms))]
    
    # -------------------------------------------------

    @staticmethod
    def __remove_zero_freqs(a: np.array, mask: np.array):
        return np.delete(a, mask, axis=-1)
            
    @property
    def mass(self) -> np.array:
        try: 
             return np.array([atom["mass"] for atom in self.atoms])
        except KeyError:
            try:
                return _default_masses[np.array(element_to_number(self.element))]
            except:
                return None

    @property
    def nuclearcharge(self) -> np.array:
        try: 
             return np.array([atom["nuclearcharge"] for atom in self.atoms])
        except KeyError:
            return None
    
    @property
    def element(self) -> list:
        return [atom["element"] for atom in self.atoms]

    @property
    def element_num(self) -> list:
        return element_to_number([atom["element"] for atom in self.atoms])
    
    @property
    def coords(self) -> np.array:
        return np.array([atom["coords"] for atom in self.atoms]).transpose()

    @property
    def center_of_mass(self) -> np.array:
        return np.sum(self.coords*self.mass[None,:], axis=1)/np.sum(self.mass)

    @property
    def coords_masscentered(self) -> np.array:
        return self.coords - self.center_of_mass[:,None]
    
    @property
    def mwnm(self) -> np.array:
        try:
            mwc = self.normal_modes*np.sqrt(self.mass[None,:,None])
            renorm = np.sqrt(np.sum(mwc**2,axis=(0,1)))
            return mwc/renorm
        except:
            return None
    
    @property
    def num_electrons(self) -> int:
        if type(self.occupation) != type(None):
            return int(np.sum(self.occupation))
        else:
            return None
    
    @property
    def num_alpha(self) -> int:
        if type(self.occupation) == type(None):
            return None
        elif self.UKS:
            return int(np.sum(self.occupation[:,0]))
        else:
            return int(np.sum(self.occupation[:])/2)
    
    @property
    def num_beta(self) -> int:
        if type(self.occupation) == type(None):
            return None
        elif self.UKS:
            return int(np.sum(self.occupation[:,1]))
        else:
            return int(np.sum(self.occupation[:])/2)
    
    @property
    def homo(self) -> int:
        occ = self.occupation
        assert len(occ.shape) == 1, "Molecule is not restricted!"
        return int(np.sum(occ == 2.0)-1)

    @property
    def lumo(self) -> int:
        return self.homo+1
    
    def __sxmo(self,x: int, s: int) -> int:
        occ = self.occupation
        assert len(occ.shape) == 2, "Molecule is not unrestricted!"
        return int(np.sum(occ[:,s] == 1.0)-x)
    
    @property
    def somo(self) -> int:
        return self.somo_a
        
    @property
    def sumo(self) -> int:
        return self.sumo_b

    @property
    def somo_a(self) -> int:
        return self.__sxmo(1,0)
        
    @property
    def sumo_a(self) -> int:
        return self.__sxmo(0,0)

    @property
    def somo_b(self) -> int:
        return self.__sxmo(1,1)
        
    @property
    def sumo_b(self) -> int:
        return self.__sxmo(0,1)

    def set_coords(self, new_coords):
        for i, atom in enumerate(self.atoms):
            atom["coords"] = new_coords[:,i]
            
            
    def get_ntos(self, state: int , threshold = 0) -> list:
        assert type(self.NTOs) is not type(None), "No NTOs found for this object."
        return [(nto['MO'], nto['Spin'], nto['Occupation']) for nto in self.NTOs[state]['NTOs']
                if nto['Occupation'] > threshold]
    
    # -------------------------------------------------

    def dislocated_coords(self, mode: int, amplitude=1.0, mass_centered = False, **kwargs) -> np.array:
        # factor = sqrt(hbar (SI) hbar (eV) / 2 m(u) hw(eV) / 1.66E-27 kg/u ) * 1E10 Ang/m
        factor = np.sqrt(1.054571e-34*6.582119e-16/2/self.mass/1.66e-27/abs(self.freqs[mode]))*1e10
        if mass_centered:
            return self.coords_masscentered + amplitude*factor[None,:]*self.mwnm[:,:,mode]
        else:
            return self.coords + amplitude*factor[None,:]*self.mwnm[:,:,mode]
    
    def write_to_xyz(self, filename: str, coords=None, infostr=""):
        if type(coords) == type(None):
            coords = self.coords
        natoms = len(self.atoms)
        
        content = f"{len(self.atoms)}\n"
        content += infostr+'\n'
        for atom in range(coords.shape[1]):
            content += f'{self.element[atom]:<2}'
            for i in range(3):
                content += '{:>23}'.format(f'{coords[i,atom]:.12e}')
            content += f'\n'
        content += '\n'

        with open(filename, "wb") as file:
            file.write(bytes(content,"utf-8"))
            
    def dislocate_to_xyz(self, mode: int, amplitude=1.0, filename=""):
        coords = self.dislocated_coords(mode,amplitude)
        infostr=f'Dislocation mode={mode} a0={amplitude}'
        if filename == "":
            filename = self.basename + f"_m{mode}_a{amplitude}.xyz"
        self.write_to_xyz(filename, coords, infostr)
        
    def __gbw2json(self, cleanup=True) -> str:
        assert os.path.exists(self.gbw), "Error: *.gbw file not found!"
        
        suffix = self.gbw.split(".")[-1]
        
        try:
            if os.name == 'posix':
                ext = ''
            elif os.name == 'nt':
                ext = '.exe'
            subprocess.run([self.orca_path+"orca_2json"+ext, self.gbw],
                           stdout=subprocess.DEVNULL)
        except:
            print("Could not create JSON from *.gbw file!")
        else:
            jsonfile = self.gbw.removesuffix(suffix)+"json"
            bibtexfile = self.gbw.removesuffix(suffix)+"JSON.bibtex"

            with open(jsonfile, 'r') as file:
                self.json = json.load(file)

            if cleanup:
                os.remove(jsonfile)
                os.remove(bibtexfile)
    
    def coords_to_json(self, coords=None) -> str:
        assert self.json != None, "No JSON template available!"
        
        coords_json = json.loads(json.dumps(self.json))

        if type(coords) == type(None):
            return coords_json
        
        for i, atom in enumerate(coords_json["Molecule"]["Atoms"]):
            coords_json["Molecule"]["Atoms"][i]["Coords"][0]=coords[0,i]
            coords_json["Molecule"]["Atoms"][i]["Coords"][1]=coords[1,i]
            coords_json["Molecule"]["Atoms"][i]["Coords"][2]=coords[2,i]
        return coords_json    

    
    def json2gbw(self,json_str="", jsonfile="", cleanup=True, **kwargs):
        if not len(json_str):
            json_str = self.json
        
        assert json_str != None, "No JSON template available!"

        if len(jsonfile) == 0:
            jsonfile = self.basename + "_json2gbw.json"
        
        with open(jsonfile, 'w') as f:
            json.dump(json_str, f, indent=4)

        try:
            if os.name == 'posix':
                ext = ''
            elif os.name == 'nt':
                ext = '.exe'
            subprocess.run([self.orca_path+"orca_2json"+ext, jsonfile, "-gbw"],
                           stdout=subprocess.DEVNULL)
        except:
             print("Could not create gbw from JSON!")
        else:
            gbwfile = jsonfile.removesuffix(".json")+"_copy.gbw"
            if cleanup:
                os.remove(jsonfile)
            return gbwfile    

    def write_orcaplot_inputfile(self, MO,
                                 spin=0, z_range=None, boundary=5., spacing=3,
                                 cubename="", inputfile="./cubeinput",
                                 plottype=1, fileformat=7, cubedims=None, **kwargs):

        if not len(cubename):
            cubename = f'{self.basename}_MO{MO}{"ab"[spin]}.cub'

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

        if cubedims == None:
            content += self.cube_dims(z_range=z_range, boundary=boundary, spacing=spacing, **kwargs)
        else:
            content += cubedims

        content += "0 0 0\n" # at1 at2 at3
        content += "0 0 0\n" # v1 v1 v1
        content += "0 0 0\n" # v2 v2 v2
        content += "0 0 0\n" # v3 v3 v3

        with open(inputfile,"wb") as f:
            f.write(bytes(content,"utf-8"))
            
        return inputfile, cubename
    
    def cube_dims(self, z_range=None ,boundary=5., spacing=3, coords=None, **kwargs) -> str:
        if type(coords) is type(None):
            coords = self.coords
        
        mins = np.min(coords, axis=1)/0.529177249
        maxs = np.max(coords, axis=1)/0.529177249

        boundary*=1/0.529177249

        def minmax(val1:float, val2:float, boundary=5., spacing=3):
            vmin=min(val1,val2)
            vmax=max(val1,val2)
            b = (np.round((vmax-vmin+2*boundary)*spacing)/spacing-(vmax-vmin))/2
            return vmin-b , vmax+b

        x_range = minmax(mins[0],maxs[0],boundary=boundary,spacing=spacing)
        y_range = minmax(mins[1],maxs[1],boundary=boundary,spacing=spacing)

        if z_range == None:
            z_range = minmax(mins[2],maxs[2],boundary=boundary,spacing=spacing)
        elif len(z_range) == 2:
            z_range = (z_range[0]/0.529177249,z_range[1]/0.529177249)
        else:
            z_range = (z_range[0]/0.529177249,z_range[0]/0.529177249+1./spacing)
            z_range = minmax(z_range[0],z_range[1],boundary=0.,spacing=spacing)

        nx=int(np.round((x_range[1]-x_range[0])*spacing)+1)
        ny=int(np.round((y_range[1]-y_range[0])*spacing)+1)
        nz=int(np.round((z_range[1]-z_range[0])*spacing)+1)

        header = ""
        header += f"{nx} {ny} {nz}\n"
        header += f"{x_range[0]:.6f} {x_range[1]:.6f}\n"
        header += f"{y_range[0]:.6f} {y_range[1]:.6f}\n"
        header += f"{z_range[0]:.6f} {z_range[1]:.6f}\n"

        return header
    
    def make_cube(self, MO, gbw_file = None, json = None, cleanup = True, **kwargs) -> str:
        
        assert type(gbw_file) is type(None) or type(json) is type(None), "JSON and gbw file path given. Please decide which one to use."
        
        deleteGBW = False
        coords = None
        if type(gbw_file) is type(None) and type(json) is type(None):
            assert self.gbw != None, "No gbw file defined with this molecule. Use self.add_source(gbw_file) to add."
            gbw_file = self.gbw
            
        elif type(json) is not type(None):
            gbw_file = self.json2gbw(json, cleanup=cleanup, **kwargs)
            coords = np.array([atom["Coords"] for atom in json["Molecule"]["Atoms"]]).transpose()
            deleteGBW = cleanup

        else:
            assert os.path.exists(gbw_file), "Error: *.gbw file not found!"
              
        inputfile, cubename = self.write_orcaplot_inputfile(MO,coords=coords,**kwargs)
        
        try:
            if os.name == 'posix':
                ext = ''
            elif os.name == 'nt':
                ext = '.exe'
            subprocess.run([self.orca_path+"orca_plot"+ext, gbw_file, inputfile],
                           stdout=subprocess.DEVNULL)
        except:
            print("Could not create cube file!")
        
        if cleanup:
            os.remove(inputfile)
        if deleteGBW:
            os.remove(gbw_file)

        return cubename
        
    def __str__(self):
        return self.filename
    
    def __dir__(self):
        all_attrs = super().__dir__()
        return [attr for attr in all_attrs if not attr.startswith("_")]

_default_masses = np.array([
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

def element_to_number(symbol):
    try:
        return _element_symbols.index(symbol)
    except ValueError:
        try:
            return np.array([_element_symbols.index(s) for s in symbol])
        except ValueError:
            return None

def number_to_element(num):
    try:
        return _element_symbols[num]
    except ValueError:
        try:
            return [_element_symbols[n] for n in num]
        except ValueError:
            return None