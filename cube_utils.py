import numpy as np
import os
from scipy.signal import convolve2d
import subprocess
from more_itertools import chunked

bohr2ang = 0.529177249

class Cube:
    def __init__(self,inputstr: str):
        self.filename = None
        self.data = None
        self.natoms = None
        self.atoms = None
        self.origin = None
        self.n1 = None
        self.n2 = None
        self.n3 = None
        self.vec1 = None
        self.vec2 = None
        self.vec3 = None
        self.nMO = None
        self.vecMO = None
        self.header = None

        try:
            self.filename = inputstr
            header, pointer_position = self.load_cube_header(inputstr)
            self.parse_cube_header(header)
            self.save_cube_data(self.load_cube_data(inputstr,pointer_position))                 
        except:
            try:
                self.parse_cube_header(inputstr)
            except:
                raise ValueError('Inputstr is neither valid file nor valid header.')
   
    @staticmethod
    def load_cube_header(filename: str) -> (str, int):
        assert os.path.exists(filename), "File does not exists."
        with open(filename, 'br') as f:
            header = ""
            for i in range(6):
                header += f.readline().decode("utf-8")
            try:
                nAtoms = abs(int(header.split("\n")[2].split()[0]))
                for i in range(nAtoms+1):
                    header += f.readline().decode("utf-8")
            except:
                raise ValueError('Could not find number of atoms in third line of header!')
            
            return header, f.tell()
    
    def parse_cube_header(self, header: str):
        def parse_cubeheader_line(line:str, factor=1.0, atoms=False):
            line = line.split()
            if atoms:
                try:
                    return int(line[0]), float(line[1]), np.array([float(x) for x in line[2:]])*factor
                except:
                    raise ValueError('Expected string with five numbers.')
                    return None, None                       
            else:
                try:
                    return abs(int(line[0])), np.array([float(x) for x in line[1:]])*factor
                except:
                    raise ValueError('Expected string with four numbers.')
                    return None, None
                    
        try:
            lines = header.split("\n")
            self.header = [lines[0].removesuffix("\n"),lines[1].removesuffix("\n")]
            self.natoms, self.origin = parse_cubeheader_line(lines[2],factor=bohr2ang)
            self.n1, self.vec1 = parse_cubeheader_line(lines[3],factor=bohr2ang)
            self.n2, self.vec2 = parse_cubeheader_line(lines[4],factor=bohr2ang)
            self.n3, self.vec3 = parse_cubeheader_line(lines[5],factor=bohr2ang)
            self.atoms = []
            for i in range(self.natoms):
                atom={}
                element, charge, coords = parse_cubeheader_line(lines[6+i], factor=bohr2ang, atoms=True)
                atom["element"] = element
                atom["charge"] = charge
                atom["coords"] = coords
                self.atoms.append(atom)
            self.nMO, self.vecMO = parse_cubeheader_line(lines[self.natoms+6])
        except:
            raise ValueError('Could not parse string as header.')
        
    @staticmethod
    def load_cube_data(filename: str, pointer_position: int) -> np.array:
        data = []
        with open(filename, 'br') as f:
            f.seek(pointer_position)
            counter = 0
            for line in f:
                line = line.split()
                data += [float(val) for val in line]
        return np.array(data)

    def save_cube_data(self,data: np.array):
        self.data = data.reshape((self.n1,self.n2,self.n3))
        
        
    def tip_wavefuncs(self, tip_height=7., roi=None, workfunction=5.4, **kwargs) -> np.array:
        if roi == None:
            roi = tip_height
        
        d3 = self.vec3[2]#np.sqrt(np.sum(self.vec3**2))

        d1 = self.vec1[0]#np.sqrt(np.sum(self.vec1**2))
        d2 = self.vec2[1]#np.sqrt(np.sum(self.vec2**2))

        tn1 = int(np.ceil(roi/d1))
        tn2 = int(np.ceil(roi/d2))

        tip = np.zeros((2*tn1+1,2*tn2+1))
        dtip = np.zeros_like(tip)
        tipx = np.linspace(-tn1*d1,tn1*d1,2*tn1+1)
        tipy = np.linspace(-tn2*d2,tn2*d2,2*tn2+1)

        kap = np.sqrt(2*9.109e-31*workfunction/1.055e-34/6.582e-16)*1e-10

        for i,j in np.ndindex(tip.shape):
            r = np.sqrt((tipx[i])**2+(tipy[j])**2+(tip_height)**2)
            tip[i,j] = np.exp(-kap*r)/(kap*r)
            dr = np.sqrt((tipx[i])**2+(tipy[j])**2+(tip_height-d3)**2)
            dtip[i,j] = np.exp(-kap*dr)/(kap*dr)

        dtip = (dtip-tip)/d3
        return tip, dtip
        
    def sim_STS(self, tip_height=7., workfunction=5.4, plane=1.5, method="TH", center_mass=True, **kwargs) -> np.array:
        
        if center_mass:
            tip_height += self.center_of_mass[2]
            plane += self.center_of_mass[2]
        
        if tip_height <= plane:
            return self.integration_plane(plane)**2

        if method == "TH":
            return self.extrapolate_WF(tip_height, plane, workfunction)**2
        
        elif method == "Bardeen":
            return self.convolve_WF(tip_height-plane, plane, workfunction, **kwargs)**2

    def convolve_WF(self, tip_height=7., plane=1.5, workfunction=5.4, roi=None, **kwargs) -> np.array:
        psi = self.integration_plane(plane=plane, n=2)
        dpsi = (psi[:,:,1]-psi[:,:,0])/self.vec3[2]#np.sqrt(np.sum(self.vec3**2))
        tip, dtip = self.tip_wavefuncs(tip_height,roi,workfunction)
        
        return convolve2d(psi[:,:,0],dtip,mode='same') - convolve2d(dpsi,tip,mode='same')
    
    def extrapolate_WF(self, tip_height=7., plane=1.5, workfunction=5.4):
        morb_plane = self.integration_plane(plane)[...,0]

        fourier = np.fft.rfft2(morb_plane)
        kx_arr = 2*np.pi*np.fft.fftfreq(morb_plane.shape[0], self.vec1[0])
        ky_arr = 2*np.pi*np.fft.rfftfreq(morb_plane.shape[1], self.vec2[1])
        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr,  indexing='ij')

        fac = 2*9.109383e-31/1.054571e-34/6.582119e-16*1e-20
        kappa = np.sqrt(kx_grid**2 + ky_grid**2 + fac*workfunction)

        dz = tip_height - plane
        return np.fft.irfft2(fourier*np.exp(-kappa*dz), self.data[...,0].shape)

    def integration_plane(self, plane=1.5, n=1) -> np.array:
        dataz = np.linspace(self.origin[2],self.origin[2]+(self.n3-1)*self.vec3[2],self.n3)
        plane_idx = np.searchsorted(dataz, plane)
        return self.data[:,:,plane_idx:plane_idx+n]
    
    @property
    def parameters(self) -> str:
        ns = np.stack([self.natoms*-1,self.n1,self.n2,self.n3])
        vecs = np.stack([self.origin,self.vec1,self.vec2,self.vec3])
        header = ""
        for i in range(4):
            header += f'{ns[i]:>5}'
            for j in range(3):
                header += '{:>12}'.format(f'{vecs[i,j]/0.529177249:.6f}')
            header += "\n"
        return header

    @property
    def coords(self) -> np.array:
        return np.array([atom["coords"] for atom in self.atoms]).transpose()

    @property
    def elements(self) -> np.array:
        return np.array([int(atom["element"]) for atom in self.atoms])

    @property
    def center_of_mass(self) -> np.array:
        masses = _default_masses[self.elements]
        return np.sum(self.coords*masses[None,:], axis=1)/np.sum(masses)


    def write_cube(self, filename: str, path="./", header=None):
        def two_line_header(header):
            header += '\n\n'
            return '\n'.join(header.split('\n')[:2])+'\n'

        def atoms_string(atoms) -> str:
            out = [""] * len(atoms)
            for a, atom in enumerate(atoms):
                line = [""] * 5
                line[0] = '{:>5}'.format(f'{atom["element"]}')
                line[1] = '{:>12}'.format(f'{atom["charge"]:.6f}')
                for i in range(3):
                    line[2+i] = '{:>12}'.format(f'{atom["coords"][i]/bohr2ang:.6f}')

                out[a] = "".join(line)
            return '\n'.join(out)+'\n'

        def data_string(data) -> str:
            out = [''] * data.shape[0]*data.shape[1]
            for ix in range(data.shape[0]):
                for iy in range(data.shape[1]):
                    out[ix*data.shape[1]+iy] = "\n".join(["".join(['{:>14}'.format(f'{a:.5E}') for a in row])
                                                               for row in list(chunked(data[ix,iy,:],6))])
            return '\n'.join(out)+'\n'
        
        
        with open(os.path.join(path,filename),"wb") as f:    
            if type(header) is type(None):
                f.write(bytes('\n'.join(self.header)+'\n','utf-8'))
                
            else:
                f.write(bytes(two_line_header(header),'utf-8'))

            f.write(bytes(self.parameters,'utf-8'))
            f.write(bytes(atoms_string(self.atoms),'utf-8'))
            f.write(bytes(''.join(['{:>5}'.format(int(a)) for a in [self.nMO]+list(self.vecMO)])+'\n','utf-8'))
            f.write(bytes(data_string(self.data),'utf-8'))




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