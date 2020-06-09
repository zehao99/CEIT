import numpy as np 
import cupy as cp
from .efem import EFEM
import progressbar
import csv
import matplotlib.pyplot as plt
from .utilities import save_parameter
from .utilities import read_parameter
from .utilities import get_config

#Depend on efem.py

class EJAC(object):
    """
    calculate Jaccobian matrix for the problem

    DEFAULT PERMITIVITY IS SET TO 1/10000 SIEMENS

    Parameters: 
                mesh : DICT element mesh See readmesh.py
                electrode_num : INT 
                electrode_centers: electrode_num * (electrode_num - 1) dimension NDARRAY
                radius: FLOAT radius of electrode (half side length)
                frequency: FLOAT input frequency
                mode: 'n' - no initial condition
                      'i' - with initial condition
                first: True - initial calculation
                       False - further calculation

    Functions:

                JAC_calculation(self): Calculate and return JAC Matrix, Auto Save to File

                JAC_calc_sudoku(self): JAC matrix for sudoku problem #NOT IMPLEMENTED

                eit_solve(self, detect_potential, lmbda = 203): Solve inverse problems

                read_JAC_2file(self, filename): Load JAC matrix from file

                save_inv_matrix(self, lmbda = 203): Calculate the inverse matrix and save to file (Preparation for realtime calculation)

                eit_solve_direct(self, detect_potential): High speed reconstruction(Based on inverse matrix MUST RUN save_inv_matrix())

                show_JAC(self): Show the jaccobian in plot

                plot_potential(self, filename = 'potential_Temp'): Plot potentials on every electrode

                plot_sensitivity(self, area_normalization = True): Plot sensitivity map (Sum of columns in JAC_Matrix)
    """
    def __init__(self, mesh, electrode_num, electrode_centers, radius = 0.1, frequency = 20000.0 * 2 * np.pi, mode = 'n', first = False, detection_bound = 40, overall_capacitance = 0):
        self.config = get_config()
        self.mode = mode
        self.first = first
        self.detection_bound = detection_bound
        self.overall_capacitance = overall_capacitance
        #Create FEM class
        #Set overall permitivity as the conductive sheet
        permi = 1/10000
        self.fwd_FEM = EFEM(mesh, electrode_num,electrode_centers, radius, frequency, perm = permi)
        self.electrode_num = electrode_num 
        #Initialize Matrix
        self.pattern_num = electrode_num * (electrode_num - 1)
        self.elem_num = self.fwd_FEM.elem_num
        self.electrode_original_potential = np.zeros((self.pattern_num))
        #Choose detection_area
        self.calc_detection_elements()
        self.JAC_matrix = np.zeros((self.pattern_num,self.elem_num))
        if self.mode == 'n':
            self.fwd_FEM.reset_capacitance(overall_capa = self.overall_capacitance) #Set overall initial value
        elif self.mode == 'i':
            self.fwd_FEM.reset_capa_to_initial(0)# Set initial value
        else:
            raise Exception('No Such Mode, Please check.')
        self.initial_capacitance = np.copy(self.fwd_FEM.elem_capacitance)
        self.calc_origin_potential()
    
    def calc_origin_potential(self):
        """
        calculate origin potential vector
        """
        for i in range(self.electrode_num):
            count = 0
            _, _, electrode_potential = self.fwd_FEM.calculation(i)
            for m in range(self.electrode_num):
                if m != i:
                    self.electrode_original_potential[i * (self.electrode_num - 1) + count] = np.abs(electrode_potential[m]) 
                    count += 1

    def JAC_calculation(self, calc_from = 0, calc_end = 16,capacitance_change = 1e-3):
        """
        calculate JAC matrix
        """
        if self.first:
            if calc_from > 0:
                self.read_JAC_np()
            #Changed matrix
            print('This might take a whilllllllllle............')
            for i in range(self.electrode_num):
                if i < calc_from:
                    continue
                if i >= calc_end:
                    break
                self.save_JAC_np()
                print("iteration: " + str(i) + "     If you want to break, press Ctrl + C. But REMEMBER THIS iteration num!")
                for j in progressbar.progressbar(range(self.elem_num)):
                    #capacitance change in JAC
                    self.fwd_FEM.elem_capacitance[j] += capacitance_change
                    _, _, electrode_potential = self.fwd_FEM.calculation(i)
                    count = 0
                    for m in range(self.electrode_num):
                        if m != i:
                            self.JAC_matrix[i * (self.electrode_num - 1) + count][j] = np.abs(electrode_potential[m])
                            count += 1
                    self.fwd_FEM.elem_capacitance[j] -= capacitance_change
            #Minus and broadcast original value calculate differiential value
            #self.JAC_matrix = (self.JAC_matrix - np.reshape(self.electrode_original_potential, (self.pattern_num,1))) / capacitance_change
            self.save_JAC_np()
            print('Congrats! You made it!')
            return self.JAC_matrix
        else:
            self.read_JAC_np()
            return self.JAC_matrix

    def calc_detection_elements(self):
        original_element = self.fwd_FEM.elem
        original_x = self.fwd_FEM.elem_param[:,7]
        original_y = self.fwd_FEM.elem_param[:,8]
        corres_index = []
        new_elem = []
        for i, element in enumerate(original_element):
            if np.abs(original_x[i]) < self.detection_bound and np.abs(original_y[i]) < self.detection_bound:
                corres_index.append(i)
                new_elem.append(element)
        self.detection_index = np.array(corres_index)
        self.detection_elem = np.array(new_elem)

    def sudoku_generate_centers(self, dim, length):
        """
        Generate center coordinates list

        Parameters: dim INT dimension of sudoku
                    length FLOAT length of the outside square area
        
        Return: centerlist NDArray dim^2 * 2
                radius     FLOAT radius for changing capacitance
        """
        single_len = length / dim
        centerlist = np.zeros((dim * dim,2))
        for y in range(dim):
            for x in range(dim):
                centerlist[x + y * dim, :] = [single_len * x, - single_len * y]
        centerlist = centerlist + [ - (length - single_len)/2 , (length - single_len)/2 ]
        return centerlist , single_len / 2

    def JAC_calc_sudoku(self,dim = 3,length = 90):
        #set sudoku dimension to 3
        file_name = "fwd_sudoku00000"
        centerlist , radius = self.sudoku_generate_centers(dim, length)
        index_len = dim * dim
        self.JAC_matrix = np.zeros((self.pattern_num, index_len))
        capacitance_change = 2e-13
        for i in progressbar.progressbar(range(self.electrode_num)):
            self.save_JAC_2file()
            for j in range(index_len):
                self.fwd_FEM.change_add_capa_geometry(centerlist[j],radius,capacitance_change,shape = "square")
                _, _, electrode_potential = self.fwd_FEM.calculation(i)
                count = 0
                for m in range(self.electrode_num):
                    if m != i:
                        self.JAC_matrix[i * (self.electrode_num - 1) + count][j] = np.abs(electrode_potential[m])
                        count += 1
                if self.mode == 'n':
                    self.fwd_FEM.reset_capacitance(overall_capa = self.overall_capacitance)
                elif self.mode == 'i':
                    self.fwd_FEM.reset_capa_to_initial(9e-13)# Set initial value
        self.JAC_matrix = (self.JAC_matrix - np.reshape(self.electrode_original_potential, (self.pattern_num,1))) / capacitance_change
        self.save_JAC_2file()

    def eliminate_non_detect_JAC(self):
        orig_JAC = np.copy(self.JAC_matrix.T)
        new_JAC = []
        for j in self.detection_index:
            new_JAC.append(orig_JAC[j,:])
        new_JAC = np.array(new_JAC)
        #save_parameter(new_JAC,'detect_JAC')
        return new_JAC.T

    def eit_solve(self, detect_potential, lmbda = 295):
        """
        detect_potential: electrode_num * (electrode_num - 1) elements NDArray vector

        lmbda: FLOAT regularization parameter
        """
        #self.normalize_sensitivity()
        J = self.eliminate_non_detect_JAC() - 1
        Q = np.eye(J.shape[1]) #* area_list
        #Q = np.diag(np.dot(J.T,J))
        delta_V = detect_potential - np.copy(self.electrode_original_potential)
        #capacitance_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V)
        #capacitance_predict = np.dot(J.T / np.reshape(area_list,(area_list.shape[0],1)), delta_V)
        capacitance_predict = self.Msolve_gpu(J, Q, lmbda, delta_V)
        #self.plot_potential(delta_V, orig_ratio = 0)
        return capacitance_predict
    
    def eit_solve_4electrodes(self, detect_potential, lmbda = 90):
        slice_list=[]
        for i in [2,6,10,14]:
            for j in [2,6,10,14]:
                if j == i:
                    pass
                if j < i:
                    slice_list.append(i * 15 + j)
                if j > i:
                    slice_list.append(i * 15 + j -1)
        J = self.eliminate_non_detect_JAC() - 1
        J = J[slice_list,:]
        Q = np.eye(J.shape[1])
        delta_V = detect_potential - np.copy(self.electrode_original_potential)
        delta_V = np.copy(delta_V[slice_list])
        capacitance_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V) * 1e-4
        return capacitance_predict
    
    def eit_solve_4electrodes_delta_V(self, delta_V, lmbda = 90):
        slice_list=[]
        for i in [2,6,10,14]:
            for j in [2,6,10,14]:
                if j == i:
                    pass
                if j < i:
                    slice_list.append(i * 15 + j)
                if j > i:
                    slice_list.append(i * 15 + j -1)
        J = (self.eliminate_non_detect_JAC() - 1)
        J = J[slice_list,:]
        Q = np.eye(J.shape[1])
        capacitance_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V) * 1e-4
        return capacitance_predict
    
    def eit_solve_8electrodes(self, detect_potential, lmbda = 265):
        slice_list=[]
        for i in [0,2,4,6,8,10,12,14]:
            for j in [0,2,4,6,8,10,12,14]:
                if j == i:
                    pass
                if j < i:
                    slice_list.append(i * 15 + j)
                if j > i:
                    slice_list.append(i * 15 + j -1)
        J = self.eliminate_non_detect_JAC() - 1
        J = J[slice_list,:]
        Q = np.eye(J.shape[1])
        delta_V = detect_potential - np.copy(self.electrode_original_potential)
        delta_V = delta_V[slice_list]
        capacitance_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V)
        return capacitance_predict
    
    def eit_solve_8electrodes_delta_V(self, delta_V, lmbda = 265):
        slice_list=[]
        for i in [0,2,4,6,8,10,12,14]:
            for j in [0,2,4,6,8,10,12,14]:
                if j == i:
                    pass
                if j < i:
                    slice_list.append(i * 15 + j)
                if j > i:
                    slice_list.append(i * 15 + j -1)
        J = self.eliminate_non_detect_JAC() - 1
        J = J[slice_list,:]
        Q = np.eye(J.shape[1])
        capacitance_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V)
        return capacitance_predict

    def eit_solve_delta_V(self, delta_V, lmbda = 295):
        """
        detect_potential: electrode_num * (electrode_num - 1) elements NDArray vector

        lmbda: FLOAT regularization parameter
        """
        J = self.eliminate_non_detect_JAC()
        Q = np.eye(J.shape[1]) #* area_list
        capacitance_predict = self.Msolve_gpu(J, Q, lmbda, delta_V)
        return capacitance_predict

    def save_inv_matrix(self, lmbda = 203):
        """
        Calculate and save inverse matrix according to lmbda
        """
        J = self.eliminate_non_detect_JAC() - 1
        Q = np.eye(J.shape[1])
        JAC_inv = np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T)
        np.save(self.config["folder_name"] + '/inv_mat.npy', JAC_inv)
    
    def read_inv_matrix(self):
        return np.load(self.config["folder_name"] + '/inv_mat.npy')
    
    def eit_solve_direct(self, detect_potential):
        """
        Use saved inverse matrix to solve
        """
        JAC_p = self.read_inv_matrix()
        delta_V = detect_potential - np.copy(self.electrode_original_potential)
        capacitance_predict = np.dot(JAC_p,delta_V)
        return capacitance_predict

    def Msolve_gpu(self, J, Q, lmbda, delta_V):
        """
        GPU Acceleration
        """
        J_g = cp.asarray(J)
        J_g_T = cp.asarray(J.T)
        Q_g = cp.asarray(Q)
        delta_V_g = cp.asarray(delta_V)
        capacitance_predict = cp.dot(cp.dot(cp.linalg.inv(cp.dot(J_g_T, J_g) + lmbda ** 2 * Q_g ), J_g_T), delta_V_g)
        capacitance_predict = cp.asnumpy(capacitance_predict)
        return capacitance_predict

    def save_JAC_np(self):
        np.save(self.config["folder_name"]+'/'+'JAC_cache.npy', self.JAC_matrix)

    def read_JAC_np(self):
        self.JAC_matrix = np.load(self.config["folder_name"]+'/'+'JAC_cache.npy')

    def save_JAC_2file(self):
        """
        Save jaccobian matrix to file
        """
        with open(self.config["folder_name"]+'/'+'jac_cache.csv', "w", newline= '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            for row in self.JAC_matrix:
                writer.writerow(row)        
    
    def read_JAC_2file(self, mode = 'normal'):
        """
        Read jaccobian matrix from file
        parameter: filename STRING
        """
        if mode == 'normal':
            with open(self.config["folder_name"]+'/'+'jac_cache.csv', newline= '') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',')
                for i, line in enumerate(reader):
                    self.JAC_matrix[i] = line
        elif mode == 'sudoku':
            self.JAC_matrix = np.zeros((self.pattern_num,9))
            with open(self.config["folder_name"]+'/'+'jac_cache.csv', newline= '') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',')
                for i, line in enumerate(reader):
                    self.JAC_matrix[i] = line
        else:
            raise Exception('Mode do not exist.')

    def show_JAC(self):
        """
        Show JAC matrix
        """
        plt.imshow(self.JAC_matrix[:,:])
        plt.show()

    def get_sensitivity_list(self):
        '''
        Sum up all numbers under different pattern for a single element in JAC_matrix
        '''
        sensitivity = np.sum(self.JAC_matrix, axis = 0)
        return sensitivity

    def normalize_sensitivity(self):
        '''
        Normalize JAC by sensitivity
        Didn't work.....
        '''
        sensitivity = self.get_sensitivity_list()
        self.JAC_matrix = self.JAC_matrix / sensitivity.T * np.mean(sensitivity)

    def delete_outside_detect(self, list_c):
        '''
        Input a all-element-wise list 
        Return elements remained in detection domain
        '''
        list_c = np.array(list_c)
        if list_c.ndim > 1:
            new_list_c = np.zeros((self.detection_index.shape[0], list_c.shape[1]))
            for i,j in enumerate(self.detection_index):
                new_list_c[i] = list_c[j]
            return new_list_c
        elif list_c.ndim == 1:
            new_list_c = np.zeros((self.detection_index.shape[0]))
            for i,j in enumerate(self.detection_index):
                new_list_c[i] = list_c[j]
            return new_list_c
        else:
            raise Exception("Transfer Shape Not Correct")


    def delete_minus_element(self, list_c):
        '''
        This is something bad
        '''
        out_list = list_c
        for i,j in enumerate(list_c):
            if j < 0:
                out_list[i] = 0
        return out_list

    def plot_sensitivity(self, area_normalization = True):
        '''
        Plot sensitivity map
        '''
        if area_normalization:
            sensitivity = self.get_sensitivity_list() / self.fwd_FEM.elem_param[:,0]
        else:
            sensitivity = self.get_sensitivity_list()
        points = self.fwd_FEM.nodes
        tri = self.fwd_FEM.elem
        x, y = points[:, 0], points[:, 1]
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.tripcolor(x, y, tri, sensitivity, shading='flat')
        fig.colorbar(im)
        ax.set_aspect('equal')
        plt.show()

