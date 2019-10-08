# -*- coding: utf-8 -*-

"""
This file contains the definition of the EnsembleNV class.
   
Copyright (C) 2019  Marco Capelli

NVsim is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from collections import Iterable

from .singleNV import SingleNV

class EnsembleNV(object):
    
    def __init__(self, NVaxis = (0,0,1), number = 100, hamiltonian = 'default', **kwargs):
        if number < 1:
            number = 100
        self.size = number
        if 0 < number < 4:
            self.NV_array = [None for idx in range(number)]
        else:
            self.NV_array = [None, None, None, None]
        self.set_ensemble(NVaxis, **kwargs)
        
    def is_ensemble(self):
        return True
            
    def _tetrahedron_axes(self):
        theta = 2*np.arccos(1/np.sqrt(3))
        reference_axes = np.zeros((4, 3))
        reference_axes[0] = [0., 0., 1.]
        reference_axes[1] = [np.sin(theta), 0., np.cos(theta)]
        reference_axes[2] = [np.sin(theta)*np.cos(2*np.pi/3), np.sin(theta)*np.sin(2*np.pi/3), np.cos(theta)]
        reference_axes[3] = [np.sin(theta)*np.cos(4*np.pi/3), np.sin(theta)*np.sin(4*np.pi/3), np.cos(theta)]
        return reference_axes
    
    def set_ensemble(self, axis = (0,0,1), **kwargs):
        reference_axes = self._tetrahedron_axes()
        axis = np.array(axis / np.linalg.norm(axis), dtype = float)
        azimuth = np.arccos(axis[2])
        if azimuth == 0:
            polar = 0.
        else:
            polar = np.arccos(axis[0] / np.sin(azimuth))
        rotation_matrix_y = np.array([[np.cos(azimuth),0.,-np.sin(azimuth)], [0.,1.,0.], [np.sin(azimuth),0.,np.cos(azimuth)]])
        rotation_matrix_z = np.array([[np.cos(polar), -np.sin(polar), 0.], [np.sin(polar), np.cos(polar), 0.], [0.,0.,1.]])
        for idx, _axis in enumerate(self.NV_array):
            new_axis = rotation_matrix_z @ rotation_matrix_y @ reference_axes[idx]
            temp_NV = SingleNV(NVaxis = new_axis, **kwargs)
            self.NV_array[idx] = temp_NV
    
    @property        
    def Hamiltonian(self):
        return self.NV_array[0].Hamiltonian
    
    @property
    def H_dimension(self):
        return self.NV_array[0].H_dimension
    
    @property
    def external_params(self):
        return self.NV_array[0].external_params
    
    @property
    def operators(self):
        return self.NV_array[0].operators
    
    @property
    def eigenstates(self):
        return self.NV_array[0].eigenstates
    
    @property
    def function_params(self):
        return self.NV_array[0].function_params
    
    def __getitem__(self, key):
        return self.NV_array[0]
    
    def set_Hamiltonian(self, hamiltonian = 'default'):
        for nv in self.NV_array:
            nv.set_Hamiltonian(hamiltonian)
            
    def set_params(self, **params):
        for nv in self.NV_array:
            nv.set_params(**params)
            
    def add_function2param(self, paramName, function = None, reset = True, **functionParamDict):
        for nv in self.NV_array:
            nv.add_function2param(paramName, function = function, reset = reset, **functionParamDict)
            
    def update_function2param(self, paramName, **functionParamDict):
        for nv in self.NV_array:
            nv.update_function2param(paramName, **functionParamDict)
            
    def static_solution(self, operator = 'emission', returnStates = None, **kwargs):
        return self._common_solution(operator = operator, returnStates = returnStates, **kwargs)
    
    def time_solution(self, time, startState = None, operator = 'emission', returnStates = False, **kwargs):
        if not isinstance(time, Iterable):
            time = [time]
        emission = np.zeros(np.size(time))
        if len(np.shape(startState)) > 1:
            nv_startState = startState[0]
        else:
            nv_startState = startState
        return_test = self.NV_array[0]._time_solution(0., nv_startState, operator = operator, returnStates = returnStates, **kwargs)
        if isinstance(return_test, tuple):
            returning_stateMatrix = True
            finalStates = np.zeros((np.size(time), np.size(self.NV_array), np.size(return_test[1])))
        else:
            returning_stateMatrix = False
        for idx, t in enumerate(time):
            if returning_stateMatrix:
                emission[idx], finalStates[idx] = self._common_solution(t, startState, operator = operator, returnStates = returnStates, **kwargs)
            else:
                emission[idx] = self._common_solution(t, startState, operator = operator, returnStates = False, **kwargs)

        if returnStates:
            return emission, finalStates
        else:
            return emission
    
    def _common_solution(self, time = None, startState = None, operator = 'emission', returnStates = False, **kwargs):
        for nv_idx, nv in enumerate(self.NV_array):
            if len(np.shape(startState)) > 1:
                nv_startState = startState[nv_idx]
            else:
                nv_startState = startState
            if nv_idx == 0:
                return_test = nv._common_solution(time = time, startState = nv_startState, operator = operator, returnStates = returnStates, **kwargs)
                if isinstance(return_test, tuple):
                    returning_stateMatrix = True
                    finalStates = np.zeros((np.size(self.NV_array), np.size(return_test[1])))
                    emission = return_test[0]
                    finalStates[nv_idx] = return_test[1]
                else:
                    returning_stateMatrix = False
                    emission = return_test
            else:
                if returning_stateMatrix:
                    temp_emission, finalStates[nv_idx] = nv._common_solution(time = time, startState = nv_startState, operator = operator, returnStates = returnStates, **kwargs)
                else:
                    temp_emission = nv._common_solution(time = time, startState = nv_startState, operator = operator, returnStates = returnStates, **kwargs)
                emission += temp_emission
        
        emission = (emission / len(self.NV_array)) * self.size
        if returning_stateMatrix:
            return emission, finalStates
        else:
            return emission
            
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    test_NVs = EnsembleNV(NVaxis = [0,0.05,1])
    test_singlenv = SingleNV(NVaxis = [0,0.05,1])
    x_axis = np.linspace(0.1e-3, 200e-3, 200)
    test_emission = np.zeros_like(x_axis)
    test_single = np.zeros_like(x_axis)
    for pp_idx, powpow in enumerate(x_axis):
        test_emission[pp_idx] = test_NVs.static_solution(excRate = 5e6, magnField = powpow, magnVector = [0, 0, 1])
        test_single[pp_idx] = test_singlenv.static_solution(excRate = 5e6, magnField = powpow, magnVector = [0, 0, 1])
    plt.plot(x_axis, test_emission, 'r-', x_axis, test_single, 'b-')
    
            
        
