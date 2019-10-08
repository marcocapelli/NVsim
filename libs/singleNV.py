# -*- coding: utf-8 -*-

"""
This file contains the definition of the SingleNV class.

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

from os import path as os_path
import numpy as np
from scipy.linalg import expm
import scipy.constants as spConst
from collections import namedtuple, Iterable
import json

#Universal constants definition
Constants = namedtuple('Constants', 'muB ge hbar pi c')
const = Constants(muB = spConst.physical_constants['Bohr magneton'][0],
                  ge = -spConst.physical_constants['electron g factor'][0],
                  pi = spConst.pi,
                  hbar = spConst.hbar,
                  c = spConst.c)

class SingleNV(object):

    Dg = 2.87e9 # Ground state zero field splitting [Hz]
    De = 1.4e9 # Excited state zero field splitting [Hz]
    gamma = const.ge * const.muB / (2*const.pi*const.hbar) # NV electron giromagnetic ratio [Hz/T]
    Leg = 1/(12e-9) # Excited state lifetime [Hz]
    Lts_p1 = 1/(2*12e-9 + 0.9e-9) # Transition rate from spin +1 excited triplet states to metastable singlet state [Hz] (@ 0 mT)
    Lts_m1 = Lts_p1 # Transition rate from spin -1 excited triplet states to metastable singlet state [Hz] (@ 0 mT)
    Lts_0 = 0 # Transition rate from spin 0 excited triplet states to metastable singlet state [Hz] (@ 0 mT)
    Lst_p1 = 1/(219e-9) # Transition rate between metastable singlet state and spin +1 ground triplet states [Hz]
    Lst_m1 = Lst_p1 # Transition rate between metastable singlet state and spin -1 ground triplet states [Hz]
    Lst_0 = Lst_p1 # Transition rate between metastable singlet state and spin 0 ground triplet state [Hz]
    sigma = 0.95e-20 # NV absorption cross-section [m^2]

#    focusArea = 5.655e-13 # Focal spot size (5x obj, 0.1NA) [m^2]

    def __init__(self, NVaxis = (1,1,1), hamiltonian = 'default', T1 = 1/(1e-3), T2 = 0.0, **kwargs):

        self._temp = {}
        self.function_params = {}
        if NVaxis is not None:
            if np.size(NVaxis) == 3:
                self.NVaxis = NVaxis / np.linalg.norm(NVaxis)
            elif np.size(NVaxis) == 1:
                NVaxis = np.deg2rad(NVaxis)
                self.NVaxis = (np.sin(NVaxis), 0, np.cos(NVaxis))
            else:
                self.NVaxis = (1,1,1)
        self.set_Hamiltonian(hamiltonian = hamiltonian)
        self.set_params(Leg = SingleNV.Leg, Lts_p1 = SingleNV.Lts_p1, Lts_m1 = SingleNV.Lts_m1, Lts_0 = SingleNV.Lts_0,
                        Lst_p1 = SingleNV.Lst_p1, Lst_m1 = SingleNV.Lst_m1, Lst_0 = SingleNV.Lst_0,
                        T1 = T1, T2 = T2, bz = 0., bxy = 0., **kwargs)

        # Define functional dependency of certain parameters from Setup parameters. NB! All lambda function should include **kwargs!
        self.add_function2param('Dg', function = lambda nv, freqMW, default, **kwargs: default - freqMW, default = SingleNV.Dg, freqMW = 0.)
        self.add_function2param('De', function = lambda nv, freqMW, default, **kwargs: default - freqMW, default = SingleNV.De, freqMW = 0.)
        self.add_function2param('bz', function = lambda nv, magnField, magnVector, **kwargs: nv.gamma * magnField * np.dot(nv.NVaxis, magnVector))
        self.add_function2param('bxy', function = lambda nv, magnField, magnVector, powerMW, **kwargs: (nv.gamma / np.sqrt(2)) * magnField * (1 - np.dot(nv.NVaxis, magnVector)**2) + powerMW, powerMW = 0.) # add the np.sqrt

    def is_ensemble(self):
        return False

    def set_Hamiltonian(self, hamiltonian = 'default'):
        dirname, filename = os_path.split(os_path.abspath(__file__))
        # TODO: Rewrite to ALWAYS find the correct folder
        filePath = dirname[:-5] + '/matrices/Hamiltonian_{}.json'.format(hamiltonian)
        with open(filePath) as file:
            json_dict = json.load(file)
            self.Hamiltonian = np.array(json_dict['Hamiltonian'])
            self.H_dimension = json_dict['dimension']
            self.external_params = json_dict['parameters']
            self.operators = {}
            self.eigenstates = None
            for key, value in json_dict['operators'].items():
                if key == 'eigenstates':
                    self.eigenstates = np.array(value)
                elif key == 'emission':
                    self.operators[key] = self.Leg * np.array(value)
                else:
                    self.operators[key] = np.array(value)
            if self.eigenstates is None:
                raise KeyError('The Hamiltonian loaded has no eigenstate vector defined.')

    def eval_Hamiltonian(self, **setup):
        str_matrix = self.Hamiltonian.flatten()
        val_matrix = np.array(list(map(eval, str_matrix)))
        return val_matrix.reshape(self.Hamiltonian.shape)

    def set_param(self, name, value, function = None, functionParamDict = None):
        self.__dict__.update({name: value})
        if function is not None:
            self.add_function2param(name, function, functionParamDict)

    def set_params(self, **params):
        for key, value in params.items():
            self.set_param(key, value)

    def add_function2param(self, paramName, function = None, reset = True, **functionParamDict):
        """ If function is None than remove the functional dependency from the parameter
        """
        if function is None:
            try:
                _, _, resetValue = self.function_params.pop(paramName)
                if reset:
                    self.__dict__.update({paramName: resetValue})
            except KeyError:
                print('Parameter {} has already no function associated with it.'.format(paramName))
        else:
            if paramName in self.__dict__:
                resetValue = self.__dict__[paramName]
            else:
                resetValue = 0
            self.function_params.update({paramName: (function, functionParamDict, resetValue)})

    def update_function2param(self, paramName, **functionParamDict):
        self.function_params[paramName][1].update(functionParamDict)

    def _update_changingParams(self, **kwargs):
        for key, values in self.function_params.items():
            values[1].update(kwargs)
            updatedValue = values[0](self, **values[1])
            self.__dict__.update({key: updatedValue})

    def _nullspace(self, matrixA, eps = 1e-2):
        # UV decomposition: A = U * diag(S) * V^H (^H is the complex conjugate).
        #                   S**2 is the vector of the eigenvalues, U and V are the matrices with the eigenvectors as row/columns
        u, s, vh = np.linalg.svd(matrixA)
        null_mask = (s <= eps) # Find the eigenvalue 0. Return an array with 'False' in each element that does not respect the condition
        null_space = np.compress(null_mask, vh, axis = 0) # Select the row (axis=0) of V^H that correspond to the eigenvalue 0
        return np.squeeze(np.asarray(null_space))

    def get_eigenstates(self, state):
        """ Get the populations of the eigenstates of the system given a flattened general density matrix (state parameter)
        """
        if np.size(state) == self.eigenstates.size:
            return state[self.eigenstates > 0]
        elif np.shape(state)[1] == self.eigenstates.size:
            newMatrix = np.zeros((np.shape(state)[0], self.eigenstates.sum()))
            for idx in range(np.shape(state)[0]):
                newMatrix[idx] = self.get_eigenstates(state[idx])
            return newMatrix
        else:
            print('Not a valid state vector has been passed.')
            return state

    def static_solution(self, operator = 'emission', returnStates = None, **kwargs):
        return self._common_solution(operator = operator, returnStates = returnStates, **kwargs)

    def _time_solution(self, time, startState = None, operator = 'emission', returnStates = False, **kwargs):
        return self._common_solution(time = time, startState = startState, operator = operator, returnStates = returnStates, **kwargs)

    def _common_solution(self, time = None, startState = None, operator = 'emission', returnStates = False, **kwargs):
        if not all(param in kwargs.keys() for param in self.external_params):
            raise ValueError('Impossible to evaluate the Hamiltonian. Some of the required paramaters are missing.')
            return -1
        self._update_changingParams(**kwargs)
        if time is not None and startState is None:
            _, startState = self.static_solution(operator = operator, returnStates = 'full', **kwargs)
        superOp = self.eval_Hamiltonian(**kwargs)
        if time is None:
            newState = self._nullspace(superOp)
        else:
            newState = expm(superOp * time) @ startState
        self.finalState = newState / np.dot(newState, self.eigenstates)
        realState = np.real(self.finalState)
        #TODO: Accept correcly-dimentioned vectors.
        if not operator in self.operators:
            raise ValueError('The operator specified (default: emission) is not defined for this Hamiltonian.')
            return -1
        else:
            emission = np.real(np.dot(self.finalState, self.operators[operator])) # Overall emission [Hz]
        if returnStates == 'full':
            return emission, realState
        elif returnStates == 'eigenstates':
            return emission, self.get_eigenstates(realState)
        else:
            return emission

    def time_solution(self, time, startState = None, operator = 'emission', returnStates = False, **kwargs):
        if not isinstance(time, Iterable):
            time = [time]
        emission = np.zeros(np.size(time))
        return_test = self._time_solution(0., startState, operator = operator, returnStates = returnStates, **kwargs)
        if isinstance(return_test, tuple):
            returning_stateMatrix = True
            finalStates = np.zeros((np.size(time), np.size(return_test[1])))
        else:
            returning_stateMatrix = False
        for idx, t in enumerate(time):
            if returning_stateMatrix:
                emission[idx], finalStates[idx] = self._time_solution(t, startState, operator = operator, returnStates = returnStates, **kwargs)
            else:
                emission[idx] = self._time_solution(t, startState, operator = operator, returnStates = False, **kwargs)

        if returnStates:
            return emission, finalStates
        else:
            return emission

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    testNV = SingleNV(T1 = 1/1e-3)
    t_axis = np.logspace(-6, -2.3, 100)
    _, startState = testNV.static_solution(excRate = 4.432e9, magnField = 0., magnVector = (0,0,1), returnStates = 'full')
    testEmission = testNV.time_solution(time = t_axis, startState = startState, operator = 'bright', excRate = 0., magnField = 0., magnVector = (0,0,1))
    plt.plot(t_axis, testEmission, 'b-')
