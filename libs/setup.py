# -*- coding: utf-8 -*-

"""
This file contains the definition of the Setup class.

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
import scipy.constants as spConst
from collections import namedtuple

from .singleNV import SingleNV
from .ensembleNV import EnsembleNV

#Universal constants definition
Constants = namedtuple('Constants', 'muB ge hbar pi c')
const = Constants(muB = spConst.physical_constants['Bohr magneton'][0],
                  ge = -spConst.physical_constants['electron g factor'][0],
                  pi = spConst.pi,
                  hbar = spConst.hbar,
                  c = spConst.c)

######################################################
# Iterable_dict support class
######################################################
class Iterable_dict(object):

    def __init__(self, **kwargs):
        self.single_values = {}
        self.changing_values = {}
        self.size = 1
        self.current_index = 0

        self.update(**kwargs)

    def update(self, **kwargs): # AND RESET!
        # Update the values saved in the object.
        for key, value in kwargs.items():
            # Beware! It creates error if you are looking at 3 different magnetic angles. Please add a 4th angle if that is the case.
            if key == 'magnVector':
                if np.size(value) == 1 or (np.shape(value)[0] == 3 and np.size(value) == 3):
                    # If it is a single angle OR a single [x,y,z] vector, then it is a single value.
                    self.single_values.update({key: value})
                    if key in self.changing_values.keys():
                        self.changing_values.pop(key)
                else:
                    self.changing_values.update({key: value})
                    self.single_values.update({key: value[0]})

            else:
                if not np.size(value) > 1:
                    self.single_values.update({key: value})
                    if key in self.changing_values.keys():
                        self.changing_values.pop(key)
                else:
                    self.changing_values.update({key: value})
                    self.single_values.update({key: value[0]})

        # Check that all changing parameters have the same length.
        self.size = 1
        for key, value in self.changing_values.items():
            changing_size = np.shape(value)[0]
            if self.size > 1 and not self.size == changing_size:
                raise ValueError('The parameter {} has a different dimention ({}) from the dimension of the experiment ({}).'.format(key, np.size(value), self.size))
            else:
                self.size = changing_size

        return self

    def __str__(self):
        full_print = str(self.single_values) + '\nThe changing parameters are:\n' + str(self.changing_values) + '\n'
        return full_print

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.size:
            self.current_index = 0
            raise StopIteration
        else:
            for key, value in self.changing_values.items():
                self.single_values.update({key: value[self.current_index]}) # It works correcly with multi-dimensional magnVector
            self.current_index += 1
            return self.single_values

    def __getitem__(self, index):
        if isinstance(index, str):
            if index in self.single_values.keys():
                return self.single_values[index]
            elif index in self.changing_values.keys():
                return self.changing_values[index]
            else:
                raise KeyError('Key {} is not in the dictionary'.format(index))
        elif isinstance(index, int):
            if 0 <= index < self.size:
                for key, value in self.changing_values.items():
                    self.single_values.update({key: value[index]}) # It works correcly with multi-dimensional magnVector
                return self.single_values
            else:
                raise KeyError('{} is out of bound.'.format(index))
        else:
            raise KeyError('{} is not a valid key.'.format(index))

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.update(**{index: value})
        else:
            raise KeyError('{} is not a valid key for assignment.'.format(index))

######################################################
# Setup class
######################################################
class Setup(object):

    def __init__(self, NV = None, **kwargs):

        if not isinstance(NV, (SingleNV, EnsembleNV)):
            raise TypeError('A single (or ensemble) NV centre object is required to define a Setup.')
        # Set default parameters
        self.paramDict = {'NV': NV, 'collectionEff': 1.,
                          'excWavelength': 532e-9, 'NA': 0.9, 'excArea': None,
                          'excPower': 1e-3, 'excRate': None,
                          'magnField': 0., 'magnVector': (0,0,1),
                          'freqMW': 2870., 'powerMW': 0.}
        self.update(**kwargs)

    def __str__(self):
        detail_str = ''
        if self.excRate > 0:
            detail_str += '\n--- Laser details ---\n'
            detail_str += '{:.1f} nm\n{:.4e} um^2\n{:.3e} Hz'.format(self.excWavelength*1e9, self.excArea*1e12, self.excRate)
        if self.magnField > 0:
            detail_str += '\n--- Magnetic field details ---\n'
            detail_str += '{:.1f} mT\n[{:.2f}, {:.2f}, {:.2f}]'.format(self.magnField*1e3, *self.magnVector)
        if self.powerMW > 0:
            detail_str += '\n--- Microwave field details ---\n'
            detail_str += '{:.1f} MHz\n{:.2e} arb.u.'.format(self.freqMW*1e-6, self.powerMW)
        if detail_str == '':
            detail_str = 'Setup is off.'
        return detail_str

    def __getattr__(self, name):
        if name in ['NV', 'excWavelength', 'NA', 'excArea', 'excPower', 'excRate', 'magnField', 'magnVector', 'freqMW', 'powerMW', 'collectionEff']:
            return self.paramDict[name]

    def update(self, **kwargs):

        updateArea = True
        updateRate = True
        for name, value in kwargs.items():
            if name == 'excArea':
                if value is not None:
                    self.paramDict['excArea'] = value # Focal spot size [m^2]
                    updateArea = False
            elif name == 'excRate':
                if value is not None:
                    self.paramDict['excRate'] = value # Excitation rate [Hz]
                    updateRate = False
            elif name == 'magnVector':
                if np.size(value) == 3:
                    self.paramDict['magnVector'] = np.array(value / np.linalg.norm(value))
                elif np.size(value) == 1:
                    magnAngle = np.deg2rad(value)
                    self.paramDict['magnVector'] = np.array([np.sin(magnAngle)/np.sqrt(2), np.sin(magnAngle)/np.sqrt(2), np.cos(magnAngle)])
                else:
                    self.paramDict['magnVector'] = np.array([0,0,1])
            else:
                self.paramDict.update({name: value})

        if updateArea:
            spot_diam = self.paramDict['excWavelength'] / (2*self.paramDict['NA'])
            self.paramDict['excArea'] = const.pi * (0.5*spot_diam)**2
        else:
            spot_diam = 2 * np.sqrt(self.paramDict['excArea'] / const.pi)
            self.paramDict['NA'] = self.paramDict['excWavelength'] / (2*spot_diam)

        if updateRate:
            self.paramDict['excRate'] = self.watt2rate(self.paramDict['excPower'])
        else:
            self.paramDict['excPower'] = self.rate2watt(self.paramDict['excRate'])

        return self

    def new(self, **kwargs):

        newDict = dict(self.paramDict)
        newDict.update({'excArea': None, 'excRate': None, 'NV': self.NV})
        for name, value in kwargs.items():
            newDict.update({name: value})

        return Setup(**newDict)

    def watt2rate(self, laserPower):
        NV_sigma = 0.95e-20 # NV absorption cross-section [m^2]
        laserPower = np.clip(laserPower, 0, 1) # Excitation power [W]
        return NV_sigma * (laserPower / self.excArea) * (self.excWavelength / (2*const.pi*const.hbar*const.c)) # Excitation rate [Hz]

    def rate2watt(self, rate):
        NV_sigma = 0.95e-20 # NV absorption cross-section [m^2]
        return (rate * self.excArea) / (self.excWavelength / (2*const.pi*const.hbar*const.c)) / NV_sigma # Excitation rate [Hz]

    def simulate_emission(self, operator = 'emission', returnStates = False, psf_correction = None, **kwargs):
        simulation_dict = Iterable_dict(**kwargs)

        newSetup = self.new()
        emission_vector = np.zeros(simulation_dict.size)
        if newSetup.NV.is_ensemble():
            states_vector = np.zeros((simulation_dict.size, len(newSetup.NV.NV_array), newSetup.NV.eigenstates.sum()))
        else:
            states_vector = np.zeros((simulation_dict.size, newSetup.NV.eigenstates.sum()))
        # Cycle through all values of the changing vector(s)
        for idx, single_dict in enumerate(simulation_dict):
            newSetup.update(**single_dict)
            refExcRate = newSetup.excRate
            if psf_correction is not None:
                returnStates = False
                sigma, n = psf_correction
                radii_x = np.linspace(0., 3*sigma, n)
                gaussian2D = lambda r: np.exp(-((r**2)/(2*sigma**2)))
                middle_r = np.append(np.convolve(radii_x, [0.5, 0.5], 'same'), [radii_x[-1]+0.5*radii_x[1]])
                for r_idx, radius in enumerate(radii_x):
                    newExcRate = refExcRate * gaussian2D(radius)
                    newSetup.update(excRate = newExcRate)
                    temp_emission = newSetup.NV.static_solution(operator = operator, **newSetup.paramDict)
                    area_factor = (middle_r[r_idx+1]**2 - middle_r[r_idx]**2) / (middle_r[-1]**2)
                    emission_vector[idx] += (temp_emission * area_factor)
            else:
                # Update with the Setup with a single value from the array
                emission_vector[idx], states_vector[idx] = newSetup.NV.static_solution(operator = operator, returnStates = 'eigenstates', **newSetup.paramDict)

        emission_vector *= self.collectionEff
        if returnStates:
            return emission_vector, states_vector
        else:
            return emission_vector

    def simulate_lifetime(self, time_array, **kwargs):
        onSetup = self.new(**kwargs)
        _, startState = onSetup.NV.static_solution(returnStates = 'full', **onSetup.paramDict)
        offSetup = self.new(**kwargs).update(excRate = 0.)
        emission = offSetup.NV.time_solution(time_array, startState = startState, operator = 'emission', **offSetup.paramDict)

        emission *= self.collectionEff
        return emission

    def simulate_ODMR(self, frequency_array, mw_power, **kwargs):
        newSetup = self.new(**kwargs)
        if isinstance(self.NV, EnsembleNV) or np.dot(newSetup.magnVector, newSetup.NV.NVaxis) < 0.9999: # Arbitrary misalignment threshold
            print('WARNING! The ODMR simulation is based on the rotating frame approximation.\nThe simultaneous presence of microwave and misalignment magnetic field breaks that approximation.\nThe ODMR result may not be correct.')
        emission = newSetup.simulate_emission(freqMW = frequency_array, powerMW = mw_power, operator = 'emission', returnStates = False)

        return emission

    def simulate_T1(self, time_array, **kwargs):
        onSetup = self.new(**kwargs)
        offSetup = self.new(**kwargs).update(excRate = 0.)
        timetrace_array_change = np.linspace(0, 1e-6, 51)
        timetrace_array_reference = np.linspace(2e-6, 3e-6, 10)
        emission_vector = np.zeros_like(time_array)
        _, common_start_state = self.NV.static_solution(returnStates = 'full', **onSetup.paramDict)
        for idx, time in enumerate(time_array):
            _, state_after_T1 = self.NV.time_solution([0., time], startState = common_start_state, returnStates = 'full', **offSetup.paramDict)
            emission_trace_change = self.NV.time_solution(timetrace_array_change, startState = state_after_T1[1], returnStates = False, **onSetup.paramDict)
            emission_trace_reference = self.NV.time_solution(timetrace_array_reference, startState = state_after_T1[1], returnStates = False, **onSetup.paramDict)
            # The coefficients are the DeltaT of the two timetraces. It is important to consider them for a correct integration under the timetrace curve.
            emission_vector[idx] = (2e-8*emission_trace_change.sum()) / (1e-7*emission_trace_reference.sum())

        return emission_vector


if __name__ == '__main__': #testing

    import matplotlib.pyplot as plt

    setupA = Setup(EnsembleNV(NVaxis = (1,1,1)), excArea = 0.574e-12, excPower = 100e-6)
    print(setupA)
    x_axis = np.linspace(0.1e-3, 400e-3, 200)
    emission = setupA.simulate_emission(excPower = 100e-6, magnField = x_axis, magnVector = (0,0,1))
    plt.plot(x_axis, emission / emission[0], 'b-')

    t_axis = np.linspace(0.1e-9, 100e-9, 200)
    lifetime = setupA.simulate_lifetime(t_axis)
    plt.figure()
    plt.semilogy(t_axis, lifetime / lifetime[0], 'b-')

    f_axis = np.linspace(2.4e9, 3.3e9, 200)
    odmr = setupA.simulate_ODMR(f_axis, mw_power = 1e6, magnField = 10e-3, magnVector = (0,0,1))
    plt.figure()
    plt.plot(f_axis, odmr / np.max(odmr), 'b-')
