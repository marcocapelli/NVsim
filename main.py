# -*- coding: utf-8 -*-

"""
This file uses the libraries to simulate the NV centre emission.

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
import matplotlib.pyplot as plt

from NVsim.setup import Setup
from NVsim.ensemble import EnsembleNV

setupA = Setup(EnsembleNV(NVaxis=(1, 1, 1)), excPower=100e-6)
print(setupA)
x_axis = np.linspace(0.1e-3, 400e-3, 200)
emission = setupA.simulate_emission(excPower=100e-6,
                                    magnField=x_axis, magnVector=(0, 0, 1))
plt.plot(x_axis, emission, 'b-')

t_axis = np.linspace(0.1e-9, 100e-9, 200)
lifetime = setupA.simulate_lifetime(t_axis)
plt.figure()
plt.semilogy(t_axis, lifetime, 'b-')

f_axis = np.linspace(2.4e9, 3.3e9, 200)
odmr = setupA.simulate_ODMR(f_axis, mw_power=1e6,
                            magnField=10e-3, magnVector=(0, 0, 1))
plt.figure()
plt.plot(f_axis, odmr, 'b-')
