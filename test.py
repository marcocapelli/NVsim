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

from libs.setup import Setup
from libs.singleNV import SingleNV

test_NV = SingleNV(NVaxis=(0, 0, 1), hamiltonian='nitrogen')
setupA = Setup(test_NV, excPower=100e-6)
print(setupA)

t_axis = np.linspace(0.1e-9, 200e-9, 1000)
timetrace = setupA.simulate_timetrace(t_axis)
plt.figure()
plt.plot(t_axis, timetrace, 'b-')

x_axis = np.linspace(0.2e-6, 1e-3, 500)
emission = setupA.simulate_emission(excPower=x_axis)
plt.figure()
plt.plot(x_axis, emission, 'b-')

# f_axis = np.linspace(2.4e9, 3.3e9, 200)
# odmr = setupA.simulate_ODMR(f_axis, mw_power=1e6)
# plt.figure()
# plt.plot(f_axis, odmr, 'b-')

plt.show()
