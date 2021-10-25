# =========================================================================================================================================================================
# Toy hybrid mode-locked laser model using a traveling-wave model (v2) and an extension of the nonlinear shrodinger equation
# - In this toy model the passive laser cavity can be represented with a split-step Fourier method
#   ... (to include dispersion, nonlinearity, etc.) or simply with loss/delay if desired
# - The traveling-wave model version 2 includes distributed gain dispersion in contrast to the first version in which a localized spectral filter was used
# - Some examples are included to simulate mode-locked lasers (work in progress)


# Photonics Research Group, Ghent University - imec
# last updated: october 2021, by Stijn Cuyvers
# for an introduction to the hybrid modeling concept: https://doi.org/https://doi.org/10.1038/s41598-021-89508-6
# for questions: stijn.cuyvers@ugent.be

# =========================================================================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
#import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
#import cProfile
import datetime


# -- Physical constants --
c = 299792458  # speed of light[m / s]
h = 6.626070040 * 1e-34  # planck's constant [J*s]
e = 1.60217662 * 1e-19  # electron charge[C]

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"


    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """
    print(x)
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def find_localmaxima(x, mph):
    # x: input array
    # mph: miminal peak height
    # returns local maxima, defined as points larger than mph and larger than its neighboring elements
    return np.where(np.logical_and(x > mph, np.logical_and(x - np.roll(x, 1) > 0, np.abs(x) - np.roll(x, -1) > 0)))[0]


def sech_functiondef(x, P0, x1, x2):
    # hyperbolic sech^2 function
    return P0 * ((1 / np.cosh((x1 - x) / x2)) ** 2)


def getspectrum(dt, lambda0, OUTPUTmonitor, get_envelope_and_phase=False):
    # Function to fourier transform pulse train and get spectrum
    # ----------------------------------------------------------------------------------------------------------
    # Arguments:
    #   dt: discretization step size [s]
    #   lambda0: central wavelength [m]
    #   OUTPUTmonitor: time trace of signal samples
    #   get_envolope_and_phase: option in case the envelope and spectral phase is to be extracted
    # Returns
    #   valid boolean (for reliability purposes, indicates whether the trace was sufficiently long)
    #   lambdas, spectrum
    #   lambdas_envelope,envelope,lambdas_envelope,phases in case get_envelope_and_phase==True

    outputmonitor_secondhalf = OUTPUTmonitor[int(np.round(0.5 * len(OUTPUTmonitor))):int(len(OUTPUTmonitor))]
    minpeakheight = 0.8 * np.max(np.abs(outputmonitor_secondhalf)) # a bit of an arbitrary choice...
    peakpositions = find_localmaxima(np.abs(outputmonitor_secondhalf), minpeakheight)

    n_o_peaks = len(peakpositions)
    if n_o_peaks < 20:
        print('The signal time trace is too short to reliably plot the spectrum.')
        return [False, 0, 0, 0, 0, 0, 0]

    else:
        valid = True
        pos2 = int(np.round(0.5 * (peakpositions[n_o_peaks - 1] + peakpositions[n_o_peaks - 2])))
        pos1 = int(np.round(0.5 * (peakpositions[int(np.round(0.5 * n_o_peaks)) - 1] + peakpositions[int(np.round(0.5 * n_o_peaks)) - 2])))
        trace_selection = outputmonitor_secondhalf[pos1:pos2]

        V = np.linspace(-0.5 * 2 * np.pi / dt, 0.5 * 2 * np.pi / dt, int(len(trace_selection)))
        W = V + 2 * np.pi * c / lambda0
        f = np.multiply(W, (1e-12) / (2 * np.pi))
        spectrum = np.multiply((dt / int(len(V))), np.square(np.abs(np.fft.fftshift(np.fft.fft(trace_selection)))))

        if get_envelope_and_phase == True:
            envelope_n_o_points = 1000
            spectrum_envelope = np.zeros((envelope_n_o_points))
            f_envelope = np.zeros((envelope_n_o_points))
            phases = np.zeros((envelope_n_o_points))
            n_o_samp = int(np.floor(int(len(spectrum)) / envelope_n_o_points))
            for i in range(envelope_n_o_points):
                if i == len(spectrum) - 1:
                    temp_array = 10 * np.log10((1 / (1e-3)) * spectrum[1 + i * n_o_samp:int(len(spectrum))])
                    temp_max = np.max(temp_array)
                    pos_max = int(np.where(temp_array == temp_max)[0])
                else:
                    temp_array = 10 * np.log10((1 / (1e-3)) * spectrum[int(i * n_o_samp):int((i + 1) * n_o_samp - 1)])
                    temp_max = np.max(temp_array)
                    pos_max = int(np.where(temp_array == temp_max)[0])
                spectrum_envelope[i] = temp_max
                f_envelope[i] = f[pos_max + i * n_o_samp]
                phases[i] = np.angle(np.fft.fftshift(np.fft.fft(trace_selection))[pos_max + i * n_o_samp])

            lambdas = np.multiply((1e-3) * c, np.reciprocal(f))
            spectrum = 10 * np.log10((1 / (1e-3)) * spectrum)
            lambdas_envelope = (1e-3) * np.multiply(c, np.reciprocal(f_envelope))
            envelope = spectrum_envelope

            return [valid, lambdas, spectrum, lambdas_envelope, envelope, lambdas_envelope, phases]
        else:
            lambdas = np.multiply((1e-3) * c, np.reciprocal(f))
            spectrum = 10 * np.log10((1 / (1.0e-3)) * spectrum)
            return [valid, lambdas, spectrum, 0, 0, 0, 0]


def getpulse(dt, OUTPUTmonitor, CARRIERmonitor,estimated_rep_rate):
    # extract pulse from output monitor, get pulse width, pulse energy, etc.

    outputmonitor_secondhalf = OUTPUTmonitor[int(np.round(0.5 * len(OUTPUTmonitor))):int(len(OUTPUTmonitor))]
    minpeakheight = 0.8 * np.max(np.abs(outputmonitor_secondhalf))
    peakpositions = detect_peaks(np.abs(outputmonitor_secondhalf), mph=minpeakheight,mpd=0.1*1/(estimated_rep_rate*dt))
    n_o_peaks = len(peakpositions)
    valid_flag = True

    if n_o_peaks > 10:
        # take the second last pulse of the trace
        pos2 = int(np.round(0.5 * (peakpositions[n_o_peaks - 1] + peakpositions[n_o_peaks - 2])))
        pos1 = int(np.round(0.5 * (peakpositions[n_o_peaks - 3] + peakpositions[n_o_peaks - 2])))
        trace_selection = outputmonitor_secondhalf[pos1:pos2]
        trace_carriers = np.zeros((len(CARRIERmonitor), pos2 - pos1))

        positionoffset = int(len(CARRIERmonitor) - len(outputmonitor_secondhalf))
        for j in range(len(CARRIERmonitor)):
            trace_carriers[j][:] = CARRIERmonitor[j][positionoffset + pos1:positionoffset + pos2]

        # define time axis in picoseconds
        T_span = dt * len(trace_selection) * 1e12
        t_axis = np.linspace(-0.5 * T_span, 0.5 * T_span, len(trace_selection))

        pulsewidth=0
        pulseenergy=0
        trace_selection_fit=trace_selection
        # sech fit to pulse
        if len(trace_selection)>50: # arbitrary minimal trace length for reliable fitting
            xdata = np.linspace(-0.5 * T_span, 0.5 * T_span, len(trace_selection))
            ydata = np.square(np.abs(trace_selection))
            guess = np.array([max(ydata), 0, 1 / 1.763])
            [fit, pcov] = curve_fit(sech_functiondef, xdata, ydata, p0=guess)
            trace_selection_fit = sech_functiondef(xdata, fit[0], fit[1], fit[2])
            pulsewidth = fit[2] * 1.763  # FWHM pulse width for sech [ps]
            pulseenergy = pulsewidth * fit[0] / 0.88  # pulse energy for sech [pJ]

        return [valid_flag, t_axis, trace_selection, trace_selection_fit, pulsewidth, pulseenergy, trace_carriers]

    else:
        valid_flag = False
        return [valid_flag, 0, 0, 0, 0, 0, 0]


def carrier_init(x, I, e, V, Nt, A, B, C):
    # carrier equation to initialize carriers at startup given the injection current I, active volume V, transparency carrier density Nt,
    # ... and the recombination parameters A,B,C
    return (I / (e * V * Nt)) - A * x - (B) * (x ** 2) - (C) * (x * x * x)


class TWM:
    # Traveling-wave model, used for modeling active semiconductor sections of the mode-locked laser
    # ----------------------------------------------------------------------------------------------------------
    # Arguments required for initialization of the TWM class:
    # dt: discretization step size [s]
    # Lambda: central operating wavelength [m]
    # Type: to distinghuish between amplfier/saturable absorber/..
    # Parameters: optional vector containing the parameters of the semiconductor section to be defined
    #
    # Notes:
    #   ... the model is currently only designed to incorporate an injection current for the amplifier/gain region
    #   ... for an absorber section simply set the current to zero and use an appropriate gain constant.

    def __init__(self, dt, lambda0, type=1, Parameters=None):
        self.dt = dt  # step time [s]
        self.lambda0 = lambda0  # wavelength [m]

        # -- Physical constants --
        self.c = c
        self.h = h
        self.e = e

        # -- Debug Mode --
        self.DEBUGMODE = False  # can be useful to trouble shoot

        # -- Semiconductor amplifier/absorber parameters --
        if Parameters != None:
            self.Nt = Parameters[0]
            self.RA = Parameters[1]
            self.RB = Parameters[2]
            self.RB = self.RB * self.Nt  # (for normalization)
            self.RC = Parameters[3]
            self.RC = self.RC * (self.Nt ** 2)  # (for normalization)
            self.chi0 = Parameters[4]
            self.beta = Parameters[5]
            self.l = Parameters[6]
            self.I = Parameters[7]
            self.epsilon = Parameters[8]
            self.alpha = Parameters[9]
            self.n_g = Parameters[10]
            self.neff = Parameters[11]
            self.conf = Parameters[12]
            self.Asemi = Parameters[13]
            self.Aqw = Parameters[14]

            # Semiconductor gain dispersion parameters
            # for details on the modeling of gain dispersion: DOI 10.1007/s11082-006-0045-2
            self.gaindisp_height = Parameters[15]  # amplitude of gain dispersion [1/m]
            self.gaindisp_peak = Parameters[16]  # relative center frequency of the Lorentzian gain spectrum
            self.gaindisp_FWHM = Parameters[17]  # FWHM of gain dispersion [m]

        else: # some default values
            # semiconductor optical amplifier
            if type == 1:
                self.Nt = 1e24  # transparency carrier density [m^3]
                self.RA = 1e9  # Nonradiative recombination rate [s^-1]
                self.RB = 7e-17  # Spontaneous recombination rate [m^3 s^-1]
                self.RB = self.RB * self.Nt  # (for normalization)
                self.RC = 1e-41  # Auger recombination rate [m^6 s^-1]
                self.RC = self.RC * (self.Nt ** 2)  # (for normalization)
                self.chi0 = 65e-21  # material gain constant [m^2]
                self.beta = 3e11  # linear internal losses semiconductor section, [s^-1]
                self.l = 450e-6  # length of semiconductor section [m]
                self.I = 29e-3  # injection current [A]
                self.epsilon = 20e-24  # Nonlinear gain compression [m^3]
                self.alpha = -2.5  # linewidth enhancement factor x transparency carrier density
                self.n_g = 3.34  # group index
                self.neff = 3.2  # effective index
                self.conf = 0.075  # confinement factor (dimensionless)
                self.Asemi = 1.54e-12  # Mode area in semiconductor section [m^2]
                self.Aqw = 1.08e-13  # area of the quantum wells (active area)
            # saturable absorber
            else:
                self.Nt = 1e24  # transparency carrier density [m^3]
                self.RA = 1e11  # Nonradiative recombination rate [s^-1]
                self.RB = 0  # Spontaneous recombination rate [m^3 s^-1]
                self.RB = self.RB * self.Nt  # (for normalization)
                self.RC = 0  # Auger recombination rate [m^6 s^-1]
                self.RC = self.RC * (self.Nt ** 2)  # (for normalization)
                self.chi0 = 2000e-21  # material gain constant [m^2]
                self.beta = 3e11  # linear internal losses semiconductor section, [s^-1]
                self.l = 40e-6  # length of semiconductor section [m]
                self.I = 0e-3  # injection current [A]
                self.epsilon = 40e-24  # Nonlinear gain compression [m^3]
                self.alpha = -1.3  # linewidth enhancement factor x transparency carrier density
                self.n_g = 3.54  # group index
                self.neff = 3.24  # effective index
                self.conf = 0.02  # confinement factor (dimensionless)
                self.Asemi = 1.74e-12  # Mode area in semiconductor section [m^2]
                self.Aqw = 1.08e-13  # area of the quantum wells

            # Semiconductor gain dispersion parameters
            # for details on the modeling of gain dispersion: DOI 10.1007/s11082-006-0045-2
            self.gaindisp_height = 50e2  # amplitude of gain dispersion [1/m]
            self.gaindisp_peak = 0  # relative center frequency of the Lorentzian gain spectrum
            self.gaindisp_FWHM = 30e-9  # FWHM of gain dispersion [m]

        # -- Dependent parameters --
        self.omega0 = 2 * np.pi * self.c / self.lambda0  # central frequency
        self.v_g = self.c / self.n_g  # group velocity
        self.dz = self.dt  # normalized spatial stepsize [s]
        self.hv = self.h * self.c / self.lambda0  # photon energy
        self.V = self.Aqw * self.l  # active volume of gain medium [m^3]
        self.gaindisp_width = 0.5 * self.gaindisp_FWHM * 2 * np.pi * self.c / (self.lambda0 ** 2)
        self.const1 = 1j * self.omega0 * self.v_g * self.conf / (2 * self.neff * self.c)
        self.const2 = self.conf / (self.Nt * self.h * self.neff * self.c / (np.pi * 2))
        self.n_o_spacesteps = int(np.round((self.l / self.v_g) / self.dz))

        # -- Pre-allocation --
        # Af, Ab designate the forward and backward propagating fields (complex values)
        # Pf, Pb designate the polarization terms to model the Lorentzian gain dispersion (complex values)
        # N designates the normalized carrier density (i.e. N/Nt)  (real valued)
        # temp_factor and preCalc boost computation speed
        self.Af = np.zeros((2, self.n_o_spacesteps), dtype=np.complex128)
        self.Ab = np.zeros((2, self.n_o_spacesteps), dtype=np.complex128)
        self.Pf = np.zeros((2, self.n_o_spacesteps), dtype=np.complex128)
        self.Pb = np.zeros((2, self.n_o_spacesteps), dtype=np.complex128)
        self.N = np.zeros((2, self.n_o_spacesteps), dtype=np.float64)

        # current vector
        self.vector_current = [self.I for j in range(self.n_o_spacesteps)]
        self.vector_activevolume = [self.V for j in range(self.n_o_spacesteps)]
        self.vector_pump = np.multiply(self.vector_current, (1. / (np.multiply(self.Nt * self.e, self.vector_activevolume))))
        self.vector_epsilons = [self.I for j in range(self.n_o_spacesteps)]

    def initialize(self, sigma=0.1e-6):

        # -- initialize field envelope samples with noisy complex values --
        self.Af[0][:] = ((1 / np.sqrt(self.Asemi)) * sigma * (np.random.rand(1, self.n_o_spacesteps) + 1j * np.random.rand(1, self.n_o_spacesteps)))
        self.Ab[0][:] = ((1 / np.sqrt(self.Asemi)) * sigma * (np.random.rand(1, self.n_o_spacesteps) + 1j * np.random.rand(1, self.n_o_spacesteps)))

        self.Pf[0][:] = ((1 / np.sqrt(self.Asemi)) * sigma * (np.random.rand(1, self.n_o_spacesteps) + 1j * np.random.rand(1, self.n_o_spacesteps)))
        self.Pb[0][:] = ((1 / np.sqrt(self.Asemi)) * sigma * (np.random.rand(1, self.n_o_spacesteps) + 1j * np.random.rand(1, self.n_o_spacesteps)))

        # solve for equilibrium carrier density (in absence of field), i.e. solve carrier equation for field=0
        root = fsolve(carrier_init, 0.5, args=(self.I, self.e, self.V, self.Nt, self.RA, self.RB, self.RC))
        if self.I!=0:
            N_eq = root[0]
        else:
            N_eq = 0.02
        sigma_N = 1e-5
        self.N[0][:] = np.real([N_eq for j in range(self.n_o_spacesteps)] + sigma_N * (np.random.rand(1, self.n_o_spacesteps)))

        print('N equil')
        print(N_eq)

    def step(self, i, leftinput, rightinput):
        # Traveling-wave model - iterative step
        # update forward- and backward propagating fields in TWM section
        # to save memory, not all samples are saved
        # only samples at time t (index iteration i) and previous time (index i-1) are saved
        # i (current iteration number)   -> mod(i,2)
        # i-1 (previous iteration)       -> 1-mod(i,2)
        # INPUTS:
        # ... leftinput (to feed the forward propagating field from the left),
        # ... rightinput (to feed the backward propagating field from the right)

        # the partial differential equations are currently implemented with a (naieve and simple) Euler scheme, improvement is possible here!

        # update polarization terms
        self.Pf[np.mod(i, 2)][:] = self.Pf[1 - np.mod(i, 2)][:] + np.multiply(self.dt, np.multiply(self.gaindisp_width, self.Af[1 - np.mod(i, 2)][:] - self.Pf[1 - np.mod(i, 2)][:]) + np.multiply(1j * self.gaindisp_peak, self.Pf[1 - np.mod(i, 2)][:]))
        self.Pb[np.mod(i, 2)][:] = self.Pb[1 - np.mod(i, 2)][:] + np.multiply(self.dt, np.multiply(self.gaindisp_width, self.Ab[1 - np.mod(i, 2)][:] - self.Pb[1 - np.mod(i, 2)][:]) + np.multiply(1j * self.gaindisp_peak, self.Pb[1 - np.mod(i, 2)][:]))

        if self.DEBUGMODE == True:
            if np.any(np.isnan(self.Pf[np.mod(i, 2)][:])) or np.any(np.isnan(self.Pb[np.mod(i, 2)][:])):
                print('ERROR: Pf / Pb contains invalid values.')
                print('iteration', str(i))
                exit()

        # update fields Af,Ab
        ChiFactor = self.const1 * (-self.neff * self.c * self.chi0 * self.Nt / self.omega0) * (1j / (1 + (self.epsilon * self.conf / (self.v_g * self.hv)) * (np.square(np.abs(self.Af[1 - np.mod(i, 2)][:])) + np.square(np.abs(self.Ab[1 - np.mod(i, 2)][:])))) + self.alpha) * (self.N[1 - np.mod(i, 2)][:] - 1)
        self.Af[np.mod(i, 2)][1:self.n_o_spacesteps] = self.Af[1 - np.mod(i, 2)][0:self.n_o_spacesteps - 1] * (1 + self.dz * (ChiFactor[0:self.n_o_spacesteps - 1] - (0.5 * self.beta + 0.5 * self.v_g * self.gaindisp_height))) + self.dz * 0.5 * self.v_g * self.gaindisp_height * (self.Pf[np.mod(i, 2)][0:self.n_o_spacesteps - 1])
        self.Ab[np.mod(i, 2)][0:self.n_o_spacesteps - 1] = self.Ab[1 - np.mod(i, 2)][1:self.n_o_spacesteps] * (1 + self.dz * (ChiFactor[1:self.n_o_spacesteps] - (0.5 * self.beta + 0.5 * self.v_g * self.gaindisp_height))) + self.dz * 0.5 * self.v_g * self.gaindisp_height * (self.Pb[np.mod(i, 2)][1:self.n_o_spacesteps])

        if self.DEBUGMODE == True:
            if np.any(np.isnan(ChiFactor)) or np.any(np.isnan(self.Af[np.mod(i, 2)][:])) or np.any(np.isnan(self.Ab[np.mod(i, 2)][:])):
                print('ERROR: Af / Ab contains invalid values.')
                print('iteration', str(i))
                exit()

        # OPTIONAL (future update): add ASE to fields (see for example: doi:10.1016/j.optcom.2008.06.039, DOI 10.1007/s11082-006-0045-2)
        # simply add complex gaussian independent variables to the forward/backward field with a variance either based on fundamental
        # values or extracted from experimental data
        enable_ASE=True
        if enable_ASE and self.I>0:
            Comega=4.4*1e-11 # frequency shift coefficient [m^3 s^-1] (source: doi:10.1016/j.optcom.2008.06.039)
            kT = 4.11*1e-21 # botlzman constant x temperature [J]
            nsp= (1-np.exp(-h*Comega*self.Nt*(self.N[1 - np.mod(i, 2)][:]-1)/(kT*2*np.pi)))
            g=self.chi0*self.Nt*(self.N[1 - np.mod(i, 2)][:]-1)
            NoiseVar=self.hv*self.conf*self.v_g*nsp*g
            # # beta_sp = 1e-4;
            # # Rfactor = 1e14;
            # # % NoiseVar = beta_sp * Rfactor * dt;
            # NoiseVar = 1 # (1e-4) * self.dt / ((self.Asemi))
            Fspf = (np.sqrt(NoiseVar) / np.sqrt(2)) * (np.random.rand(1, self.n_o_spacesteps) + 1j * np.random.rand(1, self.n_o_spacesteps))
            Fspb = (np.sqrt(NoiseVar) / np.sqrt(2)) * (np.random.rand(1, self.n_o_spacesteps) + 1j * np.random.rand(1, self.n_o_spacesteps))
            self.Af[np.mod(i, 2)][:] = self.Af[np.mod(i, 2)][:] + Fspf
            self.Ab[np.mod(i, 2)][:] = self.Ab[np.mod(i, 2)][:] + Fspb


        # boundary conditions
        self.Af[np.mod(i, 2)][0] = leftinput
        self.Ab[np.mod(i, 2)][self.n_o_spacesteps - 1] = rightinput

        # update carrier density (carrier diffusion has been omitted here, earlier simulations indicate this has a negligble effect)
        ChiFactor = self.const2 * (-self.neff * self.c * self.chi0 * self.Nt / self.omega0) * (1j / (1 + (self.epsilon * self.conf / (self.v_g * self.hv)) * (np.square(np.abs(self.Af[np.mod(i, 2)][:])) + np.square(np.abs(self.Ab[np.mod(i, 2)][:])))) + self.alpha) * (self.N[1 - np.mod(i, 2)][:] - 1)
        extraterm = -(1 / (self.Nt * self.hv)) * (np.real(np.conj(self.Ab[np.mod(i, 2)][:]) * self.gaindisp_height * (self.Ab[np.mod(i, 2)][:] - self.Pb[np.mod(i, 2)][:])) + np.real(np.conj(self.Af[np.mod(i, 2)][:]) * self.gaindisp_height * (self.Af[np.mod(i, 2)][:] - self.Pf[np.mod(i, 2)][:])))
        self.N[np.mod(i, 2)][:] = self.N[1 - np.mod(i, 2)][:] + self.dt * (
                    extraterm + self.vector_pump - (self.RA * self.N[1 - np.mod(i, 2)][:] + self.RB * np.square(self.N[1 - np.mod(i, 2)][:]) + self.RC * self.N[1 - np.mod(i, 2)][:] * np.square(self.N[1 - np.mod(i, 2)][:])) + (np.square(np.abs(self.Af[np.mod(i, 2)][:])) + np.square(np.abs(self.Ab[np.mod(i, 2)][:]))) * np.imag(ChiFactor))  # + carrier diffusion

        if self.DEBUGMODE == True:
            if np.any(np.isnan(ChiFactor)) or np.any(np.isnan(self.N[np.mod(i, 2)][:])):
                print('ERROR: N contains invalid values.')
                print('iteration', str(i))
                exit()

        # return leftoutput, rightoutput of the TWM
        return [self.Ab[1 - np.mod(i, 2)][0], self.Af[1 - np.mod(i, 2)][self.n_o_spacesteps - 1]]

class Passive:
    # Passive waveguide model, used for modeling passive sections of the mode-locked laser
    # Can model the passive waveguide with delay+loss (faster, less accurate)
    # Alternatively, the passive waveguide can be modeled through a split-step Fourier method to incorporate
    # ... waveguide dispersion, nonlinearity, etc.
    # ----------------------------------------------------------------------------------------------------------
    # Arguments required for initialization of the TWM class:
    # dt: discretization step size [s]
    # lambda: central wavelength of operation [m]
    # lp: length of the passive waveguide [m]
    # cum_delay_other_components: the reservoir saves the output samples of the TWM for split-step Fourier propation
    #                 ... the size of the reservoir should resemble the entire mode-locked laser roundtrip time (to enable the split-step Fourier method to be used consistently)
    # 				  ... (note this is in contrast to the first hybrid model demonstration of https://doi.org/https://doi.org/10.1038/s41598-021-89508-6),
    #                 ... therefore, this class should know the delay of the remaining mode-locked laser building blocks [in units of seconds]
    #                 ... this parameter is only critical when SSF is used, otherwise it is not used 
    # SSF (boolean): enable/disable split-step Fourier propagation for simulating the passive waveguide

    def __init__(self, dt, lambda0, lp, cum_delay_other_components, SSF=False, Parameters=None):
        self.dt = dt  # step time [s]
        self.lambda0 = lambda0  # wavelength [m]
        self.lp = lp  # length of passive waveguide [m]
        self.SSF = SSF # enable/disable split-step Fourier propagation (default is disable)

        # -- Physical constants --
        self.c = c
        self.h = h
        self.e = e

        # -- Waveguide parameters --
        if Parameters==None: # default parameters
            self.n_g = 3.85  # group index
            self.neff = 3  # effective mode index in SOA/SA
            self.loss_passive = 0.7 * 1e2  # time-independent loss [dB/m]
            self.beta2 = 1.3e-24  # [s^2/m], positive means normal dispersion, negative anomalous
            self.beta3 = 0.0042e-36  # [s^3/m], third  order dispersion
            self.n2 = 5 * 1e-18  # nonlinear coefficient n2 [m^2/W] (according to PhD Bart:  n2=6e-18 m^2/W)
            self.Aeff = 0.29 * 1e-12  # effective mode area [m^2]

            # parameters for FCA, TPA
            self.btpa = 6e-12 # two-photon absorption parameter [m/W]
            self.kc = 1.35e-27 # free carrier dispersion [m^3]
            self.sigma = 1.45e-21 # free carrier absorption [m^2]
            self.tau = 1e-9 # free carrier lifetime [s]

        else:
            self.n_g = Parameters[0]  # group index
            self.neff = Parameters[1]  # effective mode index in SOA/SA
            self.loss_passive = Parameters[2]  # time-independent loss [dB/m]
            self.beta2 = Parameters[3]  # [s^2/m], positive means normal dispersion, negative anomalous
            self.beta3 = Parameters[4]  # [s^3/m], third  order dispersion
            self.n2 = Parameters[5]  # nonlinear coefficient n2 [m^2/W] (according to PhD Bart:  n2=6e-18 m^2/W)
            self.Aeff = Parameters[6]  # effective mode area [m^2]

            # parameters for FCA, TPA
            self.btpa = Parameters[7]  # two-photon absorption parameter [m/W]
            self.kc = Parameters[8]  # free carrier dispersion [m^3]
            self.sigma = Parameters[9]  # free carrier absorption [m^2]
            self.tau = Parameters[10]  # free carrier lifetime [s]

        self.s = 1  # slow light factor (set to zero for now, but can be relevant, see for example: doi:10.1038/lsa.2017.8)
        self.Nc_avg = 0  # averaged free carrier density

        # -- Dependent parameters --
        self.omega0 = 2 * np.pi * self.c / self.lambda0
        self.v_g = self.c / self.n_g # group velocity [m/s]
        self.dz = self.dt
        self.hv = self.h * self.c / self.lambda0
        self.D=(self.s**2)*2*np.pi*self.btpa/(2*h*self.omega0*(self.Aeff**2))

        # dispersion vector (incorporates second-order, third-order, .. dispersion)
        self.betas_passive = [self.beta2, self.beta3]

        # Reservoir/queue are arrays holding the field samples that go in/out the passive waveguide
        # Note: the reservoir/queue arrays carry the signal samples in units of Watt [W]

        self.coldcavityRTT = ((2.0 * self.lp / self.v_g)+cum_delay_other_components) # cold cavity roundtrip time [s]
        # margin to allow exact matching of the pulse roundtrip time:
        # ... the actual pulse repetition rate can deviate slightly from the cold cavity roundtrip time due to gain/absorption, third-order dispersion
        # ... to enable exact matching of the pulse roundtrip time (necessary for SSF algorithm), take margin on the cold cavity roundtrip time
        # ... here taken to be 1% of the cold cavity roundtrip delay
        self.Reservoir_capacity_margin = int(np.round(self.coldcavityRTT/dt) / 100)
        self.Reservoir_capacity = int(int(np.round(self.coldcavityRTT/self.dt)) + self.Reservoir_capacity_margin)
        self.Reservoir = np.zeros((2, self.Reservoir_capacity), dtype=np.complex128)
        self.Queue_capacity = int(np.round(self.lp/(self.v_g*self.dt)))
        self.Queue = np.zeros((2, self.Queue_capacity), dtype=np.complex128)
        self.Reservoir_counter = self.Reservoir_capacity-self.Queue_capacity
        self.Queue_counter = 0

    def initialize(self, sigma=0.1e-6):
        # -- initialize queue samples with noisy complex values --
        self.Queue[:][:] = (sigma * (np.random.rand(2, int(self.Queue_capacity)) + 1j * np.random.rand(2, int(self.Queue_capacity))))
        self.Reservoir[:][:] = (sigma * (np.random.rand(2, int(self.Reservoir_capacity)) + 1j * np.random.rand(2, int(self.Reservoir_capacity))))

    def step(self, i, leftinput, rightinput):
        # -- Model the passive waveguide simply with a split-step Fourier propagator --

        self.Reservoir_counter += 1
        self.Queue_counter += 1
        self.Reservoir[0][self.Reservoir_counter] = leftinput
        self.Reservoir[1][self.Reservoir_counter] = rightinput

        if self.SSF == True:
            if self.Reservoir_counter == self.Reservoir_capacity - 1:
                # Window for split-step Fourier propagation should match pulse roundtrip time to satisfy periodicity condition of the Fourier transform
                # To do this, a search space is used with a span of 2 x Reservoir_capacity_margin around the cold cavity roundtrip time
                # This search space is needed as the pulse can slightly advance/lag compared to the cold cavity roundtrip time
                # The conditions to find the window size that matches the pulse roundtrip time are:
                #     - the amplitude should match at both sides of the window (periodicity of amplitude)
                #     - the derivative of the magnitude is checked to eliminate ambiguity (e.g. to differentiate leading/trailing edge of pulses)
                #     Note: periodicity of phase is tricky as a non-periodic phase is imposed by the traveling-wave part of the model
                #     ... if one would ignore the non-periodic phase, significant distortion is introduced by the split-step Fourier method (due to absence of phase periodicity)
                #     ... to solve this, a small phase correction vector is introduced (i.e. a linear increasing/decreasing phase vector) to enforce periodicity of the phase.
                #     ... After split-step Fourier propagation, the small phase correction vector is (by approximation) mitigated by multiplying with the conjugate of the phase correction vector


                deriv_mag =  np.gradient(np.abs(self.Reservoir[0][:])) # gradient vector of the magnitude of the Reservoir samples

                Res_start_pos = self.Reservoir_capacity_margin
                Res_start_pos_final = Res_start_pos

                Res_start_mag = np.abs(self.Reservoir[0][Res_start_pos])
                Res_start_mag_target = np.abs(self.Reservoir[0][self.Reservoir_capacity-1])

                devi_mag = np.abs(Res_start_mag_target - Res_start_mag)

                deriv_mag_target = deriv_mag[self.Reservoir_capacity-1]
                deriv_mag_start = deriv_mag[Res_start_pos]

                for jj in range(1,self.Reservoir_capacity_margin):
                    if np.abs(np.abs(self.Reservoir[0][Res_start_pos-jj])-Res_start_mag_target) < devi_mag:
                        if np.sign(deriv_mag[Res_start_pos-jj]) == np.sign(deriv_mag_target):
                            Res_start_pos_final =  Res_start_pos-jj
                            Res_start_mag = np.abs(self.Reservoir[0][Res_start_pos-jj])
                            devi_mag = np.abs(np.abs(self.Reservoir[0][Res_start_pos-jj])-Res_start_mag_target)
                    if np.abs(np.abs(self.Reservoir[0][Res_start_pos+jj])-Res_start_mag_target) < devi_mag:
                        if np.sign(deriv_mag[Res_start_pos+jj]) == np.sign(deriv_mag_target):
                            Res_start_pos_final =  Res_start_pos+jj
                            Res_start_mag = np.abs(self.Reservoir[0][Res_start_pos+jj])
                            devi_mag = np.abs(np.abs(self.Reservoir[0][Res_start_pos+jj])-Res_start_mag_target)

                Res_start_pos_final = Res_start_pos_final + 1
                Res_start_pos_final_1 = Res_start_pos_final

                deriv_mag = np.gradient(np.abs(self.Reservoir[1][:]))

                Res_start_pos = self.Reservoir_capacity_margin
                Res_start_pos_final = Res_start_pos

                Res_start_mag = np.abs(self.Reservoir[1][Res_start_pos])
                Res_start_mag_target = np.abs(self.Reservoir[1][self.Reservoir_capacity - 1])

                devi_mag = np.abs(Res_start_mag_target - Res_start_mag)

                deriv_mag_target = deriv_mag[self.Reservoir_capacity - 1]
                deriv_mag_start = deriv_mag[Res_start_pos]

                for jj in range(1, self.Reservoir_capacity_margin):
                    if np.abs(np.abs(self.Reservoir[1][Res_start_pos - jj]) - Res_start_mag_target) < devi_mag:
                        if np.sign(deriv_mag[Res_start_pos - jj]) == np.sign(deriv_mag_target):
                            Res_start_pos_final = Res_start_pos - jj
                            Res_start_mag = np.abs(self.Reservoir[1][Res_start_pos - jj])
                            devi_mag = np.abs(np.abs(self.Reservoir[1][Res_start_pos - jj]) - Res_start_mag_target)
                    if np.abs(np.abs(self.Reservoir[1][Res_start_pos + jj]) - Res_start_mag_target) < devi_mag:
                        if np.sign(deriv_mag[Res_start_pos + jj]) == np.sign(deriv_mag_target):
                            Res_start_pos_final = Res_start_pos + jj
                            Res_start_mag = np.abs(self.Reservoir[1][Res_start_pos + jj])
                            devi_mag = np.abs(np.abs(self.Reservoir[1][Res_start_pos + jj]) - Res_start_mag_target)

                Res_start_pos_final = Res_start_pos_final + 1
                Res_start_pos_final_2 = Res_start_pos_final

                Tspan1 = (self.Reservoir_capacity - Res_start_pos_final_1) * self.dt  # time span of window to be propagated with split-step Fourier
                Tspan2 = (self.Reservoir_capacity - Res_start_pos_final_2) * self.dt

                phi1 = np.angle(self.Reservoir[0][Res_start_pos_final_1])
                phi2 = np.angle(self.Reservoir[0][self.Reservoir_capacity-1])
                phasediff = phi2 - phi1
                phase_corr_array1 = np.exp(1j * np.linspace(0.5 * phasediff, -0.5 * phasediff, self.Reservoir_capacity - Res_start_pos_final_1))

                phi1 = np.angle(self.Reservoir[1][Res_start_pos_final_2])
                phi2 = np.angle(self.Reservoir[1][self.Reservoir_capacity-1])
                phasediff = phi2 - phi1
                phase_corr_array2 = np.exp(1j * np.linspace(0.5 * phasediff, -0.5 * phasediff, self.Reservoir_capacity - Res_start_pos_final_2))

                Propagated_field_1 = np.conj(phase_corr_array1) * self.Passive_Waveguide_Propagator(phase_corr_array1 * self.Reservoir[0][Res_start_pos_final_1:self.Reservoir_capacity], Tspan1)
                Propagated_field_2 = np.conj(phase_corr_array2) * self.Passive_Waveguide_Propagator(phase_corr_array2 * self.Reservoir[1][Res_start_pos_final_2:self.Reservoir_capacity], Tspan2)

                self.Queue[0][:] = Propagated_field_1[len(Propagated_field_1) - self.Queue_capacity: len(Propagated_field_1)]
                self.Queue[1][:] = Propagated_field_2[len(Propagated_field_2) - self.Queue_capacity: len(Propagated_field_2)]
                self.Reservoir[0][0:self.Reservoir_capacity - self.Queue_capacity] = self.Reservoir[0][self.Queue_capacity:self.Reservoir_capacity]
                self.Reservoir[1][0:self.Reservoir_capacity - self.Queue_capacity] = self.Reservoir[1][self.Queue_capacity:self.Reservoir_capacity]
                self.Reservoir_counter = self.Reservoir_capacity - self.Queue_capacity -1
                self.Queue_counter = 0
                self.Reservoir[0][self.Reservoir_counter+1:self.Reservoir_capacity] = np.zeros((1, self.Queue_capacity))
                self.Reservoir[1][self.Reservoir_counter+1:self.Reservoir_capacity] = np.zeros((1, self.Queue_capacity))

        else:
        	# -- no split-step Fourier used -- (just incorporate loss)
            if self.Reservoir_counter == self.Reservoir_capacity - 1:
                self.Queue[0][:] = np.multiply(np.exp(-self.lp * self.loss_passive * (1 / 4.343) * 0.5), self.Reservoir[0][self.Reservoir_capacity-self.Queue_capacity:self.Reservoir_capacity])
                self.Queue[1][:] = np.multiply(np.exp(-self.lp * self.loss_passive * (1 / 4.343) * 0.5), self.Reservoir[1][self.Reservoir_capacity-self.Queue_capacity:self.Reservoir_capacity])
                self.Reservoir_counter = self.Reservoir_capacity - self.Queue_capacity - 1
                self.Queue_counter = 0

        # return leftoutput, rightoutput
        return [self.Queue[1][self.Queue_counter], self.Queue[0][self.Queue_counter]]

    def Passive_Waveguide_Propagator_OLD(self, signal_in, t_span):
    	# ----
    	# *** CAN BE IGNORED, THIS IS AN OLD DRAFT OF THE FUNCTION ***
    	# ---

        # -- Split-Step Fourier propagator --
        # ARGUMENTS:
        #   - signal_in: field envelope trace for split-step Fourier propagation [units of sqrt(W)]
        #   - t_span: time span of signal_in [s]
        #   - (  L: length of the passive waveguide [m] )
        #   - (  lambda0: central wavelength [m]  )
        #   - (  betas_passive: dispersion vector, i.e. [beta2 beta3 ..]  )
        #   - (  loss: loss of the passive waveguide [dB/m]  )
        #   - (  n2: nonlinear Kerr nonlinearity [m^2/W]  )
        #   - (  Aeff: effective mode area in the passive waveguide [m^2]  )
        # OUTPUT: propagated signal trace (signal_out)
        # Note: For now, Raman effect and FCA/TPA are ignored

        # -- parameters --
        stepsize = 200e-6  # spatial stepsize for split-step Fourier propagation [m]
        npas = int(np.round(self.lp / stepsize))
        k0 = 2 * np.pi / self.lambda0
        gamma = k0 * self.n2 / self.Aeff # nonlinear parameter
        alpha = np.log(10 ** (self.loss_passive / 10))
        fR = 0  # NOT USED NOW Raman contribution, see 'Nonlinear optical phenomena in silicon waveguides: modeling and applications'

        nTime = len(signal_in)
        T = np.linspace(-t_span * 0.5, t_span * 0.5, nTime)  # time grid
        dT = t_span / nTime  # time grid spacing [s]
        V = 2 * np.pi * np.linspace(-nTime * 0.5 / t_span, (nTime * 0.5 - 1) / t_span, nTime)  # angular frequency grid around omega0 [rad/s]

        # Define propagation constant
        B = 0
        for i in range(1, len(self.betas_passive)):
            B = B + (self.betas_passive[i - 1] / np.math.factorial(i + 1)) * (V ** (i + 1))

        Dispersion_operator = np.fft.fftshift(1j * B - 0.5 * alpha)

        # step-wise propagation
        A = signal_in
        for i in range(1, npas):
            # First dispersion + losses...
            A = np.fft.ifft(np.exp(stepsize * Dispersion_operator * 0.5) * np.fft.fft(A))

            # ... then nonlinearity: Kerr effect operator + Raman contribution (not implemented for now)
            K = 1j * gamma * ((1 - fR) * (np.abs(A) ** 2))  # + fR * nTime * dT * np.fft(np.ifft(np.fftshift(hR))*np.ifft(np.abs(A)**2)))
            A = np.exp(stepsize * K) * A

            # ... and again dispersion + losses.
            A = np.fft.ifft(np.exp(stepsize * Dispersion_operator * 0.5) * np.fft.fft(A))

        signal_out = A

        return signal_out

    def Passive_Waveguide_Propagator(self, signal_in, t_span):
        # -- Split-Step Fourier propagator --
        # ARGUMENTS:
        #   - signal_in: field envelope trace for split-step Fourier propagation [units of sqrt(W)]
        #   - t_span: time span of signal_in [s]
        # OUTPUT: propagated signal trace (signal_out)
        # Note: For now, Raman effect is ignored (typically has a minor effect anyway)
        enable_FCA=1

        # For details on how the split-step Fourier method works: see book Nonliner fiber optics, Agrawal

        # -- parameters --
        stepsize = 200e-6  # spatial stepsize for split-step Fourier propagation [m], little bit arbitrary choice, often ~100µm is used in literature
        npas = int(np.round(self.lp / stepsize))
        k0 = 2 * np.pi / self.lambda0
        gamma = k0 * self.n2 / self.Aeff + 1j*(self.s**2)*self.btpa/(2*self.Aeff)# nonlinear parameter
        alpha = np.log(10 ** (self.loss_passive / 10))
        fR = 0  # NOT USED NOW Raman contribution, see 'Nonlinear optical phenomena in silicon waveguides: modeling and applications'

        nTime = len(signal_in)
        T = np.linspace(-t_span * 0.5, t_span * 0.5, nTime)  # time grid
        dT = t_span / nTime  # time grid spacing [s]
        V = 2 * np.pi * np.linspace(-nTime * 0.5 / t_span, (nTime * 0.5 - 1) / t_span, nTime)  # angular frequency grid around omega0 [rad/s]

        # calculate average free carrier density (this is a coarse approximation of the actual loss induced by free carrier absorption, improvement of accuracy possible here)
        T_span = self.dt*(len(signal_in)-1)
        E4_avg = np.sum(np.power(np.abs(signal_in),4))/len(signal_in)
        self.Nc_avg = self.Nc_avg + T_span * (self.D*E4_avg - self.Nc_avg/self.tau)

        # Define propagation constant
        B = 0
        for i in range(1, len(self.betas_passive)):
            B = B + (self.betas_passive[i - 1] / np.math.factorial(i + 1)) * (V ** (i + 1))
        if enable_FCA==1:
            Dispersion_operator = np.fft.fftshift(1j * B - 0.5 * alpha - 0.5*self.s*self.sigma*self.Nc_avg - 1j*self.s*self.kc*(2*np.pi/self.lambda0)*self.Nc_avg)
        else:
            Dispersion_operator = np.fft.fftshift(1j * B - 0.5 * alpha)

        # step-wise propagation
        A = signal_in
        for i in range(1, npas):
            # First dispersion + losses...
            A = np.fft.ifft(np.exp(stepsize * Dispersion_operator * 0.5) * np.fft.fft(A))

            # ... then nonlinearity: Kerr effect operator + Raman contribution (not implemented for now)
            K = 1j * gamma * ((1 - fR) * (np.abs(A) ** 2))  # + fR * nTime * dT * np.fft(np.ifft(np.fftshift(hR))*np.ifft(np.abs(A)**2)))
            A = np.exp(stepsize * K) * A

            # ... and again dispersion + losses.
            A = np.fft.ifft(np.exp(stepsize * Dispersion_operator * 0.5) * np.fft.fft(A))

        signal_out = A

        return signal_out

class Visualize:
    # -- Visualization class, plots output data --
    def __init__(self, dt, lambda0, n_o_timesteps,estimated_rep_rate, filename):
        self.Fontsize = 14
        self.FontsizeLegend = 12
        self.dt = dt
        self.lambda0 = lambda0
        self.n_o_timesteps = n_o_timesteps
        self.estimated_rep_rate = estimated_rep_rate
        self.filename=filename

    def Plot(self, WAVEmonitors, CARRIERmonitor):
        # -----------------------------------------------------
        # Create 1 plot with 4 subplots
        # -----------------------------------------------------
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(16, 9)

        # OUTPUT PULSE TRAIN
        OUTPUTmonitor=WAVEmonitors[0][:]
        time_axis = np.linspace(0, self.n_o_timesteps * self.dt * 1e9, len(OUTPUTmonitor))
        ax1.plot(time_axis, np.square(np.abs(OUTPUTmonitor)))
        #ax1.set_xlim([0, 80])
        ax1.set_ylim([0, np.max(np.square(np.abs(OUTPUTmonitor)))])

        # INDIVIDUAL PULSE, SECH FIT
        #[valid, t_axis, trace_selection, trace_selection_fit, pulsewidth, pulseenergy, trace_carriers] = getpulse(self.dt, OUTPUTmonitor, CARRIERmonitor,self.estimated_rep_rate)
        ymax=0
        for iii in range(len(WAVEmonitors)):
            [valid, t_axis, trace_selection, trace_selection_fit, pulsewidth, pulseenergy, trace_carriers] = getpulse(self.dt, WAVEmonitors[iii][:], CARRIERmonitor, self.estimated_rep_rate)

            if valid == True:
                legendtext='Monitor'+str(iii)
                ax3.plot(t_axis, np.square(np.abs(trace_selection)), label=legendtext)
                legendtext = 'Fit (' + str(round(pulsewidth, 1)) + ' ps, ' + str(round(pulseenergy, 1)) + ' pJ)'
                ax3.plot(t_axis, trace_selection_fit, 'r--', label=legendtext)
                ymax = np.max((ymax,np.max(np.square(np.abs(trace_selection)))))

        if valid == True:
            ax3.legend(loc="upper right", fontsize=self.FontsizeLegend)
            ax3.set_xlim([-10, 10])
            ax3.set_ylim([0, ymax])
            for ii in range(len(CARRIERmonitor)):
                ax4.plot(time_axis, CARRIERmonitor[ii][:], label='Monitor ' + str(ii))
            ax4.legend(loc="upper right", fontsize=self.FontsizeLegend)
            ax4.set_ylim([0, np.max((np.abs(trace_carriers)))])

        ''' 
        if valid == True:
            ax3.plot(t_axis, np.square(np.abs(trace_selection)), label='Sim')
            legendtext = 'Fit (' + str(round(pulsewidth, 1)) + ' ps, ' + str(round(pulseenergy, 1)) + ' pJ)'
            ax3.plot(t_axis, trace_selection_fit, 'r--', label=legendtext)
            ax3.legend(loc="upper right", fontsize=self.FontsizeLegend)
            ax3.set_xlim([-10, 10])
            ax3.set_ylim([0, np.max(np.square(np.abs(trace_selection)))])

            
            # NORM. CARRIER DENSITY
            #for ii in range(len(trace_carriers)):
            #    ax4.plot(t_axis, trace_carriers[ii][:], label='Pos ' + str(ii))
            ## ax4.plot(t_axis, trace_carriers[1][:],'--',label='Pos 2')
            ## ax4.plot(t_axis, trace_carriers[2][:],'-.',label='Pos 3')
            #ax4.legend(loc="upper right", fontsize=self.FontsizeLegend)
            #ax4.set_xlim([-500, 500])
            #ax4.set_ylim([0, np.max((np.abs(trace_carriers)))])
            

            for ii in range(len(CARRIERmonitor)):
                ax4.plot(time_axis, CARRIERmonitor[ii][:], label='Monitor ' + str(ii))
            ax4.legend(loc="upper right", fontsize=self.FontsizeLegend)
            ax4.set_ylim([0, np.max((np.abs(trace_carriers)))])
        '''
        # OPTICAL SPECTRUM
        [valid, lambdas, spectrum, lambdas_envelope, envelope, lambdas_phases, phases] = getspectrum(self.dt, self.lambda0, OUTPUTmonitor)
        if valid == True:
            freq_center = int(np.where(spectrum == max(spectrum))[0])
            lambda_center = lambdas[freq_center]
            ax2.plot(lambdas, spectrum, color='dimgrey')
            ax2.set_xlim([lambda_center - 20, lambda_center + 20])
            ax2.set_ylim([np.max(spectrum) - 25, 5 + np.max(spectrum)])

        # set axis labels
        ax1.set_xlabel('Time (ns)', fontsize=self.Fontsize)
        ax1.set_ylabel('Envelope (W)', fontsize=self.Fontsize)
        ax2.set_xlabel('Wavelength (nm)', fontsize=self.Fontsize)
        ax2.set_ylabel('Power spectral density (dBm/Hz)', fontsize=self.Fontsize)
        ax3.set_xlabel('Time offset (ps)', fontsize=self.Fontsize)
        ax3.set_ylabel('Envelope (W)', fontsize=self.Fontsize)
        ax4.set_xlabel('Time offset (ps)', fontsize=self.Fontsize)
        ax4.set_ylabel('Norm. carrier density', fontsize=self.Fontsize)
        #plt.show() #*** if you want to plot to show up after completing the simulation ***

        now = datetime.datetime.now()  # current date and time
        date_time = now.strftime("%m%d%Y-%H%M%S")
        figname=date_time+'_dt'+str(self.dt)+self.filename+'.png'
        fig.savefig(figname)

class Simulate_EXAMPLE_1GHzIIIVonSi:
    # ============================================================
    # Example 3 of a (InP-on-Si) mode-locked laser simulation with 1 GHz rep rate
    # Model based on the following laser:
    #       doi:10.1038/lsa.2016.260
    #	    although the parameters of the model still need tweaking to match the experimental results
    # *************************************************************
    #   Topology:
    #   Cavity mirror (left) <-> Semiconductor amplifier <-> Saturable absorber <-> Semiconductor amplifier <->
    #           ... Cavity waveguide (spiral) <-> Cavity mirror (right)
    #   Note: - dispersion, nonlinearity of the passive waveguides is included here (SSF is enabled)
    # *************************************************************
    def __init__(self):
        # -- Set simulation parameters --
        self.dt = 20 * 1e-15  # time step (convergence testing needed, value of 20fs typically ok)
        self.lambda0 = 1.60e-6  # wavelenght [m]
        self.lp = 37.4 * 1e-3  # length of passive cavity
        self.r_1 = 0.5  # mirror reflectivity (left)
        self.r_2 = 0.99  # mirror reflectivity (right)

        self.sim_time = 200 * (1e-9)  # Total simulation time [s]
        self.n_o_timesteps = int(np.round(self.sim_time / self.dt))

        # << SOA parameters >>
        Injection_current=43e-3 # injection current of SOA [A] #65 works #50 does not work, even with pulse injection
        InP_ng = 3.34
        InP_neff = 3.2
        InP_Aeff = 1.54 * 1e-12  # Mode area [m^2]
        InP_Aqw = 1.08 * 1e-13  # area of the quantum wells (active area)
        InP_conf = 0.075  # confinement factor
        SOA1_l = 140 * 1e-6  # length of SOA [m]
        SOA2_l = 660 * 1e-6  # length of SOA [m]
        SOA1_I = Injection_current*(SOA1_l/(SOA1_l+SOA2_l))  # injection current of SOA [A]
        SOA2_I = Injection_current*(SOA2_l/(SOA1_l+SOA2_l))  # injection current of SOA [A]

        SOA_Nt = 1e24  # transparency carrier density [m^3]
        SOA_RA = 0.8e9#1e9  # Nonradiative recombination rate [s^-1]
        SOA_RB = 7e-17  # Spontaneous recombination rate [m^3 s^-1]
        SOA_RC = 1e-41  # Auger recombination rate [m^6 s^-1]
        SOA_chi0 = 66e-21 #65e-21  # material gain constant [m^2]
        SOA_beta = 3e11  # linear internal losses semiconductor section, [s^-1]
        SOA_epsilon = 100e-24#100e-24 #20e-24  # Nonlinear gain compression [m^3]
        SOA_alpha = -2.75  # linewidth enhancement factor x transparency carrier density
        SOA_gaindisp_height = 50e2  # amplitude of gain dispersion [1/m]
        SOA_gaindisp_peak = 0  # relative center frequency of the Lorentzian gain spectrum
        SOA_gaindisp_FWHM = 50e-9 #30e-9  # FWHM of gain dispersion [m]

        # construct SOA's
        self.SOA1_parameters = [SOA_Nt, SOA_RA, SOA_RB, SOA_RC, SOA_chi0, SOA_beta, SOA1_l, SOA1_I, SOA_epsilon, SOA_alpha, InP_ng, InP_neff, InP_conf, InP_Aeff, InP_Aqw, SOA_gaindisp_height, SOA_gaindisp_peak, SOA_gaindisp_FWHM]
        self.SOA2_parameters = [SOA_Nt, SOA_RA, SOA_RB, SOA_RC, SOA_chi0, SOA_beta, SOA2_l, SOA2_I, SOA_epsilon, SOA_alpha, InP_ng, InP_neff, InP_conf, InP_Aeff, InP_Aqw, SOA_gaindisp_height, SOA_gaindisp_peak, SOA_gaindisp_FWHM]
        self.SOA_L = TWM(self.dt, self.lambda0, type=1, Parameters=self.SOA1_parameters)
        self.SOA_R = TWM(self.dt, self.lambda0, type=1, Parameters=self.SOA2_parameters)

        # << SA parameters >>
        SA_Nt = 1e24  # transparency carrier density [m^3]
        SA_RA = 0.85*1e11 #0.8*1e11#1e11  # Nonradiative recombination rate [s^-1]
        SA_RB = 0  # Spontaneous recombination rate [m^3 s^-1]
        SA_RC = 0  # Auger recombination rate [m^6 s^-1]
        SA_chi0 = 1100e-21#2000e-21  # material gain constant [m^2] #2000 works
        SA_beta = 3e11  # linear internal losses semiconductor section, [s^-1]
        SA_epsilon = 140e-24#140e-24 #40e-24  # Nonlinear gain compression [m^3]
        SA_alpha = -1.3  # linewidth enhancement factor x transparency carrier density
        SA_l = 55 * 1e-6  # length of SA [m]
        SA_gaindisp_height = 50e2  # amplitude of gain dispersion [1/m]
        SA_gaindisp_peak = 0  # relative center frequency of the Lorentzian gain spectrum
        SA_gaindisp_FWHM = 30e-9  # FWHM of gain dispersion [m]

        # construct SA
        self.SA_parameters = [SA_Nt, SA_RA, SA_RB, SA_RC, SA_chi0, SA_beta, SA_l, 0, SA_epsilon, SA_alpha, InP_ng, InP_neff, InP_conf, InP_Aeff, InP_Aqw, SA_gaindisp_height, SA_gaindisp_peak, SA_gaindisp_FWHM]
        self.SA = TWM(self.dt, self.lambda0, type=0, Parameters=self.SA_parameters)

        # -- Construct active and passive section model --
        cum_delay_other_components = 2 * self.SOA_L.n_o_spacesteps * self.dt + 2 * self.SOA_R.n_o_spacesteps * self.dt + 2 * self.SA.n_o_spacesteps * self.dt # roundtrip delay attributed to the active cavity components
        self.Spiral = Passive(self.dt, self.lambda0, self.lp, cum_delay_other_components, SSF=True)

        self.estimated_rep_rate = 1.0/((self.Spiral.lp/self.Spiral.v_g+self.SOA_L.l/self.SOA_L.v_g+self.SOA_R.l/self.SOA_R.v_g)*2)

    def run(self):
        # -- Initialize model --
        sigma = 0.1e-6  # used to scale some initial 'noise' samples
        self.SOA_L.initialize(sigma=sigma)
        self.SOA_R.initialize(sigma=sigma)
        self.SA.initialize(sigma=sigma)
        self.Spiral.initialize(sigma=sigma)

        # -- Initialize remaining field samples with noise --
        noise_samples = (sigma * (np.random.rand(8, 1) + 1j * np.random.rand(8, 1)))
        spiral_leftoutput = noise_samples[0]
        spiral_rightoutput = noise_samples[1]
        soa_l_leftoutput = noise_samples[2] / np.sqrt(self.SOA_L.Asemi)
        soa_l_rightoutput = noise_samples[3] / np.sqrt(self.SOA_L.Asemi)
        soa_r_leftoutput = noise_samples[4] / np.sqrt(self.SOA_R.Asemi)
        soa_r_rightoutput = noise_samples[5] / np.sqrt(self.SOA_R.Asemi)
        sa_leftoutput = noise_samples[6] / np.sqrt(self.SA.Asemi)
        sa_rightoutput = noise_samples[7] / np.sqrt(self.SA.Asemi)

        # -- Construct monitors for output capturing --
        WAVEmonitors = np.zeros((3,self.n_o_timesteps), dtype=np.complex128)
        CARRIERmonitors = np.zeros((2, self.n_o_timesteps), dtype=np.float64)

        # -- Pulse injection (self-starting not always straight forward! something to explore in a later stage) --
        # inject a sech pulse to reach desired operating point
        secht = np.linspace(-50e-12, 50e-12, int(round((100e-12) / self.dt)) + 1)
        pulsewidth = 10 * 1e-12 #8
        sechw = pulsewidth / 1.763
        Inputenergy = 2e-12 # energy in joules
        sechP0 = 0.88 * Inputenergy / (pulsewidth)
        inject = sech_functiondef(secht, sechP0, 0, sechw)
        pulseinjection_counter = 0

        # =================================================================================================================
        # Iterate
        percentage_cycle = np.floor(self.n_o_timesteps / 100)
        counter = 1
        start = time.perf_counter()

        print('Running EXAMPLE 1GHzIIIVonSi ...')
        print('Approximate repetition rate: '+str(self.estimated_rep_rate*1e-9)+ ' GHz')
        for i in range(1, self.n_o_timesteps):

            if pulseinjection_counter < len(secht):
            # inject pulse to reach mode-locked state
            # ... alternatively one could use a SA absorption that increases at startup (to more closely resemble experimental procedure where we gradually reverse bias the saturable absorber)
                leftinput = np.sqrt(self.r_1)*soa_l_leftoutput+inject[pulseinjection_counter]* (1 / np.sqrt(self.SOA_L.Asemi))
                pulseinjection_counter += 1
            else:
                leftinput = np.sqrt(self.r_1)*soa_l_leftoutput

            #leftinput = np.sqrt(self.r_1)*soa_l_leftoutput
            rightinput = sa_leftoutput * (np.sqrt(self.SA.Asemi) / np.sqrt(self.SOA_L.Asemi))
            [soa_l_leftoutput, soa_l_rightoutput] = self.SOA_L.step(i, leftinput, rightinput)

            leftinput = soa_l_rightoutput * (np.sqrt(self.SOA_L.Asemi) / np.sqrt(self.SA.Asemi))
            rightinput = soa_r_leftoutput * (np.sqrt(self.SOA_R.Asemi) / np.sqrt(self.SA.Asemi))
            [sa_leftoutput, sa_rightoutput] = self.SA.step(i, leftinput, rightinput)

            leftinput = sa_rightoutput * (np.sqrt(self.SA.Asemi) / np.sqrt(self.SOA_R.Asemi))
            rightinput = spiral_leftoutput * (1 / np.sqrt(self.SOA_R.Asemi))
            [soa_r_leftoutput, soa_r_rightoutput] = self.SOA_R.step(i, leftinput, rightinput)

            leftinput = soa_r_rightoutput * np.sqrt(self.SOA_R.Asemi)
            rightinput = spiral_rightoutput * np.sqrt(self.r_2)
            [spiral_leftoutput, spiral_rightoutput] = self.Spiral.step(i, leftinput, rightinput)

            # update monitors
            #OUTPUTmonitor[i - 1] = np.sqrt(1 - self.r_1) * (soa_l_leftoutput*np.sqrt(self.SOA_L.Asemi))
            WAVEmonitors[0][i - 1] = np.sqrt(1 - self.r_1) * (soa_l_leftoutput * np.sqrt(self.SOA_L.Asemi))
            WAVEmonitors[1][i - 1] = leftinput
            WAVEmonitors[2][i - 1] = spiral_leftoutput
            CARRIERmonitors[0][i - 1] = self.SOA_R.N[np.mod(i, 2)][int(np.round(self.SOA_R.n_o_spacesteps * 0.5))]
            CARRIERmonitors[1][i - 1] = self.SA.N[np.mod(i, 2)][int(np.round(self.SA.n_o_spacesteps * 0.5))]

            # -- Output completion status --
            if np.mod(i, percentage_cycle) == 0:
                print('... ' + str(counter) + ' % complete...[Elapsed time: 0' + str(time.perf_counter() - start) + ' s]')
                counter = counter + 1
                start = time.perf_counter()

        # =================================================================================================================
        # Data could be written to a csv file, e.g. using pd.DataFrame(OUTPUTmonitor).to_csv("temp.csv")
        #filename = 'Current' + str(self.SOA_L.I + self.SOA_R.I) + '_SOA_chi0' + str(self.SOA_L.chi0) + '_SA_chi0' + str(self.SA.chi0) + '_SOA_epsilon' + str(self.SOA_L.epsilon) + '_SA_epsilon' + str(self.SA.epsilon)+'_disppeak'+str(self.SOA_L.gaindisp_height)+'_SARA'+str(self.SA.RA)
        filename=''

        now = datetime.datetime.now()  # current date and time
        date_time = now.strftime("%m%d%Y-%H%M%S")
        name = date_time + '_dt' + str(self.dt) + filename + '.csv'

        #np.savetxt('OUTPUT_'+name, OUTPUTmonitor, delimiter=",")
        for i in range(0,len(WAVEmonitors)):
            np.savetxt('WAVES_MONITOR'+str(i)+'_'+ name, WAVEmonitors[i][:], delimiter=",")
        for i in range(0,len(CARRIERmonitors)):
            np.savetxt('CARRIER_MONITOR'+str(i)+'_'+ name, CARRIERmonitors[i][:], delimiter=",")

        # write parameters to txt file
        txtfilename = date_time + '_dt' + str(self.dt) + filename + '.txt'
        with open(txtfilename, "w") as text_file:
            text_file.write("Hybrid MLL model - example 3 \n")
            text_file.write("\n")

            text_file.write("--- General parameters --- \n")
            text_file.write("dt: %s \n" % self.dt)
            text_file.write("lambda0: %s \n" % self.lambda0)
            text_file.write("lp: %s \n" % self.lp)
            text_file.write("r_1: %s \n" % self.r_1)
            text_file.write("r_2: %s \n" % self.r_2)
            text_file.write("sim_time: %s \n" % self.sim_time)
            text_file.write("n_o_timesteps: %s \n" % self.n_o_timesteps)
            text_file.write("\n")

            text_file.write("--- SOA parameters --- \n")
            text_file.write("SOA1_parameters: %s \n" % self.SOA1_parameters)
            text_file.write("SOA2_parameters: %s \n" % self.SOA2_parameters)
            text_file.write("\n")

            text_file.write("--- SA parameters --- \n")
            text_file.write("SA_parameters: %s \n" % self.SA_parameters)
            text_file.write("\n")

        # -- Visualize data --
        vis = Visualize(self.dt, self.lambda0, self.n_o_timesteps,self.estimated_rep_rate,filename)
        vis.Plot(WAVEmonitors, CARRIERmonitors)


# ******************************
# ***    MAIN  ****
# ******************************
example = Simulate_EXAMPLE_1GHzIIIVonSi()
example.run()

