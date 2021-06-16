print('Version 0.1')
print('Please report bugs to eugenio.senes@cern.ch\nUse at your own risk.')
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import scipy.signal

#---------------------------------------------------------------------
############################# Units ##################################
#---------------------------------------------------------------------

def dBm2mW(dBm):
    '''
    dBm to mW conversion
    '''
    return 10**(dBm/10)

def dBm2W(dBm):
    '''
    dBm to W conversion
    '''
    return 10**((dBm-30)/10)

def mW2dBm(mW):
    '''
    mW to dBm conversion
    '''
    return  10. * (np.log10(mW))

def W2dBm(W):
    '''
    W to dBm conversion
    '''
    return 30. + 10. * (np.log10(W))

#---------------------------------------------------------------------
############################# Math ###################################
#---------------------------------------------------------------------

def gaussian(x, A, mu, sigma, normalisation=False):
    '''
    Standard gaussian. If normalisation=True then A is ignored
    '''
    if normalisation:
        A =  A=1./(sigma * np.sqrt(2*np.pi))
    return A*np.exp(-0.5*((((x-mu)/sigma)**2)))

# Calculate the mean charge
def weighted_average(x, sig_x):
    '''
    Calculate the aritmethical weighted average, usign the inverse variances weighting.
    See https://en.wikipedia.org/wiki/Inverse-variance_weighting. 
    
    Returns mean and error on the mean
    '''
    x = np.array(x); sig_x = np.array(sig_x)
    assert(all(sig_x>0))
    
    sig_x_m = 1./np.sum(1./(sig_x**2))
    x_m = np.sum(x/(sig_x**2)) * sig_x_m
    
    return x_m, sig_x_m
    
    


#---------------------------------------------------------------------
################## Import LeCroy SDA18000 Scope ######################
#---------------------------------------------------------------------

def load_data_LeCroy(data_path, header_len=5):
    '''
    Import single file from the LeCroy SDA18000 scope:

    Inputs:
    - data_path: the file including path

    Outputs:
    - data[0]: the timescale of the trace
    - data[1]: the signal of the trace
    - t_sampl: the sampling time

    '''
    data = np.loadtxt(data_path, delimiter=' ', skiprows=header_len)
    data = data.transpose()
    t_sampl = np.abs(data[0][1]-data[0][0])
    N_sampl = data[1].size


    return data[0], data[1], t_sampl

#---------------------------------------------------------------------
################## Import Tektronix MSO6 scope  ######################
#---------------------------------------------------------------------

def load_data_Tektronix(data_path, header_len=9, new_firmware=True, **kwargs):
    '''
    Import single file from the Tektronix MSO64 scope:

    Inputs:
    - data_path: the file including path

    Outputs:
    - data[0]: the timescale of the trace
    - data[1]: the signal of the trace
    - t_sampl: the sampling time

    NEW FIRMWARE:
    now all the data are stored in single file. Returns time, ch1, ch2, ...
    adapts automatically to the number of channels
    '''
    if new_firmware:
        data = np.loadtxt(data_path, delimiter=',', skiprows=14, **kwargs)
        data = data.transpose()
        # switch depending on the number of channels
        if data.shape[0] == 2:
            return data[0], data[1]
        elif data.shape[0] == 3:
            return data[0], data[1], data[2]
        elif data.shape[0] == 4:
            return data[0], data[1], data[2], data[3]
        elif data.shape[0] == 5:
            return data[0], data[1], data[2], data[3], data[4]
    else:
        data = np.loadtxt(data_path, delimiter=',', skiprows=header_len)
        data = data.transpose()
        t_sampl = np.abs(data[0][1]-data[0][0])
        N_sampl = data[1].size
        return data[0], data[1], t_sampl


#---------------------------------------------------------------------
############################### FFTs #################################
#---------------------------------------------------------------------

def doFFT(t, y):
    '''
    Do the FFT. Beware there is a factor 2 in the normalisation, so look at the right-hand side of the spectrum only.

    Inputs:
    - t:       the timescale
    - y:       the function value

    Outputs:
    - f_fft:        the frequncies
    - s_fft:        the power spectrum

    Last modified: 24.06.2019 by Eugenio Senes
    '''
    assert t.size == y.size, 'Different size in X and Y in the doFFT function'
    dt = np.abs(t[1]-t[0])
    N_sample = t.size

    norm = 2./N_sample

    s_fft = norm*np.abs(np.fft.fft(y))
    f_fft = np.fft.fftfreq(N_sample, d=dt)
    df = np.abs(f_fft[1]-f_fft[0])

    # reorder the fft
    f_fft = np.fft.fftshift(f_fft)
    s_fft = np.fft.fftshift(s_fft)

    return f_fft, s_fft

def doFFT_complex(t, y, reorder=False):
    '''
    Complex FFT. Use for the manual filtering routines.
    If reorder=True then the spectrum is refolded correctly.

    Inputs:
    - t:       the timescale
    - y:       the function value

    Outputs:
    - f_fft:        the frequncies
    - s_fft:        the power spectrum

    Last modified: 24.06.2019 by Eugenio Senes
    '''
    assert t.size == y.size, 'Different size in X and Y in the doFFT function'
    dt = np.abs(t[1]-t[0])
    N_sample = t.size

    s_fft = np.fft.fft(y)
    f_fft = np.fft.fftfreq(N_sample, d=dt)

    # reorder the fft
    if reorder:
        f_fft = np.fft.fftshift(f_fft)
        s_fft = np.fft.fftshift(s_fft)

    return f_fft, s_fft


def dB(x):
    return 20*np.log10(x)

def zero_pad_linear_baseline(x, y, final_length=8*2048, interval=15, debug=False):
    '''
    Zero-pad a signal up to the final_length (overall length). The baseline
    subtraction is done fitting the baseline in the first and last <interval>
    points.

    Inputs:
    - x, y:         the signal
    - final_length: the final length after the padding. Use powers of two for
                    performance of the FFT later.
    - interval:     number of points to consider at the beginning and end to fit
                    the baseline
    - debug:        if True, the function returns the points to fit the baseline
                    and the function with the baseline subtracted

    Outputs:
    - x,y:          the signal after the zero padding

    Last modified: 04.07.2019 by Eugenio Senes
    '''

    x_sampl = np.concatenate((x[:interval], x[-interval:]))
    y_sampl = np.concatenate((y[:interval], y[-interval:]))

    pol = np.polyfit(x_sampl, y_sampl,1)
    yy_no_bline = y - (x*pol[0] + pol[1])

    # zero padding
    assert (final_length > len(x)) & (np.mod(final_length-len(x),2) == 0)
    pad_len = final_length - len(x)
    yy_zero_pad = np.concatenate((np.zeros(int(pad_len/2)), yy_no_bline, np.zeros(int(pad_len/2))))
    dt = x[1]-x[0]
    x_expanded = np.concatenate((np.linspace(x[0]-(pad_len/2)*dt, x[0]-dt, int(pad_len/2)), x, np.linspace(x[-1]+dt, x[-1]+(pad_len/2)*dt, int(pad_len/2))))

    return [[x_sampl, y_sampl], [x, yy_no_bline], [x_expanded, yy_zero_pad]] if debug else [x_expanded, yy_zero_pad]


#---------------------------------------------------------------------
################## Signal processing routines ########################
#---> Peaks processing
#---------------------------------------------------------------------

def find_FWHM(x, y):
    '''
    Find FWHM of a peak. Based on streak camera and scope signals
    of the laser pulse. Works well for a single peak with not too
    much noise at half maximum of the peak.
    Returns coordinates of the points described in output.

    The algorithm finds the half maximum, then the closest two points
    of the signal on both sides of the peak. Interpolates a straight
    line to get the x of both points at half maximum and then returns
    the coordinates of the two.
    Finally, the location of the center peak is returned too.

    Test that everything is alright usign the test_FWHM function.

    In case of error, e.g. if one of the two sides can not be found
    because the profile sits on the edge of the data window, NaNs
    are returned.

    Inputs:
    - x, y: input data, usually x is time and y intensity.

    Outputs:
    - FWHM: the value of the FWHM
    - [FWHM, HM]: the x position of the center peak and the value of the half maximum
    - theMax: x and y coordinates of the absolute maximum of the signal
    - [left_x, left_y]: x and y coordinates of the left point at half maximum
    - [right_x, right_y]: x and y coordinates of the right point at half maximum

    Last modified: 01.07.2019 by Eugenio Senes
    '''
    assert len(x) == len(y)

    try:
        #find max
        max_idx = np.argmax(y)
        theMax = [x[max_idx], np.max(y)]
        HM = theMax[1]/2.

        # find the point before and after half max
        # split yy
        xx_l = x[:max_idx]
        yy_l = y[:max_idx]
        # find point before and after treshold
        left_a = yy_l[yy_l > theMax[1]/2][0]; left_a_x = xx_l[yy_l > theMax[1]/2][0]
        left_b = yy_l[yy_l < theMax[1]/2][-1]; left_b_x = xx_l[yy_l < theMax[1]/2][-1]
        lx = [left_b_x, left_a_x]
        ly = [left_b, left_a]
        # interpolate
        c_l = np.polyfit(lx, ly, 1)
        left_x = (HM-c_l[1])/c_l[0]
        left_y = c_l[0]*left_x + c_l[1]

        # same story, other side of the peak
        # split yy
        xx_r = x[max_idx:]
        yy_r = y[max_idx:]
        # find point before and after treshold
        right_a = yy_r[yy_r > theMax[1]/2][-1]; right_a_x = xx_r[yy_r > theMax[1]/2][-1]
        right_b = yy_r[yy_r < theMax[1]/2][0]; right_b_x = xx_r[yy_r < theMax[1]/2][0]
        rx = [right_b_x, right_a_x]
        ry = [right_b, right_a]
        # interpolate
        c_l = np.polyfit(rx, ry, 1)
        right_x = (HM-c_l[1])/c_l[0]
        right_y = c_l[0]*right_x + c_l[1]

        # find maximum location at middle peak
        FWHM_x = left_x+ 0.5*(right_x-left_x)

    except IndexError:
        FWHM_x = np.nan
        HM = np.nan
        theMax = np.nan
        left_x = np.nan
        left_y = np.nan
        right_x = np.nan
        right_y = np.nan

    FWHM = right_x - left_x

    return [FWHM, [FWHM_x, HM], theMax, [left_x, left_y], [right_x, right_y]]

def test_FWHM(x, y):
    '''
    Test function for FWHM, takes the same inputs.
    '''

    fig, ax = plt.subplots(1, figsize=(12,7))
    ax.plot(x, y, '-o', label="Input data")

    output = find_FWHM(x, y)
    print(output)

    ax.axvline(output[2][0], label='Max', color='k')
    ax.axhline(output[1][1], label='Half Max', color='r')

    ax.axvline(output[3][0],label='Left_interp', color='g')
    ax.axvline(output[4][0],label='Right_interp', color='g')

    ax.axvline(output[1][0],label='Peak_from_FWHM', color='orange')
    ax.legend()

    return None


#---------------------------------------------------------------------
################## Signal processing routines ########################
#---> Data fitting
#---------------------------------------------------------------------
def fitlin(xdata, ydata, verbose=False):
    '''
    Do a linear fit of the data. Return the parameters and the errors on them.
    Params are a*x + b.

    Last modified: 29.10.2019 by Eugenio Senes
    '''
    from scipy.optimize import curve_fit
    def model(x, a, b):
        return (a*x) + b

    p, cov = curve_fit(model, xdata, ydata)
    s_p = np.sqrt(np.diag(cov))

    if verbose:
        #fit and plot fit
        print('Model function: a*x + c')
        print('Fit params: a=' + str(p[0]) + ' c= '+ str(p[1]))
        print('Std params: s2_a=' + str(cov[0]) + ' s2_c= '+ str(cov[1]))

    return p, s_p

def fitlin_and_plot(xdata, ydata):
    '''
    Quick plot and linear fit of the data

    Last modified: 29.10.2019 by Eugenio Senes
    '''
    from scipy.optimize import curve_fit
    fig, ax = plt.subplots(1, dpi=150)
    ax.plot(xdata, ydata,'.', label='Data')

    popt, s_popt = fitlin(xdata, ydata)
    ax.plot(xdata, popt[0]*xdata+popt[1], label='Fit')
    ax.text(0.05, 0.25, 'Fit function: $a*x + b$:\n $a=$%.3E ± %.3E \n $c=$%.2E ± %.3E'
        %(popt[0],s_popt[0],popt[1], s_popt[1]),
        transform=ax.transAxes, fontsize=14,verticalalignment='top')
    ax.legend(frameon=True)

    return fig, ax



#---------------------------------------------------------------------
################## Signal processing routines ########################
#---> Filtering
#---------------------------------------------------------------------

def design_cheby2(N, gstop, fstop, ts, print_filter=True):
    '''
    Design a Cheby2 filter, attenuating gstop dB at fstop.

    Inputs:
    - N:        filter order
    - gstop:    attenuation at the stopband
    - fstop:    frequency of the stopband
    - ts:       sampling time

    Outputs:
    - a, b:     the filter polinomial coefficients

    Last modified: 16.08.2019 by Eugenio Senes
    '''
    # filter designs
    b, a = scipy.signal.cheby2(N, gstop, fstop*2*ts)

    if print_filter:
        w, h = scipy.signal.freqz(b,a)

        fig, ax = plt.subplots(1)
        ax.plot(1e-9*w/np.pi /(2*ts), 20 * np.log10(abs(h)), '-o', label='Cheby2')


        ax.set_xlabel('Frequency (GHz)')
        ax.grid()
        ax.legend()
        ax.axvline(x=fstop*1e-9)

    return a, b
