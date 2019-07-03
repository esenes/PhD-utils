print('Version 0.1')
print('Please report bugs to eugenio.senes@cern.ch\nUse at your own risk.')
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

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

def load_data_Tektronix(data_path, header_len=9):
    '''
    Import single file from the Tektronix MSO6 scope:

    Inputs:
    - data_path: the file including path

    Outputs:
    - data[0]: the timescale of the trace
    - data[1]: the signal of the trace
    - t_sampl: the sampling time

    '''
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

def dB(x):
    return 20*np.log10(x)

#---------------------------------------------------------------------
################## Signal processing routines ########################
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
