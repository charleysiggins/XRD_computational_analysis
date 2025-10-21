# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:10:29 2025

@author: Charley
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import curve_fit

def loaddata(file_path):
    """
    Loads data and metadata from given file
    
    Parameters:
    file_path (str): Path to input .dat file containing all information
        
    Returns:
    tuple: Returns a tuple of metadata dictionary, 2 theta array and intensity array
    """

    with open(file_path, 'r') as file:
        lines = file.readlines() 
    
    metadata = {}
    data_start_index = None 
    
    for i, line in enumerate(lines):
        if "=" in line:
            key, value = line.strip().split("=", 1)
            metadata[key.strip()] = value.strip()
        elif line.strip() and line[0].isdigit():
            data_start_index = i
            break 
    
    data = np.genfromtxt(file_path, skip_header=data_start_index) 
    theta_2 = data[:, 0]  
    intensity = data[:, 1] 
    
    return metadata, theta_2, intensity

def smoothintensity(intensity, window_size=5):
    """
    Applies moving average to smooth intensity array
    
    Parameters:
    intensity (np.ndarray): raw intensity data
    window_size (int): width of moving average filter (5)

    Returns:
    np.ndarray: smoothed intensity data
    
    I thought of the idea of using a weighted average and then found how to do so on https://www.projectpro.io/recipes/compute-averages-sliding-window-over-array
    """
    
    return np.convolve(intensity, np.ones(window_size)/window_size, mode='same')

def simplepeakfinding(theta_2, intensity, metadata):
    """
    Finds rough peak positions and assigns approximate Miller indices
    
    Parameters:
    theta_2 (np.ndarray): 2 theta values (in degrees)
    intensity (np.ndarray): smoothed intensity data
    metadata (dict): wavelength from metadata

    Returns:
    tuple: Tuple containing peak positions, peak intensities and assigned hkl dictionary
    """

    wavelength = float(metadata.get("Wavelength (Angstroms)", 1.54)) 

    min_height = np.max(intensity) * 0.1
    peaks, _ = find_peaks(intensity, height=min_height, distance=5) 
    
    peak_positions = theta_2[peaks]
    peak_intensities = intensity[peaks] 

    theta = np.radians(peak_positions / 2)
    d_values = wavelength / (2 * np.sin(theta)) 

    cubic_hkl = {
        "(111)": 1/np.sqrt(3),
        "(200)": 1/2,
        "(220)": 1/np.sqrt(8),
        "(311)": 1/np.sqrt(11),
        "(222)": 1/np.sqrt(12),
        "(400)": 1/4, 
    } 
    
    if len(d_values) > 0:
        a_estimate = d_values[0] / cubic_hkl["(111)"]
    else:
        a_estimate = None


    assigned_hkl = {}
    for i, d in enumerate(d_values):
        if a_estimate is not None:
            best_hkl = min(
                cubic_hkl,
                key=lambda h: abs(a_estimate * cubic_hkl[h] - d)
            )
        else:
            best_hkl = "Unknown"
        assigned_hkl[peak_positions[i]] = best_hkl 

    return peak_positions, peak_intensities, assigned_hkl

def plotdata(theta_2, smoothed_intensity, peak_positions, peak_intensities, assigned_hkl):
    """
    Plots diffraction pattern with annotated peaks
    
    Parameters:
    theta_2 (np.ndarray): 2 theta values (in degrees)
    smoothed_intensity (np.ndarray): smoothed and background corrected intensity
    peak_positions (np.ndarray): array of peak 2 theta values
    peak_intensities (np.ndarray): array of peak intensities
    assigned_hkl (dict): mapping of 2 theta to assigned Miller indices

    Returns:
    Displays the plot
    """

    plt.figure(figsize=(10, 8))
    plt.plot(theta_2, smoothed_intensity, linestyle='-', color='black', label="Corrected & Smoothed Data")
    for px, py in zip(peak_positions, peak_intensities):
        plt.scatter(px, py, color='red', marker='x', s=50)
        plt.annotate(assigned_hkl.get(px, ""), (px, py),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=10, color='red')
    plt.yscale('log')
    plt.xlabel(r'2$\theta$ (degrees)')
    plt.ylabel('Intensity (counts)')
    plt.title('X-ray Diffraction Pattern - py23c2s')
    plt.xlim(10, 80)
    plt.ylim(10, 1000)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    
def backgroundmodel(t, I1, centre):
    """
    Quadratic model for background intensity
    
    Parameters:
    t (np.ndarray): 2 theta values
    I1 (float): max intensity of background parabola
    centre (float): 2 theta position of the parabola peak

    Returns:
    np.ndarray: modeled background intensity
    """

    return I1 - ((t - centre)**2 / 100.0)

def fitbackground(theta_2, intensity, naive_positions, exclusion_width=2.0):
    """
    Fits quadratic background to intensity data (outside peak regions)

    Parameters:
    theta_2 (np.ndarray): 2 theta values
    intensity (np.ndarray): raw intensity data
    naive_positions (np.ndarray): peak positions to exclude
    exclusion_width (float): width to exclude around each peak (2.0)

    Returns:
    tuple: fitted background parameters: I1 and centre)
    """
    mask = np.ones_like(theta_2, dtype=bool) 
    for ppos in naive_positions:
        exclude_min = ppos - exclusion_width
        exclude_max = ppos + exclusion_width
        near_peak = (theta_2 >= exclude_min) & (theta_2 <= exclude_max)
        mask[near_peak] = False 

    x_bg = theta_2[mask] 
    y_bg = intensity[mask] 

    guess_I1 = np.max(y_bg) 
    guess_center = np.median(x_bg) 
    
    popt, _ = curve_fit(backgroundmodel, x_bg, y_bg, p0=[guess_I1, guess_center]) 
    I1_fit, ctheta_fit = popt
    return I1_fit, ctheta_fit

def complexpeakfinding(theta_2, intensity):
    """
    Finds peaks in background-subtracted intensity using a lower threshold
    
    Parameters:
    theta_2 (np.ndarray): 2 theta values
    intensity (np.ndarray): corrected smoothed intensity

    Returns:
    np.ndarray: indices of detected peaks
    """

    min_height = np.max(intensity) * 0.02
    peaks, _ = find_peaks(intensity, height=min_height, distance=5)
    return peaks 

def gaussian(x, A, x0, sigma):
    """
    Defines a Gaussian.
    
    Parameters:
    x (np.ndarray): x-axis values (2-theta)
    A (float): Amplitude (peak height)
    x0 (float): Centre of the peak (2-theta)
    sigma (float): Standard deviation (controls peak width)

    Returns:
    np.ndarray: Gaussian function values at x
    """

    return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

def gaussianfitting(theta_2, intensity, peaks, fit_range=5):
    """
    Fits gaussian curves to each peak

    Parameters:
    theta_2 (np.ndarray): 2 theta values
    intensity (np.ndarray): corrected intensity
    peaks (np.ndarray): indices of detected peaks
    fit_range (int): number of points to fit on each side of the peak (5)

    Returns:
    list: list of dictionaries containing fitted peaks and uncertainties
    
    This function was sent to Benjamin Maher, Alistair Ewing and Joseph Meakin
    Calculation of area variance was found using Gen AI
    """

    peak_info = []

    for peak_index in peaks:
        peak_theta = theta_2[peak_index]
        peak_intensity = intensity[peak_index]

        idx_min = max(0, peak_index - fit_range)
        idx_max = min(len(theta_2), peak_index + fit_range)
        fit_x = theta_2[idx_min:idx_max]
        fit_y = intensity[idx_min:idx_max] 

        p0 = [peak_intensity, peak_theta, 0.1] 

        try:
            popt, pcov = curve_fit(gaussian, fit_x, fit_y, p0=p0) 
            A, x0, sigma = popt
            perr = np.sqrt(np.diag(pcov))  
            dA, dx0_err, dsigma = perr

            fwhm = 2.355 * sigma
            fwhm_err = 2.355 * dsigma

            area = A * sigma * np.sqrt(2 * np.pi) 
            cov_A_sigma = pcov[0, 2]
            dIdA = sigma * np.sqrt(2 * np.pi)
            dIdsigma = A * np.sqrt(2 * np.pi)
            area_var = (dIdA**2)*(dA**2) + (dIdsigma**2)*(dsigma**2) \
                       + 2*dIdA*dIdsigma*cov_A_sigma
            area_err = np.sqrt(abs(area_var))

            info = {
                "2theta": x0,
                "2theta_error": dx0_err,
                "FWHM": fwhm,
                "FWHM_error": fwhm_err,
                "Area": area,
                "Area_error": area_err
            }
            peak_info.append(info)

        except RuntimeError:
            print(f"Warning: Gaussian fit failed for peak at 2 theta:{peak_theta:.2f}")

    return peak_info

def dspacing(wavelength, two_theta_detected):
    """
    Calculates d using Bragg’s Law

    Parameters:
    wavelength (float): wavelength in Angstroms
    two_theta_detected (float): 2 theta position of the peak (degrees)

    Returns:
    float: calculated d spacing in A
    """ 
    theta = np.radians(two_theta_detected / 2)
    return wavelength / (2 * np.sin(theta))

def dspacingerror(wavelength, two_theta, two_theta_err):
    """
    Estimates uncertainty in d from 2 theta error (uses finite differences)

    Parameters:
    wavelength (float): wavelength
    two_theta (float): measured 2 theta
    two_theta_err (float): uncertainty in 2 theta

    Returns:
    float: uncertainty in d
    
    I found the finite differences method of calculating uncertainties from using Gen AI, but coded it myself.
    This function was sent to Joseph Meakin
    """

    theta = np.radians(two_theta / 2)
    d = wavelength / (2*np.sin(theta))

    theta_plus = np.radians((two_theta + two_theta_err)/2)
    d_plus = wavelength / (2*np.sin(theta_plus))
    return abs(d_plus - d)

def parsehkl(hkl_label):
    """
    Parses a string Miller index label into a tuple of integers.

    Parameters:
    hkl_label (str): e.g. '(311)'

    Returns:
    tuple: tuple of integers (h, k, l)
    """
    if hkl_label == "Unknown":
        return (None, None, None)

    s = hkl_label.strip("()")
    return tuple(int(x) for x in s)


def latticeparameter(d_val, hkl_label):
    """
    Computes a for cubic crystals

    Parameters:
    d_val (float): d
    hkl_label (str): Miller index

    Returns:
    float: lattice parameter a
    """

    (h, k, l) = parsehkl(hkl_label)
    return d_val * np.sqrt(h**2 + k**2 + l**2)

def latticeparametererror(d_val, d_err, hkl_label):
    """
    Calculates uncertainty in a.

    Parameters:
    d_val (float): d
    d_err (float): uncertainty in d
    hkl_label (str): Miller index label

    Returns:
    float: Uncertainty in a
    """

    (h, k, l) = parsehkl(hkl_label)
    factor = np.sqrt(h**2 + k**2 + l**2)
    return factor * d_err

def alloycomposition(a_alloy, a_Au, a_Cu):
    """
    Uses Vegard’s Law to estimate alloy composition
    
    Parameters:
    a_alloy (float): a of alloy
    a_Au (float): a of Au
    a_Cu (float): a of Cu

    Returns:
    float: Estimated atomic fraction of Au
    """

    return (a_alloy - a_Cu)/(a_Au - a_Cu)

def compositionerror(a_alloy, a_alloy_err, a_Au, a_Cu):
    """
    Estimates error in composition from uncertainty in lattice parameter.

    Parameters:
    a_alloy (float): a of alloy
    a_alloy_err (float): uncertainty in a_alloy
    a_Au (float): a of Au
    a_Cu (float): a of Cu

    Returns:
    float: Uncertainty in composition
    """

    return abs(a_alloy_err/(a_Au - a_Cu))

def scherrersizeanderror(wavelength, two_theta, d2theta, fwhm_deg, dfwhm_deg, K=0.9):
    """
    Estimates crystallite size using Scherrer equation and finds uncertainty.

    Parameters:
    wavelength (float): wavelength
    two_theta (float): 2 theta peak position
    d2theta (float): uncertainty in 2 theta
    fwhm_deg (float): FWHM of peak
    dfwhm_deg (float): uncertainty in FWHM
    K (float): shape factor (0.9)

    Returns:
    tuple: crystallite size and its uncertainty (L, dL)
    """

    beta = np.radians(fwhm_deg)
    dbeta = np.radians(dfwhm_deg)
    theta = np.radians(two_theta/2)
    dtheta = np.radians(d2theta/2)

    L = (K*wavelength)/(beta*np.cos(theta))
    frac_err = np.sqrt((dbeta/beta)**2 + (dtheta/np.tan(theta))**2)
    dL = L*frac_err
    return L, dL

def ProcessData(filename):
    """
    Main function for autograder

    Parameters:
    filename (str): path to file

    Returns:
    dict: dictionary with peak data, alloy composition, lattice parameter and grain size (all with errors)
    """
    metadata, theta_2, intensity = loaddata(filename)
    smoothed = smoothintensity(intensity, window_size=7)
    naive_positions, naive_heights, naive_hkl = simplepeakfinding(theta_2, smoothed, metadata)
    
    plotdata(theta_2, smoothed, naive_positions, naive_heights, naive_hkl)

    I1_fit, ctheta_fit = fitbackground(theta_2, intensity, naive_positions)
    background = backgroundmodel(theta_2, I1_fit, ctheta_fit)
    corrected = intensity - background
    corrected = np.maximum(corrected, 1.0)
    corrected = gaussian_filter1d(corrected, sigma=3)
    corrected = uniform_filter1d(corrected, size=10)
    if np.min(corrected) < 0:
        corrected += abs(np.min(corrected)) + 1.0

    peaks = complexpeakfinding(theta_2, corrected)
    fitted_peaks = gaussianfitting(theta_2, corrected, peaks)

    a_Au = 0.40782
    a_Cu = 0.36149
    wavelength = float(metadata.get("Wavelength (Angstroms)", 1.54))

    results_list = []
    a_values = []
    a_errors = []

    for peak in fitted_peaks:
        peak_theta = peak["2theta"]
        closest_hkl = None
        smallest_diff = 5.0

        for naive_theta in naive_positions:
            diff = abs(naive_theta - peak_theta)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_hkl = naive_hkl[naive_theta]

        if closest_hkl is None or closest_hkl == "Unknown":
            continue

        d = dspacing(wavelength, peak_theta) * 0.1  
        d_err = dspacingerror(wavelength, peak_theta, peak["2theta_error"]) * 0.1 #converting angstroms to nm

        a = latticeparameter(d, closest_hkl)
        a_err = latticeparametererror(d, d_err, closest_hkl)

        a_values.append(a)
        a_errors.append(a_err)

        results_list.append({
            "2theta": peak["2theta"],
            "2theta_error": peak["2theta_error"],
            "d": d,
            "d_error": d_err,
            "FWHM": peak["FWHM"],
            "FWHM_error": peak["FWHM_error"],
            "Area": peak["Area"],
            "Area_error": peak["Area_error"]
        })

    if len(a_values) > 0:
        a_avg = np.mean(a_values)
        a_err_avg = np.sqrt(np.sum(np.array(a_errors)**2)) / len(a_errors)
        composition = alloycomposition(a_avg, a_Au, a_Cu)
        composition = max(0.0, min(1.0, composition))
        composition_err = compositionerror(a_avg, a_err_avg, a_Au, a_Cu)
    else:
        a_avg = None
        a_err_avg = None
        composition = None
        composition_err = None

    if len(fitted_peaks) > 0:
        first_peak = fitted_peaks[0]
        L, dL = scherrersizeanderror(
            wavelength,
            first_peak["2theta"],
            first_peak["2theta_error"],
            first_peak["FWHM"],
            first_peak["FWHM_error"]
        )
        L *= 0.1  
        dL *= 0.1
    else:
        L, dL = None, None

    return {
        "Peaks": results_list,
        "Composition": composition,
        "Composition_error": composition_err,
        "Grain size": L,
        "Grain size_error": dL
    }

if __name__ == "__main__":
    file_path = r"C:\Users\Charley\OneDrive - University of Leeds\computing 2\sem2 coursework\NEW CWK\assessmentdata.dat"
    results = ProcessData(file_path)
    print("\nFinal Results Dictionary:\n")
    print(results)




