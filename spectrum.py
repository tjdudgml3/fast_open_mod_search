import functools
import math

import mmh3
import numba as nb
import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum
import logging
from ann_solo.config import config
import copy
import re

@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of high enough quality to be used for matching.

    Parameters
    ----------
    spectrum : np.ndarray
        M/z peaks of the sspectrum whose quality is checked.

    Returns
    -------
    bool
        True if the spectrum has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)


@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The spectrum peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def process_spectrum(spectrum: MsmsSpectrum, is_library: bool) -> MsmsSpectrum:
    """
    Process the peaks of the MS/MS spectrum according to the config.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The spectrum that will be processed.
    is_library : bool
        Flag specifying whether the spectrum is a query spectrum or a library
        spectrum.

    Returns
    -------
    MsmsSpectrum
        The processed spectrum. The spectrum is also changed in-place.
    """
    if spectrum.is_processed:
        return spectrum

    min_peaks = config.min_peaks
    min_mz_range = config.min_mz_range

    spectrum = spectrum.set_mz_range(config.min_mz, config.max_mz)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    if config.resolution is not None:
        spectrum = spectrum.round(config.resolution, 'sum')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    if config.remove_precursor:
        spectrum = spectrum.remove_precursor_peak(
            config.remove_precursor_tolerance, 'Da', 2)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    spectrum = spectrum.filter_intensity(
        config.min_intensity, (config.max_peaks_used_library if is_library else
                               config.max_peaks_used))
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = config.scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(
            scaling, max_rank=(config.max_peaks_used_library if is_library else
                               config.max_peaks_used))

    spectrum.intensity = _norm_intensity(spectrum.intensity)

    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing of library spectra for multiple queries.
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum


@functools.lru_cache(maxsize=None)
def get_dim(min_mz, max_mz, bin_size):
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Args:
        min_mz: The minimum mass in the mass range (inclusive).
        max_mz: The maximum mass in the mass range (inclusive).
        bin_size: The bin size (in Da).

    Returns:
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    min_mz, max_mz = float(min_mz), float(max_mz)
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return round((end_dim - start_dim) / bin_size), start_dim, end_dim



def generate_theoretical_spectrum(sequence):
    """
    주어진 아미노산 시퀀스를 통해 이론적인 b,y 이온 스펙트럼 생성.

    Args:
        sequence (str): 아미노산 시퀀스 (ex: "ACDEFGHIK")

    Returns:
        mz (list): b,y 이온의 m/z 값 리스트
        intensity (list): 각 이온의 intensity (모두 1로 고정)
    """
    amino_acid_mass = {
            "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
            "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
            "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
            "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
            "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841
    }
    amino_acid_mass["1"] = amino_acid_mass["C"] + 57.021
    amino_acid_mass["2"] = amino_acid_mass["N"] + 0.984
    amino_acid_mass["3"] = 42.011
    amino_acid_mass["4"] = 43.005
    amino_acid_mass["5"] = amino_acid_mass["M"] + 15.995
    amino_acid_mass["6"] = amino_acid_mass["Q"] + 0.984
    
    # H (1.00784), OH (17.00274), H2O (18.01056) 질량
    proton_mass = 1.00784
    h2o_mass = 18.01056
    b_ions = []
    y_ions = []

    # b ion 계산
    b_mass = 0
    for i, aa in enumerate(sequence):
        b_mass += amino_acid_mass[aa]  # 아미노산 질량 누적
        b_mz = b_mass + proton_mass  # 단일 양자화된 상태 [M+H]+
        b_ions.append(b_mz)

    # y ion 계산
    y_mass = h2o_mass  # H2O 질량 포함 시작
    for i in range(len(sequence) - 1, -1, -1):  # C-말단부터 N-말단 방향
        y_mass += amino_acid_mass[sequence[i]]  # 아미노산 질량 누적
        y_mz = y_mass + proton_mass  # 단일 양자화된 상태 [M+H]+
        y_ions.append(y_mz)

    # mz와 intensity 반환
    mz = b_ions + y_ions  # b, y 이온 결합
    intensity = [1] * len(mz)  # 모든 intensity를 1로 고정

    return mz, intensity

@functools.lru_cache(maxsize=None)
def hash_idx(bin_idx: int, hash_len: int) -> int:
    """
    Hash an integer index to fall between 0 and the given maximum hash index.

    Parameters
    ----------
    bin_idx : int
        The (unbounded) index to be hashed.
    hash_len : int
        The maximum index after hashing.

    Returns
    -------
    int
        The hashed index between 0 and `hash_len`.
    """
    return mmh3.hash(str(bin_idx), 42, signed=False) % hash_len

# def get_complimentary_spectrum(spectrum, precursor_mass, precursor_charge):
#     new_spectrum = copy.deepcopy(spectrum)
#     mass_tolerance = 0.04
    
#     for idx in range(len(spectrum)):
#         peak_charge = 1
#         precursor_cal = precursor_mass*precursor_charge - precursor_charge*1.00784
#         ion_cal = spectrum[idx][0]*peak_charge - peak_charge*1.00784
#         complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
#         sum_inten = 0
#         for idx2 in range(len(spectrum)):
#             if abs(spectrum[idx2][0]-complimentary_peak) < mass_tolerance:
#                 sum_inten += spectrum[idx2][1]
#         if sum_inten > 0 :
#             new_spectrum[idx][1] = spectrum[idx][1] + sum_inten
#             temp = spectrum[idx][0]
#         else:
#             new_spectrum.append([complimentary_peak, spectrum[idx][1]]) 
    
#     new_spectrum.sort()
    
#     return new_spectrum


def get_spectrum_weight(spectrum):
    new_spectrum = []
    for idx in range(len(spectrum.mz)):
        
        if spectrum.mz[idx] > 0.5*(spectrum.precursor_mz*spectrum.precursor_charge):
            new_spectrum.append([spectrum.mz[idx],spectrum.intensity[idx]*0.5])
        else:
            new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
    return new_spectrum


def get_complimentary_spectrum(spectrum):
    new_spectrum = []
    mass_tolerance = 0.04
    
    for idx in range(len(spectrum.mz)):
        peak_charge = 1
        precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
        ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
        complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
        sum_inten = 0
        new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
        new_spectrum.append([complimentary_peak, spectrum.intensity[idx]]) 
    
    new_spectrum.sort()
    
    return new_spectrum




def get_complimentary_spectrum_half(spectrum):
    new_spectrum = []
    mass_tolerance = 0.04
    
    for idx in range(len(spectrum.mz)):
        peak_charge = 1
        precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
        ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
        complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
        sum_inten = 0
#         new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
#         if complimentary_peak>0.5*(spectrum.precursor_mz*spectrum.precursor_charge):
#                 new_intensity = 0.5*spectrum.intensity[idx]
#         else:
#             new_intensity = spectrum.intensity[idx]

        new_intensity = spectrum.intensity[idx]
#             new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
        new_spectrum.append([complimentary_peak, new_intensity])
#         new_spectrum.append([complimentary_peak, spectrum.intensity[idx]]) 
    
    new_spectrum.sort()
    
    return new_spectrum

def get_complimentary_spectrum_weight(spectrum):
    new_spectrum = []
    mass_tolerance = 0.04
    
    for idx in range(len(spectrum.mz)):
        peak_charge = 1
        precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
        ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
        complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
        sum_inten = 0
        new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
        if complimentary_peak>0.5*(spectrum.precursor_mz*spectrum.precursor_charge):
                new_intensity = 0.5*spectrum.intensity[idx]
        else:
            new_intensity = spectrum.intensity[idx]

#         new_intensity = spectrum.intensity[idx]
#             new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
        new_spectrum.append([complimentary_peak, new_intensity])
#         new_spectrum.append([complimentary_peak, spectrum.intensity[idx]]) 
    
    new_spectrum.sort()
    
    return new_spectrum


def window_filtering(spectrum, window_size=100, top_n=4, mz_range=(0, 4000)):
    """
    Filters the given spectrum using a window-based approach.

    Parameters:
        spectrum (list): A list of [mz, intensity] pairs representing the spectrum.
        window_size (int): The size of each m/z window (default: 100).
        top_n (int): The number of top peaks to retain in each window (default: 3).
        mz_range (tuple): The range of m/z to consider (default: (0, 3000)).

    Returns:
        list: A filtered spectrum with top peaks in each window.
    """
    filtered_spectrum = []

    # Sort the spectrum by m/z values
    spectrum = sorted(spectrum, key=lambda x: x[0])

    # Iterate through the m/z range in steps of window_size
    for start in range(mz_range[0], mz_range[1], window_size):
        end = start + window_size

        # Collect peaks within the current window
        window_peaks = [peak for peak in spectrum if start <= peak[0] < end]

        # Sort peaks by intensity in descending order and take the top N
        top_peaks = sorted(window_peaks, key=lambda x: x[1], reverse=True)[:top_n]
#         top_peaks_new = []
        if top_peaks:
            top_inten = max([inten[1] for inten in top_peaks])
            top_peaks_new = []
            for peak in top_peaks:
                top_peaks_new.append([peak[0], peak[1]/top_inten])
            top_peaks = top_peaks_new
        # Add the top peaks to the filtered spectrum
        filtered_spectrum.extend(top_peaks)

    return filtered_spectrum


def get_spectrum_query_seq(spectrum):
    new_spectrum = []
    mass_tolerance = 0.04
    
    for idx in range(len(spectrum.mz)):
        peak_charge = 1
        precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
        ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
        complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
        sum_inten = 0
        new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
#         new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
#         new_spectrum.append([complimentary_peak, spectrum.intensity[idx]]) 
    
    new_spectrum.sort()
#     logging.debug(new_spectrum)
    filtered_spectrum = window_filtering(new_spectrum)
#     for idx1 in range(len(new_spectrum)):
        
    filtered_spectrum.sort()
#     logging.debug(filtered_spectrum)
#     exit()
    return filtered_spectrum


def get_spectrum_query_seq_com(spectrum):
    new_spectrum = []
    mass_tolerance = 0.04
    
    for idx in range(len(spectrum.mz)):
        peak_charge = 1
        precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
        ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
        complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
        sum_inten = 0
#         new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
        new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
#         new_spectrum.append([complimentary_peak, spectrum.intensity[idx]]) 
    
    new_spectrum.sort()
#     logging.debug(new_spectrum)
    filtered_spectrum = window_filtering(new_spectrum)
#     for idx1 in range(len(new_spectrum)):
        
    filtered_spectrum.sort()
#     logging.debug(filtered_spectrum)
#     exit()
    return filtered_spectrum


def get_complimentary_spectrum_annotated_half(spectrum):
    
    new_spectrum = []
    mass_tolerance = 0.025
    raw_list = []
    for a in spectrum.annotation:
        if len(a)>1:
            raw_list.append(a[1])
        else:
            raw_list.append(a[0])
            
    # logging.debug(raw_list)
    for idx in range(len(spectrum.mz)):
        complimentary_peak = 0
        peak_charge = 1
        loss = 0
        if  b'?/n' != raw_list[idx]:
            anno = str(raw_list[idx])[2:-1]
#             logging.debug(anno)
            
            
            first = anno.split("/")[0]
#             logging.debug(first)
#             exit()
            if "^" not in first:
                peak_charge = 1
            else:
                peak_charge = int(first.split("^")[1][0])

            if "-" in first:
                try:
                    loss = int(first.split("-")[1][:2])
                except:
                    loss = int(first.split("-")[1][:1])
            else:
                loss = 0
                
#             peak_charge = spectrum.annotation[idx][0].charge
#             if peak_charge > 1:
#                 new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
#                 continue
            
            precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
            ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784 + loss
            complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge -loss/peak_charge
            sum_inten = 0
#             sum_inten_nh3 = 0
#             sum_inten_h2o = 0
#             logging.debug(f"original mz = {spectrum.mz[idx]}, complementary = {complimentary_peak}, precusor = {spectrum.precursor_mz}, {spectrum.precursor_charge}, annotation = {first}, precursor_cal = {precursor_cal}, peak_charge = {peak_charge}, ion_cal = {ion_cal}")
            new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
            
            
    new_spectrum.sort()
#     logging.debug(new_spectrum)
#     exit()
    # print(new_spectrum)
    return new_spectrum

def get_complimentary_spectrum_lib(spectrum):
    
    new_spectrum = []
    mass_tolerance = 0.025
    raw_list = []
    for a in spectrum.annotation:
        if len(a)>1:
            raw_list.append(a[1])
        else:
            raw_list.append(a[0])
            
    # logging.debug(raw_list)
    for idx in range(len(spectrum.mz)):
        complimentary_peak = 0
        peak_charge = 1
        loss = 0
        if  b'?/n' != raw_list[idx]:
            anno = str(raw_list[idx])[2:-1]
#             logging.debug(anno)
            
            
            first = anno.split("/")[0]
#             logging.debug(first)
#             exit()
            if "^" not in first:
                peak_charge = 1
            else:
                peak_charge = int(first.split("^")[1][0])

            if "-" in first:
                try:
                    loss = int(first.split("-")[1][:2])
                except:
                    loss = int(first.split("-")[1][:1])
            else:
                loss = 0
                
#             peak_charge = spectrum.annotation[idx][0].charge
#             if peak_charge > 1:
#                 new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
#                 continue
            
            precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
            ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
            complimentary_peak = (precursor_cal - ion_cal + 1.00784)
            sum_inten = 0
#             sum_inten_nh3 = 0
#             sum_inten_h2o = 0
#             logging.debug(f"original mz = {spectrum.mz[idx]}, complementary = {complimentary_peak}, precusor = {spectrum.precursor_mz}, {spectrum.precursor_charge}, annotation = {first}, precursor_cal = {precursor_cal}, peak_charge = {peak_charge}, ion_cal = {ion_cal}")
            
    
            #for new method
            if complimentary_peak>0.5*(spectrum.precursor_mz*spectrum.precursor_charge):
                new_intensity = 0.5*spectrum.intensity[idx]
            else:
                new_intensity = spectrum.intensity[idx]
            
            
#             new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
            new_spectrum.append([complimentary_peak, new_intensity])
            
            
    new_spectrum.sort()
#     logging.debug(new_spectrum)
#     exit()
    # print(new_spectrum)
    return new_spectrum



def get_complimentary_spectrum_annotated(spectrum):
    
    new_spectrum = []
    mass_tolerance = 0.025
    raw_list = []
    for a in spectrum.annotation:
        if len(a)>1:
            raw_list.append(a[1])
        else:
            raw_list.append(a[0])
            
    # logging.debug(raw_list)
    for idx in range(len(spectrum.mz)):
        complimentary_peak = 0
        if len(spectrum.annotation[idx]) > 1:
            peak_charge = spectrum.annotation[idx][0].charge
            if peak_charge > 1:
                new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
                continue
                
            precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
            ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
            complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
            sum_inten = 0
#             sum_inten_nh3 = 0
#             sum_inten_h2o = 0
            nh3_idx = 9999
            h2o_idx = 9999
            
            for idx_i in range(len(spectrum.mz)):
                
                temp_annotation = ''
                
                #validation needed ->raw_list
                if b'?/n' != raw_list[idx_i]:
                    # logging.debug(f"raw = {raw_list[idx_i]}")
                    raw_list[idx_i] = str(raw_list[idx_i])
                    # logging.debug(f"str = {raw_list[idx_i]}")
                    temp_annotation = raw_list[idx_i].split(",")[0][2:]
                    # logging.debug(temp_annotation)
                
                # logging.debug(f"annotation = {temp_annotation}")    
                if abs(spectrum.mz[idx] - 17.031 -spectrum.mz[idx_i]) < mass_tolerance and "-17" in temp_annotation:
                    nh3_idx = idx_i
                    # logging.debug(f"nh3")
#                     new_spectrum.append([complimentary_peak-17.031, spectrum.intensity[idx]*0.3])
#                     sum_inten_nh3 += spectrum[idx_i][1]
#                     complementary_peak_nh3 = complimentary_peak - 17.031
                #neutral loss h2o    
                if abs(spectrum.mz[idx] - 18.015 -spectrum.mz[idx_i]) < mass_tolerance and "-18" in temp_annotation:
#                     new_spectrum.append([complimentary_peak-18.015, specterum.intensity[idx]*0.3])
                    h2o_idx = idx_i
                    # logging.debug(f"h2o")

                if abs(spectrum.mz[idx_i]-complimentary_peak) < mass_tolerance:
                    sum_inten += spectrum.intensity[idx_i]
                    
            if sum_inten > 0:
                new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx] + sum_inten])
            else:
                new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
                new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
            if nh3_idx !=9999:
                new_spectrum.append([complimentary_peak-17.031, spectrum.intensity[idx]*0.3])
            if h2o_idx != 9999:
                new_spectrum.append([complimentary_peak-18.015, spectrum.intensity[idx]*0.3])
                
            
        else:
            new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
            
            
    new_spectrum.sort()
    # print(new_spectrum)
    return new_spectrum


def get_spectrum_half(spectrum):
    pass

           
#     for idx in range(len(spectrum)):
#         complimentary_peak = 0
#         complementary_peak_i_1 = 0
#         complementary_peak_i_2 = 0
#         complementary_peak_nh3 = 0
#         complementary_peak_h2o = 0
#         if spectrum[idx][2]:
#             peak_charge = spectrum[idx][2].charge
#             if peak_charge > 1:
#                 continue
#             precursor_cal = precursor_mass*precursor_charge - precursor_charge*1.00784
#             ion_cal = spectrum[idx][0]*peak_charge - peak_charge*1.00784
#             complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
#             sum_inten = 0
#             sum_inten_i_1 = 0
#             sum_inten_i_2 = 0
#             sum_inten_nh3 = 0
#             sum_inten_h2o = 0
#             i_1_idx = 9999
#             i_2_idx = 9999
#             nh3_idx = 9999
#             h2o_idx = 9999
            
            
#             for idx_i in range(len(spectrum)):
#                 #isotope 1
#                 if abs(spectrum[idx][0] + 1-spectrum[idx_i][0]) < mass_tolerance:
#                     i_1_idx = idx_i
# #                     sum_inten_i_1 += spectrum[idx_i][1]
# #                     complementary_peak_i_1 = complimentary_peak + 1
#                 #isotope 2    
#                 if abs(spectrum[idx][0] + 2-spectrum[idx_i][0]) < mass_tolerance:
#                     i_2_idx = idx_i
# #                     sum_inten_i_2 += spectrum[idx_i][1]
# #                     complementary_peak_i_2 = complimentary_peak + 2
#                 #neutral loss nh3
#                 if abs(spectrum[idx][0] - 17.031 -spectrum[idx_i][0]) < mass_tolerance:
#                     nh3_idx = idx_i
# #                     sum_inten_nh3 += spectrum[idx_i][1]
# #                     complementary_peak_nh3 = complimentary_peak - 17.031
#                 #neutral loss h2o    
#                 if abs(spectrum[idx][0] - 18.015 -spectrum[idx_i][0]) < mass_tolerance:
#                     h2o_idx = idx_i
# #                     sum_inten_h2o += spectrum[idx_i][1]
# #                     complementary_peak_h2o = complimentary_peak - 18.015
#             if complimentary_peak > 0:
#                 for idx2 in range(len(spectrum)):
#                     if abs(spectrum[idx2][0]-complimentary_peak) < mass_tolerance:
#                         sum_inten += spectrum[idx2][1]
#                     elif  abs(spectrum[idx2][0]-complimentary_peak - 1) < mass_tolerance:
#                         sum_inten_i_1 += spectrum[idx2][1]
#                     elif  abs(spectrum[idx2][0]-complimentary_peak - 2) < mass_tolerance:
#                         sum_inten_i_2 += spectrum[idx2][1]
#                     elif  abs(spectrum[idx2][0]-complimentary_peak + 17.031 ) < mass_tolerance:
#                         sum_inten_nh3 += spectrum[idx2][1]
#                     elif  abs(spectrum[idx2][0]-complimentary_peak + 18.015) < mass_tolerance:
#                         sum_inten_h2o += spectrum[idx2][1]
#                 if sum_inten > 0:
#                     new_spectrum[idx][1] = spectrum[idx][1] + sum_inten
#                 else:
#                     new_spectrum.append([complimentary_peak, spectrum[idx][1]])
                    
#                 if sum_inten_i_1 > 0:
#                     new_spectrum[i_1_idx][1] = spectrum[i_1_idx][1] + sum_inten_i_1
#                 elif i_1_idx != 9999:
#                     new_spectrum.append([complimentary_peak+1, spectrum[i_1_idx][1]]) 
                    
#                 if sum_inten_i_2 > 0:
#                     new_spectrum[i_2_idx][1] = spectrum[i_2_idx][1] + sum_inten_i_2
#                 elif i_2_idx != 9999:
#                     new_spectrum.append([complimentary_peak+2, spectrum[i_2_idx][1]]) 
                    
#                 if sum_inten_nh3 > 0:
#                     new_spectrum[nh3_idx][1] = spectrum[nh3_idx][1] + sum_inten_nh3
#                 elif nh3_idx != 9999:
#                     new_spectrum.append([complimentary_peak - 17.031, spectrum[nh3_idx][1]]) 
                    
#                 if sum_inten_h2o > 0:
#                     new_spectrum[h2o_idx][1] = spectrum[h2o_idx][1] + sum_inten_h2o
#                 elif h2o_idx != 9999:
#                     new_spectrum.append([complimentary_peak - 18.015, spectrum[h2o_idx][1]]) 

                
            
#             # complimentary_peak = ()
#             # complimentary_peak = precursor_mass - spectrum[idx][0] + 1.00784*2
#         # complimentary_peak = precursor_mass - spectrum[idx][0] + 1.00784*2
#             # print(f"peak = {spectrum[idx][0]}, complementary ={complimentary_peak}, precursor_mass ={precursor_mass}, precursor_cal = {precursor_cal}, peak_charge = {spectrum[idx][2].charge}, annotation = {spectrum[idx][2]}")
        
#     # exit()
# #     logging.debug(new_spectrum)
# #     for i,[mz, inten] in enumerate(new_spectrum):
# #         logging.debug(f"idx = {i}, mz = {mz}, intensity = {inten}")
# #     for i,[mz, inten,_] in enumerate(spectrum):
# #         logging.debug(f"idx = {i}, mz = {mz}, intensity = {inten}")
    
    
#     new_spectrum.sort()
#     # print(new_spectrum)
#     return new_spectrum

# cnt = 0
# def get_complimentary_spectrum_annotated1(spectrum, precursor_mass, precursor_charge):
    
#     new_spectrum = copy.deepcopy(spectrum)
#     new_spectrum = [[mz, inten] for [mz, inten, _] in new_spectrum]
#     mass_tolerance = 0.04
    
#     for idx in range(len(spectrum)):
#         complimentary_peak = 0
       
#         if spectrum[idx][2]:
#             peak_charge = spectrum[idx][2].charge
#             if peak_charge > 1:
#                 continue
#             precursor_cal = precursor_mass*precursor_charge - precursor_charge*1.00784
#             ion_cal = spectrum[idx][0]*peak_charge - peak_charge*1.00784
#             complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
            
#             sum_inten = 0
#             if complimentary_peak > 0:
#                 for idx2 in range(len(spectrum)):
#                     if abs(spectrum[idx2][0]-complimentary_peak) < mass_tolerance:
#                         sum_inten += spectrum[idx2][1]
                
#                 multiplier = complimentary_peak/2400 + 1/12    
                
#                 multiplier2 = complimentary_peak/3750 + 2/25
                
#                 if sum_inten > 0:
#                     new_spectrum[idx][1] = spectrum[idx][1] + sum_inten
#                     isotope1 = sum_inten*multiplier
#                     if complimentary_peak >500:
#                         new_spectrum.append([complimentary_peak+1, isotope1])
#                     if complimentary_peak > 800:
#                         new_spectrum.append([complimentary_peak+2, isotope1*multiplier2])
#                     new_spectrum.append([complimentary_peak-17.031, sum_inten*0.3])
#                     new_spectrum.append([complimentary_peak-18.015, sum_inten*0.3])
                    
#                 else:
#                     new_spectrum.append([complimentary_peak, spectrum[idx][1]])
#                     isotope1_1 = spectrum[idx][1]*multiplier
#                     if complimentary_peak > 500:
                        
#                         new_spectrum.append([complimentary_peak+1, isotope1_1])
#                     if complimentary_peak > 800:
#                         new_spectrum.append([complimentary_peak+2, isotope1_1*multiplier2])
#                     new_spectrum.append([complimentary_peak-17.031, spectrum[idx][1]*0.3])
#                     new_spectrum.append([complimentary_peak-18.015, spectrum[idx][1]*0.3])
                    
                       
#             # complimentary_peak = ()
#             # complimentary_peak = precursor_mass - spectrum[idx][0] + 1.00784*2
#         # complimentary_peak = precursor_mass - spectrum[idx][0] + 1.00784*2
#             # print(f"peak = {spectrum[idx][0]}, complementary ={complimentary_peak}, precursor_mass ={precursor_mass}, precursor_cal = {precursor_cal}, peak_charge = {spectrum[idx][2].charge}, annotation = {spectrum[idx][2]}")
        
#     # exit()
# #     logging.debug(new_spectrum)
# #     for i,[mz, inten] in enumerate(new_spectrum):
# #         logging.debug(f"idx = {i}, mz = {mz}, intensity = {inten}")
# #     for i,[mz, inten,_] in enumerate(spectrum):
# #         logging.debug(f"idx = {i}, mz = {mz}, intensity = {inten}")
    
    
#     new_spectrum.sort()
#     # print(new_spectrum)
#     return new_spectrum

# def get_complimentary_spectrum_annotated(spectrum):
    
# #     new_spectrum = copy.deepcopy(spectrum)
# #     new_spectrum = [[mz, inten] for [mz, inten, _] in new_spectrum]
#     mass_tolerance = 0.04
#     new_spectrum = []
    
#     for idx in range(len(spectrum.mz)):
#         complimentary_peak = 0
        
#         if spectrum.annotation[idx]:
#             peak_charge = spectrum.annotation[idx].charge
#             if peak_charge > 1:
#                 new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
#                 continue
#             precursor_cal = spectrum.precursor_mz*spectrum.precursor_charge - spectrum.precursor_charge*1.00784
#             ion_cal = spectrum.mz[idx]*peak_charge - peak_charge*1.00784
#             complimentary_peak = (precursor_cal - ion_cal + peak_charge*1.00784)/peak_charge
            
#             sum_inten = 0
#             if complimentary_peak > 0:
#                 for idx2 in range(len(spectrum.mz)):
#                     if abs(spectrum.mz[idx2]-complimentary_peak) < mass_tolerance:
#                         sum_inten += spectrum.mz[idx2]
                
# #                 multiplier = 0.00027125*complimentary_peak + 0.0817923096774193
                
# #                 multiplier2 = (0.000180833333333333*complimentary_peak*complimentary_peak + 0.163421198156682*complimentary_peak) / (complimentary_peak + 301.23723162522)
                
#                 if sum_inten > 0:
#                     new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx] + sum_inten])
# #                     isotope1 = sum_inten*multiplier
# #                     if complimentary_peak >300:
# #                         new_spectrum.append([complimentary_peak+1, isotope1])
# #                     if complimentary_peak > 500:
# #                         new_spectrum.append([complimentary_peak+2, isotope1*multiplier2])
# #                     new_spectrum.append([complimentary_peak-17.031, sum_inten*0.3])
# #                     new_spectrum.append([complimentary_peak-18.015, sum_inten*0.3])
                    
#                 else:
#                     new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
#                     new_spectrum.append([complimentary_peak, spectrum.intensity[idx]])
# #                     isotope1_1 = spectrum.intensity[idx]*multiplier
# #                     if complimentary_peak > 300:
                        
# #                         new_spectrum.append([complimentary_peak+1, isotope1_1])
# #                     if complimentary_peak > 500:
# #                         new_spectrum.append([complimentary_peak+2, isotope1_1*multiplier2])
# #                     new_spectrum.append([complimentary_peak-17.031, spectrum.intensity[idx]*0.3])
# #                     new_spectrum.append([complimentary_peak-18.015, spectrum.intensity[idx]*0.3])
                    
#         else:
#             new_spectrum.append([spectrum.mz[idx], spectrum.intensity[idx]])
                
#             # complimentary_peak = ()
#             # complimentary_peak = precursor_mass - spectrum[idx][0] + 1.00784*2
#         # complimentary_peak = precursor_mass - spectrum[idx][0] + 1.00784*2
#             # print(f"peak = {spectrum[idx][0]}, complementary ={complimentary_peak}, precursor_mass ={precursor_mass}, precursor_cal = {precursor_cal}, peak_charge = {spectrum[idx][2].charge}, annotation = {spectrum[idx][2]}")
        
#     # exit()
# #     logging.debug(new_spectrum)
# #     for i,[mz, inten] in enumerate(new_spectrum):
# #         logging.debug(f"idx = {i}, mz = {mz}, intensity = {inten}")
# #     for i,[mz, inten,_] in enumerate(spectrum):
# #         logging.debug(f"idx = {i}, mz = {mz}, intensity = {inten}")
    
    
#     new_spectrum.sort()
#     # print(new_spectrum)
#     return new_spectrum


def spectrum_to_vector_com_lib(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, is_lib = True,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """

#     try:
#         temp = spectrum.annotation
#         spectrum = get_complimentary_spectrum_annotated(spectrum)

#     except:
#         spectrum = get_complimentary_spectrum(spectrum)

#     spectrum = get_complimentary_spectrum_annotated_half(spectrum)
    spectrum_temp = spectrum
    spectrum = get_complimentary_spectrum_half(spectrum)
#     spectrum = get_complimentary_spectrum_lib(spectrum)
    
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')

    for mz, intensity in spectrum:
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity

    if is_lib:
        # 아미노산의 분자량 정보 (Da)
        seq = spectrum_temp.peptide
        seq = seq.replace("C[160]","1")
        seq = seq.replace("N[115]","2")
        seq = seq.replace("n[43]","3")
        seq = seq.replace("n[44]","4")
        seq = seq.replace("M[147]","5")
        seq = seq.replace("Q[129]","6")
#         seq = seq.replace("C[160]","C")

#         seq = seq.replace(".","")
#         seq = seq.replace("+","")
#         seq = seq.replace("-","")
#         seq = seq.replace("[","")
#         seq = seq.replace("]","")
#         seq = seq.replace("n","")
#         seq = seq.replace("c","")
#         seq = re.sub("[0-9]", "", seq)
        mz, intensity = generate_theoretical_spectrum(seq)

        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in zip(mz, intensity):
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
    else:
        spectrum = get_spectrum_query_seq_com(spectrum_temp)
        vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
        
        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in spectrum:
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
#         vector_seq = vector.copy()
#         vector_seq = np.zeros(800)
    
    if norm:
        vector /= np.linalg.norm(vector)
        vector_seq /= np.linalg.norm(vector_seq)
        
    vector = np.concatenate((vector_seq, vector))
    
    return vector

def spectrum_to_vector_com_lib_weight(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, is_lib = True,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """

#     try:
#         temp = spectrum.annotation
#         spectrum = get_complimentary_spectrum_annotated(spectrum)

#     except:
#         spectrum = get_complimentary_spectrum(spectrum)

#     spectrum = get_complimentary_spectrum_annotated_half(spectrum)
    spectrum_temp = spectrum
    spectrum = get_complimentary_spectrum_weight(spectrum)
#     spectrum = get_complimentary_spectrum_lib(spectrum)
    
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')

    for mz, intensity in spectrum:
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity

    if is_lib:
        # 아미노산의 분자량 정보 (Da)
        seq = spectrum_temp.peptide
        seq = seq.replace("C[160]","1")
        seq = seq.replace("N[115]","2")
        seq = seq.replace("n[43]","3")
        seq = seq.replace("n[44]","4")
        seq = seq.replace("M[147]","5")
        seq = seq.replace("Q[129]","6")
#         seq = seq.replace("C[160]","C")

#         seq = seq.replace(".","")
#         seq = seq.replace("+","")
#         seq = seq.replace("-","")
#         seq = seq.replace("[","")
#         seq = seq.replace("]","")
#         seq = seq.replace("n","")
#         seq = seq.replace("c","")
#         seq = re.sub("[0-9]", "", seq)
        mz, intensity = generate_theoretical_spectrum(seq)

        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in zip(mz, intensity):
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
    else:
        spectrum = get_spectrum_query_seq_com(spectrum_temp)
        vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
        
        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in spectrum:
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
#         vector_seq = vector.copy()
#         vector_seq = np.zeros(800)
    
    if norm:
        vector /= np.linalg.norm(vector)
        vector_seq /= np.linalg.norm(vector_seq)
        
    vector = np.concatenate((vector_seq, vector))
    
    return vector

def spectrum_to_vector_com(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, is_lib = True,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """

#     try:
#         temp = spectrum.annotation
#         spectrum = get_complimentary_spectrum_annotated(spectrum)

#     except:
#         spectrum = get_complimentary_spectrum(spectrum)
#     with open("9_1_complementary.txt", "a") as f:
#         f.write("BEGIN IONS\n")
#         for a,b in zip(spectrum.mz, spectrum.intensity):
#             f.write(f"{a},{b}\n")
#         f.write("END IONS\n")
    spectrum_temp = spectrum
    spectrum = get_complimentary_spectrum_half(spectrum)
    
#     with open("9_1_complementary.txt", "a") as f:
#         f.write("BEGIN IONS\n")
#         for a,b in spectrum:
#             f.write(f"{a},{b}\n")
#         f.write("END IONS\n")
    
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')

    for mz, intensity in spectrum:
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity
    if is_lib:
        # 아미노산의 분자량 정보 (Da)
        seq = spectrum_temp.peptide
        seq = seq.replace("C[160]","1")
        seq = seq.replace("N[115]","2")
        seq = seq.replace("n[43]","3")
        seq = seq.replace("n[44]","4")
        seq = seq.replace("M[147]","5")
        seq = seq.replace("Q[129]","6")
        mz, intensity = generate_theoretical_spectrum(seq)

        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in zip(mz, intensity):
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
    else:
        spectrum = get_spectrum_query_seq_com(spectrum_temp)
        vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
        
        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in spectrum:
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
#         vector_seq = vector.copy()
#         vector_seq = np.zeros(800)
    
    if norm:
        vector /= np.linalg.norm(vector)
        vector_seq /= np.linalg.norm(vector_seq)
        
    vector = np.concatenate((vector_seq, vector))
    
    return vector


def spectrum_to_vector_weight(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, is_lib = True,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """

#     try:
#         temp = spectrum.annotation
#         spectrum = get_complimentary_spectrum_annotated(spectrum)

#     except:
#         spectrum = get_complimentary_spectrum(spectrum)
#     with open("9_1_complementary.txt", "a") as f:
#         f.write("BEGIN IONS\n")
#         for a,b in zip(spectrum.mz, spectrum.intensity):
#             f.write(f"{a},{b}\n")
#         f.write("END IONS\n")
    spectrum_temp = spectrum
    spectrum = get_spectrum_weight(spectrum)
    
#     with open("9_1_complementary.txt", "a") as f:
#         f.write("BEGIN IONS\n")
#         for a,b in spectrum:
#             f.write(f"{a},{b}\n")
#         f.write("END IONS\n")
    
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')

    for mz, intensity in spectrum:
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity
    
    if is_lib:
        # 아미노산의 분자량 정보 (Da)
        seq = spectrum_temp.peptide
        seq = seq.replace("C[160]","1")
        seq = seq.replace("N[115]","2")
        seq = seq.replace("n[43]","3")
        seq = seq.replace("n[44]","4")
        seq = seq.replace("M[147]","5")
        seq = seq.replace("Q[129]","6")
        
#         seq = seq.replace("C[160]","C")

#         seq = seq.replace(".","")
#         seq = seq.replace("+","")
#         seq = seq.replace("-","")
#         seq = seq.replace("[","")
#         seq = seq.replace("]","")
#         seq = seq.replace("n","")
#         seq = seq.replace("c","")
#         seq = re.sub("[0-9]", "", seq)
        mz, intensity = generate_theoretical_spectrum(seq)

        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in zip(mz, intensity):
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
    else:
        spectrum = get_spectrum_query_seq(spectrum_temp)
        vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
        
        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in spectrum:
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
#         vector_seq = vector.copy()
#         vector_seq = np.zeros(800)
    
    if norm:
        vector /= np.linalg.norm(vector)
        vector_seq /= np.linalg.norm(vector_seq)
        
    vector = np.concatenate((vector_seq, vector))
        
    return vector


#cluster method 바꿈
def spectrum_to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, is_lib = True, spectrum_seq = None,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')
    
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity
    
    
    if is_lib:
#         vector_seq = np.zeros((hash_len,), np.float32)
#         for mz, intensity in spectrum_seq:
#             bin_idx = math.floor((mz - min_bound) // bin_size)
#             if hash_len is not None:
#                 bin_idx = hash_idx(bin_idx, hash_len)
#             vector_seq[bin_idx] += intensity
        # 아미노산의 분자량 정보 (Da)
        seq = spectrum.peptide
        seq = seq.replace("C[160]","1")
        seq = seq.replace("N[115]","2")
        seq = seq.replace("n[43]","3")
        seq = seq.replace("n[44]","4")
        seq = seq.replace("M[147]","5")
        seq = seq.replace("Q[129]","6")
#         seq = seq.replace("C[160]","C")

#         seq = seq.replace(".","")
#         seq = seq.replace("+","")
#         seq = seq.replace("-","")
#         seq = seq.replace("[","")
#         seq = seq.replace("]","")
#         seq = seq.replace("n","")
#         seq = seq.replace("c","")
#         seq = re.sub("[0-9]", "", seq)
        mz1, intensity1 = generate_theoretical_spectrum(seq)

        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in zip(mz1, intensity1):
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
            
        
    else:
        spectrum = get_spectrum_query_seq(spectrum)
        vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
        
        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in spectrum:
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
#         vector_seq = vector.copy()
#         vector_seq = np.zeros(800)
    
    if norm:
        vector /= np.linalg.norm(vector)
        vector_seq /= np.linalg.norm(vector_seq)
        
    vector = np.concatenate((vector_seq, vector))
    
    return vector

#cluster method 바꿈
def spectrum_to_vector_query_seq(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, is_lib = True, spectrum_seq = None,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')
    
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity
    
    
    if is_lib:
#         vector_seq = np.zeros((hash_len,), np.float32)
#         for mz, intensity in spectrum_seq:
#             bin_idx = math.floor((mz - min_bound) // bin_size)
#             if hash_len is not None:
#                 bin_idx = hash_idx(bin_idx, hash_len)
#             vector_seq[bin_idx] += intensity
        # 아미노산의 분자량 정보 (Da)
        seq = spectrum.peptide
        seq = seq.replace("C[160]","1")
        seq = seq.replace("N[115]","2")
        seq = seq.replace("n[43]","3")
        seq = seq.replace("n[44]","4")
        seq = seq.replace("M[147]","5")
        seq = seq.replace("Q[129]","6")
#         seq = seq.replace("C[160]","C")

#         seq = seq.replace(".","")
#         seq = seq.replace("+","")
#         seq = seq.replace("-","")
#         seq = seq.replace("[","")
#         seq = seq.replace("]","")
#         seq = seq.replace("n","")
#         seq = seq.replace("c","")
#         seq = re.sub("[0-9]", "", seq)
        mz, intensity = generate_theoretical_spectrum(seq)

        vector_seq = np.zeros((hash_len,), np.float32)
        for mz, intensity in zip(mz, intensity):
            bin_idx = math.floor((mz - min_bound) // bin_size)
            if hash_len is not None:
                bin_idx = hash_idx(bin_idx, hash_len)
            vector_seq[bin_idx] += intensity
            
        
    else:
        vector_seq = vector.copy()
#         vector_seq = np.zeros(800)
    
    if norm:
        vector /= np.linalg.norm(vector)
        vector_seq /= np.linalg.norm(vector_seq)
        
    vector = np.concatenate((vector_seq, vector))
    
    return vector


class SpectrumSpectrumMatch:

    def __init__(self, query_spectrum: MsmsSpectrum,
                 library_spectrum: MsmsSpectrum = None,
                 search_engine_score: float = math.nan,
                 q: float = math.nan,
                 num_candidates: int = 0):
        self.query_spectrum = query_spectrum
        self.library_spectrum = library_spectrum
        self.search_engine_score = search_engine_score
        self.q = q
        self.num_candidates = num_candidates

    @property
    def sequence(self):
        return (self.library_spectrum.peptide
                if self.library_spectrum is not None else None)

    @property
    def query_identifier(self):
        return self.query_spectrum.identifier

    @property
    def query_index(self):
        return self.query_spectrum.index
    
    @property
    def query_std(self):
        return self.query_spectrum.std

    @property
    def library_identifier(self):
        return (self.library_spectrum.identifier
                if self.library_spectrum is not None else None)

    @property
    def retention_time(self):
        return self.query_spectrum.retention_time

    @property
    def charge(self):
        return self.query_spectrum.precursor_charge

    @property
    def exp_mass_to_charge(self):
        return self.query_spectrum.precursor_mz

    @property
    def calc_mass_to_charge(self):
        return (self.library_spectrum.precursor_mz
                if self.library_spectrum is not None else None)

    @property
    def is_decoy(self):
        return (self.library_spectrum.is_decoy
                if self.library_spectrum is not None else None)
