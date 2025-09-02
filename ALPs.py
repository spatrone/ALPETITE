#####################################

#Preliminaries

#####################################

from platform import python_version
print("Python version: ", python_version())

import numpy as np
print("Numpy version: ", np.__version__)

import vegas
print("Vegas version: ", vegas.__version__)

import os
current_path = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(current_path)

PETITE_home_dir= parent_dir.split('examples')[0]

print("PETITE home directory:", PETITE_home_dir)
# folder where VEGAS dictionaries are stored
# dictionary_dir = "data/VEGAS_dictionaries/"
dictionary_dir = "/data_400GeV_default/"

from numpy.random import random
from PETITE.dark_shower import *
from PETITE.shower import *
import pickle as pk
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
from datetime import datetime
import random
import scipy.integrate as integrate
from scipy.stats import loguniform
import sys
from scipy.special import lambertw
from scipy.optimize import brentq, minimize_scalar
import time
from scipy.optimize import fsolve
from PETITE.physical_constants import alpha_em, m_electron, GeV, m_proton
import warnings
from scipy.interpolate import LinearNDInterpolator
from itertools import cycle
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, MaxNLocator
import cProfile
profile = cProfile.Profile()
import pstats
import json
import pandas as pd
from labellines import labelLines
import matplotlib.patheffects as pe

# Helper class for pretty-printing the verification output
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
font0 = FontProperties()
font = font0.copy()
font.set_size(24)
font.set_family('serif')
labelfont=font0.copy()
labelfont.set_size(20)
labelfont.set_weight('bold')
legfont=font0.copy()
legfont.set_size(18)
legfont.set_weight('bold')

plt.rcParams['xtick.labelsize'] = 13  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 13   # Font size for y-axis tick labels
plt.rcParams["axes.labelsize"] = 15  # Label axes - Set the desired font size
plt.rcParams["axes.titlesize"] = 14   # title - Set the desired font size
plt.rcParams['legend.title_fontsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Configure Matplotlib to use LaTeX for all text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Or "Times New Roman"
})


#####################################

#AUXILIARIES

#####################################


def set_size(w,h, ax=None):
    """ Helper function to set figure size.
        Input:
            w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def checkandsave(filename,stuff,overwrite=False):
    # Check if the file already exists
    if not overwrite:
        if os.path.exists(filename):
            # If the file exists, generate a new filename
            i = 1
            while True:
                new_filename = f'{os.path.splitext(filename)[0]}_{i}{os.path.splitext(filename)[1]}'  # Append _1, _2, etc.
                if not os.path.exists(new_filename):
                    break
                i += 1
    else:
        new_filename = filename

    # Open the file with the new filename for writing
    with open(new_filename, 'wb') as f:
        pickle.dump(stuff, f)
    print(f"Data saved on {new_filename}!")
    return None


def invGeV_to_cm(invGev):
    return 0.197*1e-13*invGev

def cm_to_invGeV(cm):
    return cm/(0.197*1e-13)

def invGeV_to_sec(invGev):
    return 0.197*1e-23/3*invGev

def LorentzNorm(p):
    return np.sqrt(p[0]**2-p[1]**2-p[2]**2-p[3]**2)

def GammaDecay_yy(ma,gayy):
    #[ma]=GeV, [gayy]=GeV**-1
    return gayy**2 * ma**3 / (64*np.pi)

def GammaDecay_ee(ma,gaee):
    #[ma]=GeV, [gaee]=1
    if ma>m_electron:
        return gaee**2 * 4 * np.pi * alpha_em * ma/(8*np.pi) * np.sqrt(1-4*m_electron**2/ma**2)
    else:
        return 1e-10

def decay_weight(w_LLP_gone, gayy, l0, Lpipe):
    w_LLP=w_LLP_gone * gayy**2
    w_tilde=w_LLP*l0/Lpipe
    return -np.exp(-w_tilde)*np.expm1(-w_LLP)


def axions_exp_weights(axions, params, is_electron_based):
    angle_accept, Lpipe, Ecut, gayy_coeff, gaee_coeff = params['angle_accept'], params['Lpipe'], params['Ecut'], params['gayy_coeff'], params['gaee_coeff']
    for axion in axions:
        axion_p = axion['p'][-3:]
        omega = axion['p'][0]
        ma=axion['m_a']
        # Angle weight
        if np.arccos(axion_p[2] / np.sqrt(axion_p @ axion_p)) < angle_accept:
            angle_weight = 1
        else:
            angle_weight = 0

        # Energy cut weight
        if omega < Ecut:
            w_energycut = 0
        else:
            w_energycut = 1

        # Decay weight for LLP
        betagamma = np.sqrt(omega**2 - ma**2) / ma
        
        decay_length_electrons= invGeV_to_cm(betagamma / GammaDecay_ee(ma, 1))
        decay_weight_LLP_electrons = gaee_coeff **2 * Lpipe * 100 / decay_length_electrons
        
        decay_length_gammas = invGeV_to_cm(betagamma / GammaDecay_yy(ma, 1))
        decay_weight_LLP_gammas = gayy_coeff **2 * Lpipe * 100 / decay_length_gammas

        decay_weight_LLP= decay_weight_LLP_electrons + decay_weight_LLP_gammas 
        
        axion['w_angle'] = angle_weight
        axion['w_LLP_gayy_one'] = decay_weight_LLP_gammas
        axion['w_LLP_gaee_one'] = decay_weight_LLP_electrons
        axion['w_LLP_gone'] = decay_weight_LLP
        
        #Old prescription: only diphoton decay
        #axion['w_LLP_gone'] = decay_weight_LLP_gammas
        axion['w_Ecut'] = w_energycut
        
        #scaling w_prod by coefficient of the coupling
        if is_electron_based:
            axion['w_prod_scaled'] = axion['w_prod'] * gaee_coeff ** 2
        else:
            axion['w_prod_scaled'] = axion['w_prod'] * gayy_coeff ** 2
        
    return axions


#####################################
#PRIMAKOFF
#####################################

#################
# BORN Montecarlo
      
def axion_gen_Q2_3mom_born(omega, ma):
    """
    The ultra-efficient log-space sampler, modified to return the accepted Q^2 value
    along with the 3-momentum for validation purposes.
    """
    if omega <= ma:
        return None, None

    # --- 1. Setup Kinematics ---
    if ma / omega < 1e-3:
        Q2_min = (ma**4) / (4 * omega**2)
        Q2_max = 4 * omega**2
    else:
        pa_magnitude_stable = np.sqrt(omega**2 - ma**2)
        Q2_min = 2*omega**2 - ma**2 - 2*omega*pa_magnitude_stable
        Q2_max = 2*omega**2 - ma**2 + 2*omega*pa_magnitude_stable
    if Q2_min <= 0: Q2_min = (ma**4) / (4 * omega**2)

    z_min, z_max = np.log(Q2_min), np.log(Q2_max)

    # --- 2. Define Target Function and Find Envelope ---
    def Y_func(Q2):
        if Q2 <= 0: return -np.inf
        return -(Q2**2 + 2 * Q2 * (ma**2 - 2*omega**2) + ma**4) / Q2
        
    Q2_peak = ma**2
    Q2_to_check = [Q2_min, Q2_max]
    if Q2_min < Q2_peak < Q2_max:
        Q2_to_check.append(Q2_peak)
    M_envelope = max(Y_func(q) for q in Q2_to_check)
    
    # --- 3. The Accept-Reject Loop ---
    while True:
        z_candidate = np.random.uniform(z_min, z_max)        
        Q2_candidate = np.exp(z_candidate)
        acceptance_prob = Y_func(Q2_candidate) / M_envelope
        if np.random.uniform(0, 1) <= acceptance_prob:
            Q2_sample = Q2_candidate
            break

    # --- 4. Kinematics Calculation ---
    pa_magnitude = np.sqrt(omega**2 - ma**2)
    cos_theta = (2 * omega**2 - ma**2 - Q2_sample) / (2 * omega * pa_magnitude)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * np.random.uniform(0, 1)
    
    px = pa_magnitude * sin_theta * np.cos(phi)
    py = pa_magnitude * sin_theta * np.sin(phi)
    pz = pa_magnitude * cos_theta
    
    return Q2_sample, np.array([px, py, pz])


#################
# Sampling
def dsigmadQ2_born(ma, omega, Q2, c=1):
    #c=alpha_em*Z**2*gayy**2
    numerator = (2 * Q2 * (2 * omega**2 - ma**2) - ma**4 - Q2**2)
    denominator = 32 * Q2**2 * omega**2
    return c*numerator/denominator

def cdf_sigma_born(ma, omega, Q2):
    Cm1=ma**4/(32*omega**2)
    C1=-1/(32*omega**2)
    Clog=(1.0/8-ma**2/(16*omega**2))
    return Cm1/Q2 + C1*Q2 + Clog*np.log(Q2)

def sigma_born(ma, omega, c=1):
    #c=alpha_em*Z**2*gayy**2
    
    #for numerical stability, kinematic boundaries
    if ma / omega < 1e-3:
        Q2_min = (ma**4) / (4 * omega**2)
        Q2_max = 4 * omega**2
    else:
        pa_mag = np.sqrt(omega**2 - ma**2)
        Q2_min = 2*omega**2 - ma**2 - 2*omega*pa_mag
        Q2_max = 2*omega**2 - ma**2 + 2*omega*pa_mag
    
    return c*(cdf_sigma_born(ma, omega, Q2_max)-cdf_sigma_born(ma, omega, Q2_min))

#Form Factors
def G2el(Q2, Z, A):    
    #atomic cutoff
    a=111/(Z**(1./3)*m_electron) #GeV^-1
    
    #nuclear cutoff
    d=0.4*A**(-1./3) #GeV
    
    atomicFF=a**2*Q2/(1+a**2*Q2)
    nuclearFF=1/(1+Q2/d**2)
    
    return (atomicFF*nuclearFF)**2
    
def G2inel(Q2, Z, A):
    #atomic cutoff
    aprime=773/(Z**(2./3)*m_electron) #GeV^-1
    
    #nuclear cutoffs
    mup=2.79
    mup2m1overmp2=(mup**2-1)/m_proton**2
    LambdaH=0.84 #GeV
    
    atomicFF=aprime**2*Q2/(1+aprime**2*Q2)
    nuclearFF=(1+Q2*mup2m1overmp2/4)/(1+Q2/LambdaH**2)**4
    #the nuclear form factor is not squared, see footnote 4 in https://arxiv.org/pdf/2006.09419 (Celentano_2020)
    return (atomicFF)**2*nuclearFF
 
def G2tot(Q2, Z, A, inelastic_on):
    G2tot = G2el(Q2, Z, A)
    if inelastic_on:
        G2tot+= G2inel(Q2, Z, A)/Z
    return G2tot

def Nsigma_born(gayy, ndensity, omega, ma, Z, A):
    return alpha_em * (invGeV_to_cm(gayy))**2 * Z**2 * ndensity * sigma_born(ma, omega)

def Nsigma_FF(gayy, ndensity, omega, ma, Z, A, Q2, inelastic_on):
    return alpha_em * (invGeV_to_cm(gayy))**2 * Z**2 * ndensity * G2tot(Q2, Z, A, inelastic_on) * sigma_born(ma, omega)

def photons_from_beam(pbeam, shower, plothisto=False):
    """
    Generates shower photons and translates them into a list of simple,
    self-contained dictionaries for further processing.
    """
    custom_photons = []
    n_total_photons = 0
    start_time = time.time()
    omega_min=shower.min_energy
    N_primaries = len(pbeam) # The number of primaries for THIS batch

    for i, primary_particle in enumerate(pbeam):
        # Generate the shower from one primary particle
        shower_particles = shower.generate_shower(primary_particle)
        
        # --- The Translation Step ---
        for p in shower_particles:
            if p.get_pid() == 22 and p.get_p0()[0] > omega_min:
                # Create the custom dictionary for each valid photon
                photon_dict = {
                    'p': p.get_p0(),
                    'weight': p.get_ids().get('weight', 1.0), # Use .get for safety
                    'rotation_matrix': p.rotation_matrix(),
                    'N_primaries': N_primaries # Embed the batch history
                }
                custom_photons.append(photon_dict)

        # Progress bar logic
        n_total_photons = len(custom_photons)
        elapsed_time = time.time() - start_time
        avg_time_per_shower = elapsed_time / (i + 1)
        eta = avg_time_per_shower * (N_primaries - (i + 1))
        progress = f"Generated {i+1}/{N_primaries} showers ({n_total_photons} photons > {omega_min} GeV) | ETA: {eta:.2f}s"
        sys.stdout.write(f'\r{progress}\033[K')
        sys.stdout.flush()
    print()
    
    # Plotting logic remains the same, but uses dictionary access
    if plothisto and custom_photons:
        omega = [photon['p'][0] for photon in custom_photons]
        # Set log-spaced bins for the histogram
        min_energy = min(omega) if omega else 1e-3  # Use a small value if no photons
        max_energy = max(omega) if omega else 1e3   # Use a large value if no photons
        bins = np.logspace(np.log10(min_energy), np.log10(max_energy), 50)
        
        plt.figure(figsize=(8, 6))
        plt.hist(omega, bins=bins, color='blue', alpha=0.7)
        plt.xscale('log')  # Set x-axis to log scale
        plt.title(f'Photon Energy Distribution (E > {omega_min} GeV)')
        plt.xlabel('Photon Energy (GeV)')
        plt.ylabel('Counts')
        plt.show()
    
    return custom_photons
    
def count_photons_above_ma(photons,ma):
    photons_above_ma=0
    for photon in photons:
        if  photon.get_p0()[0]>ma: photons_above_ma+=1
    return photons_above_ma
        
    
def convert_phot_to_axions_primakoff(shower, photons, ma, params, return_stats=False):
    """
    Converts a list of custom photon dictionaries into a list of axion dictionaries
    via the Primakoff effect. This version is decoupled from the Particle object.

    Args:
        shower: The shower object, needed for final weight calculation.
        photons (list of dict): A list of custom photon dictionaries.
        ma (float): The axion mass.
        params (dict): Physics parameters.
        return_stats (bool): If True, returns a dictionary of statistics alongside the axions.

    Returns:
        A list of final axion dictionaries, or a tuple (axions, stats) if return_stats is True.
    """

    # Pre-filtering
    photons_to_process = [p for p in photons if p['p'][0] > ma]
    n_photons_to_process = len(photons_to_process)
    
    pre_axions = [] # A temporary list to hold axions before final weighting
    start_time = time.time()

    # --- 2. Main Generation Loop ---
    for i, photon_dict in enumerate(photons_to_process):
        
        # Get photon energy from the dictionary
        omega = photon_dict['p'][0]
        
        # Sampled from BORN cross section
        Q2, p_rest_frame = axion_gen_Q2_3mom_born(omega, ma)
        
        # Reconstruct the lab-frame momentum using the stored rotation matrix
        rotation_matrix = photon_dict['rotation_matrix']
        axion_p_lab_frame = rotation_matrix @ p_rest_frame
            
        # The axion energy is the same as the photon's energy for zero nuclear recoil
        axion_4_momentum = np.array(np.hstack([omega, axion_p_lab_frame]))

        # Create a "pre-axion" dictionary. It contains all info needed for the next step.
        pre_axions.append({
            'm_a': ma,
            'p': axion_4_momentum,
            'w_photon': photon_dict['weight'],
            'Q2': Q2,
            'N_primaries': photon_dict['N_primaries']
        })

        # --- Progress Bar ---
        if (i + 1) % 100 == 0 or (i + 1) == n_photons_to_process:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n_photons_to_process - (i + 1)) if i + 1 > 0 else 0
            progress_str = (f'\rConverting {i+1}/{n_photons_to_process} photons | ETA: {eta:.2f}s')
            sys.stdout.write(f'{progress_str}\033[K')
            sys.stdout.flush()

    print() # Newline after progress bar

    # --- 3. Finalization ---
    # Finalize the production weights. It returns also the count of the invalid production weights
    final_axions, inv_prod_w_count = axions_prod_weight_primakoff(shower, pre_axions, params)

    if return_stats:
        full_stats = {
            'total_photons_in_batch': len(photons),
            'above_ma': n_photons_to_process,
            'converted': len(final_axions),
            'inv_prod_weight': inv_prod_w_count
        }
        return final_axions, full_stats

    return final_axions


def axions_prod_weight_primakoff(shower, axions, params):
    Z, A, ndensity, inelastic_on = params['Z_T'], params['A_T'], params['ndensity'], params['inelastic_on']
    cnt = 0

    for axion in axions:
        omega = axion['p'][0]
        Q2 = axion['Q2']
        ma = axion['m_a']
        primary_weight=axion['w_photon']

        # Production weight specific to Primakoff, it includes the full form factor G2tot
        prod_weight = Nsigma_FF(1, ndensity, omega, ma, Z, A, Q2, inelastic_on) / shower._NSigmaPhoton(omega)
        
        if np.isinf(prod_weight) or np.isnan(prod_weight):
            cnt += 1
            prod_weight = 0

        axion['w_prod'] = prod_weight * primary_weight
        axion['G2tot'] = G2tot(Q2, Z, A, inelastic_on)

    return axions, cnt

#####################################
#DARKVECTOR GENERATION WITH PETITE
#####################################

def vectors_from_beam(part_beam, dark_shower):
    # Generate all types of Dark Vectors given a list of initial particles
    vector_from_el_prim = []
    vector_from_el = []
    vector_from_pos = []
    vector_comp = []
    vector_ann = []
    
    N_primaries=len(part_beam)
    
    cnt_zero_brem = 0
    cnt_zero_ann=0
    cnt_zero_comp=0

    for part0 in tqdm(part_beam):
        s0SM = dark_shower.generate_shower(part0)
        s0BSM = dark_shower.generate_dark_shower(ExDir=list(s0SM))

        for V in s0BSM[1]:
            ids = V.get_ids()
            genprocess = ids["generation_process"]
            parent_pid = ids["parent_PID"]
            p = V.get_p0()
            w = ids["weight"]
            parent_E = ids["parent_E"]
            gen_number = ids["generation_number"]
            
            
            DarkVector = {
                    'p': p,
                    'weight': w,
                    'parent_pid': parent_pid,
                    'parent_E': parent_E,
                    'gen_number': gen_number,
                    'gen_process': genprocess,
                    'N_primaries': N_primaries
            }
            
            if np.isnan(np.sum(p)):
                warnings.warn("The momentum has nan entry!", UserWarning)

            if genprocess == "DarkBrem":
                kyn_vars = ids.get("kinematics_vars")
                diff_x_sec = ids.get("diff_x_sec")

                DarkVector.update({
                    'kinematics_vars': kyn_vars,
                    'diff_x_sec': diff_x_sec
                })

                if w == 0:
                    cnt_zero_brem += 1
                    continue

                if gen_number==1:
                    if parent_pid==11 and w!=0: 
                        vector_from_el_prim.append(DarkVector)
                if parent_pid==11 and w!=0: 
                    vector_from_el.append(DarkVector)
                if parent_pid==-11 and w!=0: 
                    vector_from_pos.append(DarkVector)

            elif genprocess in ["DarkComp_bound", "DarkComp"]:
                if w == 0:
                    cnt_zero_comp += 1
                    continue
                
                vector_comp.append(DarkVector)
                
            elif genprocess in ["DarkAnn_bound", "DarkAnn"]:
                if w == 0:
                    cnt_zero_comp += 1
                    continue

                vector_ann.append(DarkVector)

            else:
                print(f"Warning! The process {genprocess} is active and was not categorized!")

    print(f"\nDark Vector Generation Summary - mV = {dark_shower._mV} GeV")
    print("=" *  50)
    print(f"{'Source':<20} {'Total':>10} {'Zero Weight':>15}")
    print("-" * 50)
    print(f"{'From electrons':<20} {len(vector_from_el):>10} {cnt_zero_brem:>15}")
    print(f"{'From positrons':<20} {len(vector_from_pos):>10} {cnt_zero_brem:>15}")
    print(f"{'Annihilation':<20} {len(vector_ann):>10} {cnt_zero_ann:>15}")
    print(f"{'Compton':<20} {len(vector_comp):>10} {cnt_zero_comp:>15}")
    print("=" * 50)

    return {
        "from_el_prim": vector_from_el_prim,
        "from_el": vector_from_el,
        "from_pos": vector_from_pos,
        "from_ann": vector_ann,
        "from_comp":vector_comp
    }

#####################################

#BREM

#####################################

#################
# XSections
                                                     
def dsig_dx_dcostheta_dark_brem_exact_tree_level(x0, x1, x2, mV, params, E_inc, method = None):
    """Exact Tree-Level Dark Photon Bremsstrahlung  
       e (ep) + Z -> e (epp) + V (w) + Z
       result it dsigma/dx/dcostheta where x=E_darkphoton/E_beam and theta is angle between beam and dark photon

       Input parameters needed:
            x0, x1, x2:  kinematic parameters related to energy of emitted vector, cosine of its angle and the momentum transfer to the nucleus (precise relation depends on method see below.
            me (mass of electron)
            mV (mass of dark photon)
            Ebeam (incident electron energy)
            ZTarget (Target charge)
            ATarget (Target Atomic mass number)  
            MTarget (Target mass)
    """
    me = params['me']
    Ebeam = E_inc
    MTarget = params['mT']
    alpha_em= params['alpha_em']
    
    #SamADD_START
    GeV=1
    m_proton=938.272088*1e-3
    #SamADD_END
    
    if method is None:
        method = 'Log'
    if method == 'Log':
        x, l1mct, lttilde = x0, x1, x2
        one_minus_costheta = 10**l1mct    
        costheta = 1.0 - one_minus_costheta
        ttilde = 10**lttilde
        Jacobian = one_minus_costheta*ttilde*np.log(10.0)**2
    elif method == 'Standard':
        x, costheta, ttilde = x0, x1, x2
        Jacobian = 1.0

    # kinematic boundaries
    if x*Ebeam < mV:
        return 0.
    
    k = np.sqrt((x * Ebeam)**2 - mV**2)
    p = np.sqrt(Ebeam**2 - me**2)
    V = np.sqrt(p**2 + k**2 - 2*p*k*costheta)
    
    
    utilde = -2 * (x*Ebeam**2 - k*p*costheta) + mV**2
    
    discr = utilde**2 + 4*MTarget*utilde*((1-x)*Ebeam + MTarget) + 4*MTarget**2 * V**2
    # kinematic boundaries
    if discr < 0:
        return 0.
        
    Qplus = V * (utilde + 2*MTarget*((1-x)*Ebeam + MTarget)) + ((1-x)*Ebeam + MTarget) * np.sqrt(discr)
    Qplus = Qplus/(2*((1-x)*Ebeam + MTarget)**2-2*V**2)
    
    Qminus = V * (utilde + 2*MTarget*((1-x)*Ebeam + MTarget)) - ((1-x)*Ebeam + MTarget) * np.sqrt(discr)
    Qminus = Qminus/(2*((1-x)*Ebeam + MTarget)**2-2*V**2)
    
    Qplus = np.fabs(Qplus)
    Qminus = np.fabs(Qminus)
    
    tplus = 2*MTarget*(np.sqrt(MTarget**2 + Qplus**2) - MTarget)
    tminus = 2*MTarget*(np.sqrt(MTarget**2 + Qminus**2) - MTarget)

    # Physical region checks
    if tplus < tminus:
        return 0.
    
    tconv = (2*MTarget*(MTarget + Ebeam)*np.sqrt(Ebeam**2 + m_electron**2)/(MTarget*(MTarget+2*Ebeam) + m_electron**2))**2
    t = ttilde*tconv
    if t > tplus or t < tminus:
        return 0.
            
    q0 = -t/(2*MTarget)
    q = np.sqrt(t**2/(4*MTarget**2)+t)
    costhetaq = -(V**2 + q**2 + me**2 -(Ebeam + q0 -x*Ebeam)**2)/(2*V*q)

    # kinematic boundaries
    if np.fabs(costhetaq) > 1.:
        return 0.
    mVsq2mesq = (mV**2 + 2*me**2)
    Am2 = -8 * MTarget * (4*Ebeam**2 * MTarget - t*(2*Ebeam + MTarget)) * mVsq2mesq
    A1 = 8*MTarget**2/utilde
    Am1 = (8/utilde) * (MTarget**2 * (2*t*utilde + utilde**2 + 4*Ebeam**2 * (2*(x-1)*mVsq2mesq - t*((x-2)*x+2)) + 2*t*(-mV**2 + 2*me**2 + t)) - 2*Ebeam*MTarget*t*((1-x)*utilde + (x-2)*(mVsq2mesq + t)) + t**2*(utilde-mV**2))
    A0 = (8/utilde**2) * (MTarget**2 * (2*t*utilde + (t-4*Ebeam**2*(x-1)**2)*mVsq2mesq) + 2*Ebeam*MTarget*t*(utilde - (x-1)*mVsq2mesq))
    Y = -t + 2*q0*Ebeam - 2*q*p*(p - k*costheta)*costhetaq/V 
    W= Y**2 - 4*q**2 * p**2 * k**2 * (1 - costheta**2)*(1 - costhetaq**2)/V**2
    
    if W == 0.:
        print("x, costheta, t = ", [x, costheta, t])
        print("Y, q, p, k, costheta, costhetaq, V" ,[Y, q, p, k, costheta, costhetaq, V])
        
    # kinematic boundaries
    if W < 0:
        return 0.
    
    phi_integral = (A0 + Y*A1 + Am1/np.sqrt(W) + Y * Am2/W**1.5)/(8*MTarget**2)

    formfactor_separate_over_tsquared = Gelastic_inelastic_over_tsquared(params, t)
    
    ans = formfactor_separate_over_tsquared*np.power(alpha_em, 3) * k * Ebeam * phi_integral/(p*np.sqrt(k**2 + p**2 - 2*p*k*costheta))
    
    return(ans*tconv*Jacobian)

def dsig_dx_dcostheta_axion_brem_exact_tree_level(x0, x1, x2, ma, params, E_inc, method=None):
    """Exact Tree-Level Axion Photon Bremsstrahlung  
       e (ep) + Z -> e (epp) + a (w) + Z
       result it dsigma/dx/dcostheta where x=E_axion/E_beam and theta is angle between beam and axion

       Input parameters needed:
            x0, x1, x2:  kinematic parameters related to energy of emitted vector, cosine of its angle and the momentum transfer to the nucleus (precise relation depends on method see below.
            me (mass of electron)
            ma (mass of axion)
            Ebeam (incident electron energy)
            ZTarget (Target charge)
            ATarget (Target Atomic mass number)  
            MTarget (Target mass)
    """
    me = params['me']
    Ebeam = E_inc
    MTarget = params['mT']
    alpha_em= params['alpha_em']
    
    GeV=1
    m_proton=938.272088*1e-3

    if method is None:
        method = 'Log'
    if method == 'Log':
        x, l1mct, lttilde = x0, x1, x2
        one_minus_costheta = 10**l1mct    
        costheta = 1.0 - one_minus_costheta
        ttilde = 10**lttilde
        Jacobian = one_minus_costheta*ttilde*np.log(10.0)**2
    elif method == 'Standard':
        x, costheta, ttilde = x0, x1, x2
        Jacobian = 1.0

    # kinematic boundaries
    if x*Ebeam < ma:
        return 0.
    
    k = np.sqrt((x * Ebeam)**2 - ma**2)
    p = np.sqrt(Ebeam**2 - me**2)
    V = np.sqrt(p**2 + k**2 - 2*p*k*costheta)
    
    
    utilde = -2 * (x*Ebeam**2 - k*p*costheta) + ma**2
    
    discr = utilde**2 + 4*MTarget*utilde*((1-x)*Ebeam + MTarget) + 4*MTarget**2 * V**2
    # kinematic boundaries
    if discr < 0:
        return 0.
        
    Qplus = V * (utilde + 2*MTarget*((1-x)*Ebeam + MTarget)) + ((1-x)*Ebeam + MTarget) * np.sqrt(discr)
    Qplus = Qplus/(2*((1-x)*Ebeam + MTarget)**2-2*V**2)
    
    Qminus = V * (utilde + 2*MTarget*((1-x)*Ebeam + MTarget)) - ((1-x)*Ebeam + MTarget) * np.sqrt(discr)
    Qminus = Qminus/(2*((1-x)*Ebeam + MTarget)**2-2*V**2)
    
    Qplus = np.fabs(Qplus)
    Qminus = np.fabs(Qminus)
    
    tplus = 2*MTarget*(np.sqrt(MTarget**2 + Qplus**2) - MTarget)
    tminus = 2*MTarget*(np.sqrt(MTarget**2 + Qminus**2) - MTarget)

    # Physical region checks
    if tplus < tminus:
        return 0.
    
    tconv = (2*MTarget*(MTarget + Ebeam)*np.sqrt(Ebeam**2 + m_electron**2)/(MTarget*(MTarget+2*Ebeam) + m_electron**2))**2
    t = ttilde*tconv
    if t > tplus or t < tminus:
        return 0.
            
    q0 = -t/(2*MTarget)
    q = np.sqrt(t**2/(4*MTarget**2)+t)
    costhetaq = -(V**2 + q**2 + me**2 -(Ebeam + q0 -x*Ebeam)**2)/(2*V*q)

    # kinematic boundaries
    if np.fabs(costhetaq) > 1.:
        return 0.
    Am2 = 4 * MTarget * ma**2 * (-4 * Ebeam**2 * MTarget + 2 * Ebeam * t + MTarget * t)
    A1 = 4*MTarget**2/utilde
    Am1 = (4 * MTarget * (Ebeam**2 * (8 * MTarget * ma**2 * (x - 1) - 4 * MTarget * t * x**2) - 
                     2 * Ebeam * t * (ma**2 * (x - 2) + utilde * x) + 
                     MTarget * (2 * ma**2 * t + utilde**2))) / utilde
    A0 = (4 * MTarget * (-4 * Ebeam**2 * MTarget * ma**2 * (x - 1)**2 + 2 * Ebeam * t * (ma**2 - x * (ma**2 + utilde)) + MTarget * (ma**2 * t + 2 * utilde**2))) / utilde**2
    Y = -t + 2*q0*Ebeam - 2*q*p*(p - k*costheta)*costhetaq/V 
    W= Y**2 - 4*q**2 * p**2 * k**2 * (1 - costheta**2)*(1 - costhetaq**2)/V**2
    
    if W == 0.:
        print("x, costheta, t = ", [x, costheta, t])
        print("Y, q, p, k, costheta, costhetaq, V" ,[Y, q, p, k, costheta, costhetaq, V])
        
    # kinematic boundaries
    if W < 0:
        return 0.
    
    phi_integral = (A0 + Y*A1 + Am1/np.sqrt(W) + Y * Am2/W**1.5)/(8*MTarget**2)

    formfactor_separate_over_tsquared = Gelastic_inelastic_over_tsquared(params, t)
    
    ans = formfactor_separate_over_tsquared*np.power(alpha_em, 3) * k * Ebeam * phi_integral/(p*np.sqrt(k**2 + p**2 - 2*p*k*costheta))
    
    return(ans*tconv*Jacobian)

mu_p = 2.79  # https://journals.aps.org/prd/pdf/10.1103/PhysRevD.8.3109
def Gelastic_inelastic_over_tsquared(EI, t):
    """
    Form factor squared used for elastic/inelastic contributions to Dark Bremsstrahlung Calculation
    Rescaled by 1/t^2 to make it easier to integrate over t
    (Scales like Z^2 in the small-t limit)
    See Eq. (9) of Gninenko et al (Phys. Lett. B 782 (2018) 406-411)
    """

    Z = EI["Z_T"]
    A = EI["A_T"]
    c1 = (111 * Z ** (-1 / 3) / m_electron) ** 2
    c2 = 0.164 * GeV**2 * A ** (-2 / 3)
    Gel = (1.0 / (1.0 + c1 * t)) ** 2 * (1 + t / c2) ** (-2)
    ap2 = (773.0 * Z ** (-2.0 / 3) / m_electron) ** 2
    Ginel = (
        Z
        / ((c1**2 * Z**2))
        * np.power((ap2 / (1.0 + ap2 * t)), 2.0)
        * ((1.0 + (mu_p**2 - 1.0) * t / (4.0 * m_proton**2)) / (1.0 + t / 0.71) ** 4)
    )
    return Z**2 * c1**2 * (Gel + Ginel)

#################
# DarkVectors and Axion
                            
def convert_vec_to_axions_brem(vector_array, ma, params, return_stats=False):
    axion_array=[]
    
    for vector in vector_array:
        # 1. Automatically inherit all fields by copying the dictionary.
        axion = vector.copy()

        # 2. Extract necessary values from the new 'axion' dictionary for calculation.
        original_weight = axion['weight']
        E_inc = axion['parent_E']
        x0, x1, x2 = axion['kinematics_vars']
        diff_x_sec = axion['diff_x_sec']
        
        # 3. Perform the w_prod calculation.
        dsigma_vec = dsig_dx_dcostheta_dark_brem_exact_tree_level(x0, x1, x2, ma, params, E_inc)
        dsigma_ax = dsig_dx_dcostheta_axion_brem_exact_tree_level(x0, x1, x2, ma, params, E_inc)
        
        # --- Validation Checks ---
        if abs(dsigma_vec / diff_x_sec - 1) > 1e-8: 
            warnings.warn("The computed differential cross-section of the Dark Vector does not match the PETITE one!", UserWarning)
        if diff_x_sec == 0:
            warnings.warn("The provided differential cross-section of the Dark Vector is zero!", UserWarning)
        
        # Calculate rescaled weight, handling division by zero gracefully.
        rescaled_weight = (original_weight * dsigma_ax / diff_x_sec) if diff_x_sec != 0 else 0.0

        # --- Mass Validation ---
        p = axion['p']
        mV = np.sqrt(p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2)
        if abs(mV / ma - 1) > 1e-5: 
            warnings.warn(f"The mass of the vector (mv: {mV:.4g}) and of the axion (ma: {ma:.4g}) do not match!", UserWarning)
        
        # 4. Update the dictionary with the transformed values.
        axion['m_a'] = ma                          # Add new axion mass key
        axion['w_prod'] = rescaled_weight          # Add the new production weight
        
        # 5. Remove keys specific to the original vector that are no longer needed.
        del axion['weight']
        del axion['kinematics_vars']
        del axion['diff_x_sec']

        # All other keys ('N_primaries', 'gen_number', etc.) are already there from the copy.
        axion_array.append(axion)
    
    return axion_array
                         
                        
#####################################

#Annihilation-Compton

#####################################
                            
                            
def convert_vec_to_axions_ann_comp(vector_array, ma, params):
    axion_array=[]
    
    for vector in vector_array:
        # 1. Automatically inherit all fields by copying the dictionary.
        axion = vector.copy()

        # 2. Perform the specific calculations for this conversion.
        original_weight = axion['weight']
        rescaled_weight = original_weight * ma**2 / (2 * (2 * m_electron**2 + ma**2))
        
        # 3. Validate the momentum (using the key from the copied dict).
        p = axion['p']
        mV = LorentzNorm(p)
        if mV / ma - 1 > 1e-5: 
            warnings.warn(f"The mass of the vector (mv: {mV}) and of the axion (ma: {ma}) do not match up to one part over 10^5!", UserWarning)
        
        # 4. Update the new dictionary with transformed values.
        axion['m_a'] = ma                          # Add new axion mass key
        axion['w_prod'] = rescaled_weight          # Add the new production weight
        del axion['weight']                        # Remove the original vector weight

        # All other keys ('N_primaries', 'gen_number', etc.) are already there from the copy.
        axion_array.append(axion)
    
    return axion_array



#####################################

#SENSITIVITIES CURVES

#####################################


def gayy_LLP_approx(axions, params):
    N_gamma=params['POT']
    N_discovery=params['N_discovery']
    
    W_tot=0
    for axion in axions:
        W_tot+= axion['w_prim']*axion['w_angle']*axion['w_prod_scaled']*axion['w_LLP_gone']*axion['w_Ecut']
    return (W_tot*N_gamma/N_discovery)**(-0.25)


def gayy_approx_right(axions, params):
    """
    Provides a good initial guess for the right-hand (large g) sensitivity bound.
    
    """
    N_gamma = params['POT']
    N_discovery = params['N_discovery']
    l0 = params['l0']
    Lpipe = params['Lpipe'] # We need Lpipe for the correction
    R = N_discovery / N_gamma
    
    W_prod_tot = 0
    C_eff_num = 0
    
    # Check for division by zero if Lpipe is not set or is zero
    if Lpipe == 0:
        return None # Or handle as an error
        
    for axion in axions:
        prod_weight = axion['w_prim'] * axion['w_Ecut'] * axion['w_angle'] * axion['w_prod_scaled']
        W_prod_tot += prod_weight
        
        # The exponent C for a single axion is (l0/Lpipe) * w_LLP_gone.
        # This is dimensionless, as required.
        C_axion = (l0 / Lpipe) * axion['w_LLP_gone']
        C_eff_num += C_axion * prod_weight

    if W_prod_tot == 0:
        return None
        
    C_eff = C_eff_num / W_prod_tot
    
    Z = R / (W_prod_tot)
    arg = -C_eff * Z
    
    if arg < -1/np.e or arg >= 0:
        return None

    with np.errstate(all='ignore'):
        lambert_sol = lambertw(arg, k=-1)
    
    if np.iscomplex(lambert_sol) or np.isnan(lambert_sol):
        return None
        
    y = -lambert_sol.real
    
    g_squared = y / C_eff
    return np.sqrt(g_squared)

def find_bracket_endpoint_log(func, log_start, step, max_steps=30):
    """
    Searches for a bracketing endpoint in log-space by adding/subtracting a step.
    'func' is the function to evaluate (takes log-space input).
    'log_start' is the known point (e.g., the peak).
    'step' is the value to add/subtract from the log (e.g., +0.5 or -0.5).
    """
    y_start = func(log_start)
    log_current = log_start
    for _ in range(max_steps):
        log_current += step
        if func(log_current) * y_start < 0:
            return log_current  # Found a valid endpoint
    return None # Failed to find a bracket

def sensitivity_exact_fast(axions, params, tolerance=1e-7):
    """
    The definitive robust solver. It operates in log-space and uses an active
    search to guarantee valid brackets for brentq.
    """
    # --- Step 1 & 2: Setup and Vectorization (Same as before) ---
    N_gamma, N_discovery, l0, Lpipe = params['POT'], params['N_discovery'], params['l0'], params['Lpipe']
    R = N_discovery / N_gamma
    log_R = np.log10(R)

    w_prim = np.array([ax['w_prim'] for ax in axions])
    w_Ecut = np.array([ax['w_Ecut'] for ax in axions])
    w_angle = np.array([ax['w_angle'] for ax in axions])
    w_prod = np.array([ax['w_prod_scaled'] for ax in axions])
    w_LLP_gone = np.array([ax['w_LLP_gone'] for ax in axions])

    # --- Step 3: Define the Log-Space Function ---
    def f_log_space(log_gayy):
        gayy = np.power(10.0, log_gayy)
        decay_factors = decay_weight(w_LLP_gone, gayy, l0, Lpipe)
        W_val = (gayy)**2 * np.sum(w_prim * w_Ecut * w_angle * w_prod * decay_factors)
        if W_val <= 0: return log_R - 100
        return np.log10(W_val) - log_R

    # --- Step 4: Find the Peak  ---
    res_peak = minimize_scalar(lambda log_g: -f_log_space(log_g), bracket=(-12, 2), method='brent')
    log_g_peak = res_peak.x
    if f_log_space(log_g_peak) < 0: # Check if peak is high enough
        return [None, None], [None, None]

    # --- Step 5: Get Guesses  ---
    g_left_guess = gayy_LLP_approx(axions, params)
    g_right_guess = gayy_approx_right(axions, params)

    # --- Step 6: Find Roots with a ROBUST BRACKET SEARCH ---
    log_g_left, log_g_right = None, None

    # Find Left Root
    if g_left_guess is not None:
        # Search LEFT from the peak to find a point where f is negative
        b_left = find_bracket_endpoint_log(f_log_space, log_g_peak, step=-0.5)
        if b_left is not None:
            try: # The bracket is now guaranteed to be valid
                log_g_left = brentq(f_log_space, a=b_left, b=log_g_peak, xtol=tolerance)
            except ValueError: pass # Should never happen now

    # Find Right Root
    if g_right_guess is not None:
        # Search RIGHT from the peak to find a point where f is negative
        b_right = find_bracket_endpoint_log(f_log_space, log_g_peak, step=+0.5)
        if b_right is not None:
            try: # The bracket is now guaranteed to be valid
                log_g_right = brentq(f_log_space, a=log_g_peak, b=b_right, xtol=tolerance)
            except ValueError: pass # Should never happen now
            
    # --- Step 7: Final Conversion ---
    left_bound = np.power(10.0, log_g_left) if log_g_left is not None else None
    right_bound = np.power(10.0, log_g_right) if log_g_right is not None else None

    return [left_bound, right_bound], [g_left_guess, g_right_guess]


                            
#####################################

#CLASS

#####################################                           
                                                       
    
class AxionShower:
    
    EXPERIMENT_DEFAULTS = {
        'SHIP': {
            'Lpipe': 50.0,
            'l0': 33.5,
            'A_detector': 24.0,
            'Ecut': 0.200,
            'POT': 6e20,
            'shower_material': 'molybdenum',
            'primaries': 'photons'
        },
        'BDX': {
            'Lpipe': 3.0,
            'l0': 20.0,
            'A_detector': 0.275,  
            'Ecut': 0.300,     
            'POT': 1e22,
            'shower_material': 'aluminum',
            'primaries': 'electrons'
        }
        # Add other experiments here in the future
    }

    def __init__(self, DATA_folder, shower_material=None, experiment=None, primaries=None, petite_home_dir=PETITE_home_dir, dictionary_dir=dictionary_dir, **kwargs):
        """
        Initializes the AxionShower simulation environment.

        Args:
            DATA_folder (str): The directory to store data, relative to petite_home_dir.
            shower_material (str, optional): The material for the shower. If None, the
                default for the selected experiment will be used. This serves as an override.
            experiment (str, optional): The name of the experiment ('SHIP', 'BDX').
                Defaults to 'SHIP'. This sets the base configuration.
            petite_home_dir (str, optional): Path to the PETITE project home.
            dictionary_dir (str, optional): Path to the dictionary directory.
            **kwargs: Other specific parameter overrides (e.g., POT=1e22, Lpipe=100).
        """
        self.PETITE_home = petite_home_dir
        self.dict_dir = dictionary_dir
        self.DATA_folder_path = os.path.join(petite_home_dir, DATA_folder)
        
        self.experiment = experiment or 'SHIP'
        if self.experiment not in self.EXPERIMENT_DEFAULTS:
            raise ValueError(f"Experiment '{self.experiment}' is not defined. Available options are: {list(self.EXPERIMENT_DEFAULTS.keys())}")
        
        # --- Add shower_material to kwargs if provided ---
        if shower_material is not None:
            kwargs['shower_material'] = shower_material
        if primaries is not None:
            kwargs['primaries'] = primaries
        
        # Smart State Initialization
        self.logger = Logger(self.DATA_folder_path)
        existing_state = self.logger._load_log_state()

        if existing_state is not None:
            print(f"[State] Loaded existing state from {os.path.basename(self.logger.state_file_path)}")
            self.params_dict = existing_state
        else:
            print(f"[State] No existing state found. Initializing new state for '{self.experiment}' experiment.")
            self.params_dict = {}
            self._update_params_dict(**kwargs)
            
        self.shower_material = self.params_dict.get('shower_material')
        self.primaries = self.params_dict.get('primaries')
        
        print(self.primaries)
        
        print("\n--- Initialized AxionShower State ---")
        # Use json.dumps for pretty printing the dictionary
        print(json.dumps(self.params_dict, indent=4))
        print("-------------------------------------\n")
    
    ##############
    #UTILITIES
    ##############
    
    def _set_primaries(self, name, print_updates=False):
        """
        Convenience method to update the primary particle type.
        This correctly updates the central parameter dictionary.
        """
        print(f"--> Setting primary particle type to '{name}'...")
        self._update_params_dict(primaries=name, print_updates=print_updates)
    
    def _update_dict_dir(self,new_dict_dir):
        self.dict_dir = new_dict_dir
    
    def _update_params_dict(self, primaries=None, shower_material=None, gayy_coeff=None, gaee_coeff=None, Lpipe=None, l0=None, A_detector=None, Ecut=None, POT=None, Ndiscovery=None, shower_cutoff=None, inelastic_on=None, print_updates=False):
        """
        Updates the internal parameter dictionary based on experiment defaults and explicit overrides.
        """
        # 1. Start with the defaults for the chosen experiment
        base_params = self.EXPERIMENT_DEFAULTS[self.experiment].copy()

        # 2. Collect all explicit overrides from the function arguments.
        overrides = {
            k: v for k, v in locals().items() 
            if v is not None and k not in ['self', 'print_updates', 'base_params']
        }
        
        # 3. Merge overrides into the base parameters
        final_params = base_params
        final_params.update(overrides)

        # 4. Set defaults for non-experimental parameters if they weren't provided
        final_params.setdefault('gayy_coeff', alpha_em/np.pi)
        final_params.setdefault('gaee_coeff', 2*m_electron/np.sqrt(4*np.pi*alpha_em))
        final_params.setdefault('shower_cutoff', 0.05)
        final_params.setdefault('inelastic_on', True)
        final_params.setdefault('N_discovery', 5)

        # --- Use the finalized material to initialize Shower and get properties ---
        # This must be done *after* the final material has been determined.
        current_material = final_params['shower_material']
        s = Shower(self.PETITE_home + self.dict_dir, current_material, 1)
        (Z, A, rho, _) = s.get_material_properties()
        ndensity = rho / A * 6.022e23
        angle_accept = np.sqrt(final_params['A_detector']) / (2 * (final_params['Lpipe'] + final_params['l0']))
        
        # Build the dictionary of all parameters for the final update
        new_params = {
            'rho': rho,
            'ndensity': ndensity,
            'angle_accept': angle_accept,
            'me': m_electron,
            'alpha_em': alpha_em,
            'mT': A,
            "Z_T": Z,
            "A_T": A,
        }
        new_params.update(final_params)

        # Printing changes & Updating dict
        changes_dict = {}
        for key in new_params:
            old_value = self.params_dict.get(key, None)
            new_value = new_params[key]
            if old_value != new_value:
                changes_dict[key] = (old_value, new_value)

        if print_updates:
            print("\nParameter Update Summary:")
            if not changes_dict:
                print("No changes detected.")
            else:
                print(f"{'Parameter':<17} {'Old Value':<15} {'New Value':<15}")
                print("="*47)
                for key, (old_val, new_val) in changes_dict.items():
                    old_str = f"{old_val:.4g}" if isinstance(old_val, float) else "None" if old_val is None else str(old_val)
                    new_str = f"{new_val:.4g}" if isinstance(new_val, float) else str(new_val)
                    print(f"{key:<17} {old_str:<15} {new_str:<15}")
            print("-" * 47)

        self.params_dict.update(new_params)
        self.shower_material = self.params_dict['shower_material']
        
        return changes_dict
        
    @staticmethod
    def _load_dict(path,VB=True):
        try:
            with open(path, 'rb') as f:
                if VB: print(f"[load] {path}.")
                return pickle.load(f)
        except Exception as e:
            print(f"[load] {path}: {e}")
            return {}
    
    @staticmethod    
    def _save_dict(data, file_path, VB=True):
        """Saves a dictionary to a pickle file."""
        if VB: print(f"--> Saving data back to: {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def _dict_merge(dict1,dict2):
        merged = {}

        for key in set(dict1) | set(dict2):
            merged[key] = []
            if key in dict1:
                merged[key].extend(dict1[key])
            if key in dict2:
                merged[key].extend(dict2[key])
        return merged
    
    @staticmethod
    def _save_single_mass(output_dir, mass, new_data, run_id, prefix='ax'):
        """
        Saves a batch of data for a single mass to a unique file in the output directory.
        """
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct a unique path for this batch, e.g., 'annihilation_axs/ax_0.1_1678886400.pkl'
        file_path = os.path.join(output_dir, f'{prefix}_{float(mass)}_{run_id}.pkl')

        # The data is stored in the format {mass: [list_of_axions]}
        with open(file_path, 'wb') as f:
            pickle.dump({mass: new_data}, f)
        print(f"[save] Batch data for mass {mass} saved to {os.path.basename(file_path)}")
    
        
    def _add_key(self, data_structure, key_to_add, value_to_add, dict_label):
        """
        Internal recursive method. It finds target dictionaries and ALWAYS 
        overwrites the key if it exists.
        """
        if isinstance(data_structure, dict):
            # Check if it's our target dictionary.
            if dict_label in data_structure:
                # Simplified logic: always add or overwrite the key.
                data_structure[key_to_add] = value_to_add
            else:
                # If not a target, continue searching in its values.
                for value in data_structure.values():
                    self._add_key(value, key_to_add, value_to_add, dict_label)
    
        elif isinstance(data_structure, list):
            # Continue searching in its items.
            for item in data_structure:
                self._add_key(item, key_to_add, value_to_add, dict_label)
        
        return data_structure

    def add_key_to_pkl_file(self, file_path, key_to_add, value_to_add, dict_label='p'):
        """
        Loads a .pkl file, modifies its data, and saves it back.
        The 'overwrite' flag now controls the FILE operation.

        Args:
            file_path (str): The full path to the pickle file.
            key_to_add (str): The name of the new key (e.g., 'N_primaries').
            value_to_add: The value to assign to the new key.
            dict_label (str): The key that identifies a target dictionary. Defaults to 'p'.
            overwrite (bool): If False, the method will abort if the file already
                              exists to prevent accidental overwrites. Defaults to True.
        """
        print(f"\nProcessing file: {file_path}")

        try:
            # 1. Load data
            data = self._load_dict(file_path)

            # 2. Call the internal recursive method
            modified_data = self._add_key(
                data, 
                key_to_add, 
                value_to_add, 
                dict_label
            )

            # 3. Save data, overwriting the original file
            self._save_dict(modified_data, file_path)
            print("✅ File successfully saved.")

        except FileNotFoundError:
            print(f"❌ ERROR: The file to be modified was not found at '{file_path}'. Please check the path.")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
    
    def _rename_key_recursively(self, data_structure, old_key, new_key, overwrite_existing):
        """Internal recursive method to rename a key."""
        if isinstance(data_structure, dict):
            # First, check if the old key exists to be renamed.
            if old_key in data_structure:
                # If the new key already exists, only proceed if allowed.
                if new_key in data_structure and not overwrite_existing:
                    print(f"--> SKIPPING rename: New key '{new_key}' already exists.")
                else:
                    # Perform the rename using pop()
                    data_structure[new_key] = data_structure.pop(old_key)
            
            # Continue searching deeper in the structure regardless
            for value in data_structure.values():
                self._rename_key_recursively(value, old_key, new_key, overwrite_existing)
        
        elif isinstance(data_structure, list):
            for item in data_structure:
                self._rename_key_recursively(item, old_key, new_key, overwrite_existing)
        
        return data_structure

    def rename_key_in_pkl_file(self, file_path, old_key, new_key, overwrite=True, overwrite_if_new_key_exists=False):
        """
        Loads a .pkl file, renames a key everywhere it appears, and saves it back.

        Args:
            file_path (str): The full path to the pickle file.
            old_key (str): The name of the key to rename.
            new_key (str): The new name for the key.
            overwrite (bool): If False, aborts if the FILE already exists.
            overwrite_if_new_key_exists (bool): If True, will perform the rename even
                if it overwrites an existing key with the new name. Defaults to False for safety.
        """
        print(f"\nAttempting to rename key '{old_key}' to '{new_key}' in file: {file_path}")
        if not overwrite and os.path.exists(file_path):
            print(f"❌ ABORTING: File '{file_path}' already exists and file overwrite is set to False.")
            return
        
        try:
            data = self._load_dict(file_path)
            modified_data = self._rename_key_recursively(data, old_key, new_key, overwrite_if_new_key_exists)
            self._save_dict(modified_data, file_path)
            print("✅ File successfully updated with renamed key.")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
         
    def _summarize_recursive(self, data_structure, indent, max_depth):
        """Internal recursive method to print the structure and sample values."""
        
        if len(indent) // 2 >= max_depth:
            print(f"{indent}[... Truncated at depth {max_depth} ...]")
            return

        # Case 1: The object is a dictionary
        if isinstance(data_structure, dict):
            print(f"dict ({len(data_structure)} keys)")
            for key, value in data_structure.items():
                print(f"{indent}  '{key}': ", end="")
                self._summarize_recursive(value, indent + "  ", max_depth)
        
        # Case 2: The object is a list
        elif isinstance(data_structure, list):
            if not data_structure:
                print("list (0 items) []")
            else:
                print(f"list ({len(data_structure)} items)")
                print(f"{indent}  [0]: ", end="") # Analyze the structure of the first item
                self._summarize_recursive(data_structure[0], indent + "  ", max_depth)
        
        # Case 3: The object is a terminal value (the end of a branch)
        else:
            # For numpy arrays, printing shape is safer and more informative
            if isinstance(data_structure, np.ndarray):
                print(f"<numpy.ndarray, shape={data_structure.shape}, dtype={data_structure.dtype}>")
            # For strings, print them but truncate if they are too long
            elif isinstance(data_structure, str):
                limit = 70
                val_str = str(data_structure)
                if len(val_str) > limit:
                    val_str = val_str[:limit] + '...'
                print(f"<{type(data_structure).__name__}> '{val_str}'")
            # For all other simple types (int, float, bool, None), print the value directly
            else:
                print(f"<{type(data_structure).__name__}> {data_structure}")

    def summarize_pkl_file(self, file_path, max_depth=10):
        """
        Loads a pickle file and prints a summary of its entire nested structure
        and sample values to avoid IOPub data rate errors in Jupyter.
        """
        print(f"\n--- Structural Summary of: {file_path} ---")
        if not os.path.exists(file_path):
            print("❌ ERROR: File not found.")
            return
        
        try:
            data = self._load_dict(file_path)
            self._summarize_recursive(data, indent="", max_depth=max_depth)
        except Exception as e:
            print(f"❌ An unexpected error occurred during summarization: {e}")
    
    ##############
    #GENERATION METHODS
    ##############
    
    def gen_all_vectors(self, masses, out_dir, run_id, active_processes=None):
        """
        Generates dark vectors and saves them to a uniquely identified batch file.
        """
        print("\n> Generating Dark Vectors... (Slow Foundational Step)")
        for index, ma in enumerate(masses):
            shower_cutoff=max(self.params_dict['shower_cutoff'],ma)
            print(f"\n> Initiating dark showers for ma={ma}, Ecutoff={shower_cutoff}.")
            
            s = DarkShower(self.PETITE_home + self.dict_dir,
                           self.shower_material, shower_cutoff, ma)
            if s._mV != ma:
                raise RuntimeError("Warning! Mass is not in the grid!")
            
            vec = vectors_from_beam(self.beam, s)
            self._save_single_mass(out_dir, ma, vec, run_id, prefix='vec')
        return True
                    
    def gen_ann_comp_axions_from_vectors(self, masses, vec_dir, out_ann_dir, out_comp_dir):
        """
        (Stage 2) Fast conversion. Finds ALL existing vector batches for a given mass,
        converts them to Annihilation and Compton axions, and saves a corresponding
        axion batch file for each, preserving the original run_id.
        """
        print("\n> Converting All Vectors to Ann+Comp Axions (Fast Conversion Step)...")
        os.makedirs(out_ann_dir, exist_ok=True)
        os.makedirs(out_comp_dir, exist_ok=True)
        if not os.path.exists(vec_dir):
            print(f"  Warning: Vector directory '{vec_dir}' not found. Cannot perform conversion.")
            return False

        for index, ma in enumerate(masses):

            # --- Find all unprocessed vector files for this mass ---
            vector_files_to_process = []
            for filename in os.listdir(vec_dir):
                if filename.startswith(f'vec_{float(ma)}_') and filename.endswith('.pkl'):
                    # Extract the original run_id from the vector filename
                    vector_run_id = filename.replace(f'vec_{float(ma)}_', '').replace('.pkl', '')

                    # Check if corresponding axion files already exist.
                    # If EITHER is missing, we need to reprocess this vector batch.
                    axion_ann_path = os.path.join(out_ann_dir, f'ax_{float(ma)}_{vector_run_id}.pkl')
                    axion_comp_path = os.path.join(out_comp_dir, f'ax_{float(ma)}_{vector_run_id}.pkl')

                    if not os.path.exists(axion_ann_path) or not os.path.exists(axion_comp_path):
                        vector_files_to_process.append(os.path.join(vec_dir, filename))

            if not vector_files_to_process:
                print(f"  No new vector batches to process for ma={ma}.")
                continue

            print(f"  Found {len(vector_files_to_process)} new vector batch(es) to convert for ma={ma}.")

            for vector_file_path in vector_files_to_process:
                vector_run_id = os.path.basename(vector_file_path).replace(f'vec_{float(ma)}_', '').replace('.pkl', '')
                print(f"    - Converting batch {vector_run_id}...")

                v = self._load_dict(vector_file_path).get(ma, {})
                if not v: continue

                # The conversion functions inherit 'n_primaries' from the vector dicts
                a_ann = convert_vec_to_axions_ann_comp(v.get("from_ann", []), ma, self.params_dict)
                a_comp = convert_vec_to_axions_ann_comp(v.get("from_comp", []), ma, self.params_dict)

                # Save the axion files with the SAME run_id as their parent vector file
                self._save_single_mass(out_ann_dir, ma, a_ann, vector_run_id, prefix='ax')
                self._save_single_mass(out_comp_dir, ma, a_comp, vector_run_id, prefix='ax')
                print(f"      Saved corresponding Ann+Comp axion batches for run_id {vector_run_id}.")
                print(f"\nAnn/Comp Axion Conversion Summary - ma = {ma} GeV - batch no {vector_run_id}\n" + "="*55)
                print(f"{'Kind':<20} {'Converted':>20}\n" + "-"*55)
                print(f"{'Annihilation':<20} {len(a_ann):>20}")
                print(f"{'Compton':<20} {len(a_comp):>20}\n" + "="*55)

        return True
       

    def gen_brem_axions_from_vectors(self, masses, vec_dir, out_el_dir, out_pos_dir, out_prim_dir=None):
        """
        (Stage 2) Fast conversion. Finds ALL existing vector batches for a given mass,
        converts them to Bremsstrahlung axions, and saves a corresponding axion
        batch file for each, preserving the original run_id.
        """
        print("\n> Converting All Vectors to Bremsstrahlung Axions (Fast Conversion Step)...")
        os.makedirs(out_el_dir, exist_ok=True)
        os.makedirs(out_pos_dir, exist_ok=True)
        if self.primaries == 'electrons' and out_prim_dir: os.makedirs(out_prim_dir, exist_ok=True)

        if not os.path.exists(vec_dir):
            print(f"  Warning: Vector directory '{vec_dir}' not found. Cannot perform conversion.")
            return False

        for index, ma in enumerate(masses):
           
            # --- Find all unprocessed vector files for this mass ---
            vector_files_to_process = []
            for filename in os.listdir(vec_dir):
                if filename.startswith(f'vec_{float(ma)}_') and filename.endswith('.pkl'):
                    vector_run_id = filename.replace(f'vec_{float(ma)}_', '').replace('.pkl', '')

                    # Check if corresponding output axion files exist
                    axion_el_path = os.path.join(out_el_dir, f'ax_{float(ma)}_{vector_run_id}.pkl')
                    axion_pos_path = os.path.join(out_pos_dir, f'ax_{float(ma)}_{vector_run_id}.pkl')

                    # Assume if the main shower components are missing, we need to re-process
                    if not os.path.exists(axion_el_path) or not os.path.exists(axion_pos_path):
                        vector_files_to_process.append(os.path.join(vec_dir, filename))
                    # Additionally check for primary if applicable
                    elif self.primaries == 'electrons' and out_prim_dir:
                        axion_prim_path = os.path.join(out_prim_dir, f'ax_{float(ma)}_{vector_run_id}.pkl')
                        if not os.path.exists(axion_prim_path):
                            vector_files_to_process.append(os.path.join(vec_dir, filename))

            if not vector_files_to_process:
                print(f"  No new vector batches to process for ma={ma}.")
                continue

            print(f"  Found {len(vector_files_to_process)} new vector batch(es) to convert for ma={ma}.")

            for vector_file_path in vector_files_to_process:
                vector_run_id = os.path.basename(vector_file_path).replace(f'vec_{float(ma)}_', '').replace('.pkl', '')
                print(f"    - Converting batch {vector_run_id}...")

                v = self._load_dict(vector_file_path).get(ma, {})
                if not v: continue

                prim_no='n/a'
                # Assumes 'convert_vec_to_axions_brem' inherits 'n_primaries'
                if self.primaries == 'electrons' and out_prim_dir:
                    a_prim = convert_vec_to_axions_brem(v.get("from_el_prim", []), ma, self.params_dict)
                    self._save_single_mass(out_prim_dir, ma, a_prim, vector_run_id, prefix='ax')
                    prim_no=len(a_prim)

                a_el = convert_vec_to_axions_brem(v.get("from_el", []), ma, self.params_dict)
                self._save_single_mass(out_el_dir, ma, a_el, vector_run_id, prefix='ax')

                a_pos = convert_vec_to_axions_brem(v.get("from_pos", []), ma, self.params_dict)
                self._save_single_mass(out_pos_dir, ma, a_pos, vector_run_id, prefix='ax')
                print(f"      Saved corresponding Brem axion batches for run_id {vector_run_id}.")
                print(f"\nBremsstrahlung Axion Conversion Summary - ma = {ma} GeV - batch no {vector_run_id}\n" + "="*55)
                print(f"{'Kind':<20} {'Converted':>20}\n" + "-"*55)
                print(f"{'Primary':<20} {prim_no:>20}")
                print(f"{'From electrons':<20} {len(a_el):>20}")
                print(f"{'From positrons':<20} {len(a_pos):>20}\n" + "="*55)
        return True

    
    def gen_primary_primakoff_axions(self, masses, out_prim_dir, run_id):
        """
        Generates axions from the primary photon beam and saves to a unique batch file.
        """
        print("\n> Generating Primary Primakoff Axions...")
        if self.primaries != 'photons':
            print("  Skipping: Primary beam is not photons.")
            return False

        for index, ma in enumerate(masses):
            s = Shower(self.PETITE_home + self.dict_dir, self.shower_material, ma)
            
            primary_photon_dicts = [
                {'p': p.get_p0(), 'weight': p.get_ids().get('weight', 1.0),
                 'rotation_matrix': p.rotation_matrix(), 'N_primaries': len(self.beam)}
                for p in self.beam
            ]
            
            a_prim, s_prim = convert_phot_to_axions_primakoff(s, primary_photon_dicts, ma, self.params_dict, return_stats=True)
            self._save_single_mass(out_prim_dir, ma, a_prim, run_id, prefix='ax')
                
        print(f"\nPrimary Primakoff Axion Conversion Summary — ma = {ma} GeV")
        print("=" * 85)
        print(f"{'Source':<15}  {'Converted':>12} {'Invalid weight':>20}")
        print("-" * 85)
        if s_prim:
            print(f"{'primary':<15}  {s_prim['converted']:>12} "
                  f"{s_prim.get('inv_prod_weight', 0):>20}")
        print("=" * 85 + "\n")
        print()
                    
        return True
    
    def gen_sm_shower_photons(self, masses, photons_dir, run_id, plothisto=False):
        """
        (Stage 1) Runs the slow SM shower to generate intermediate photons and saves them
        as a unique batch file. This is analogous to gen_all_vectors.
        """
        print("\n> Generating SM Shower Photons (Slow Foundational Step)...")
        os.makedirs(photons_dir, exist_ok=True)

        for index, ma in enumerate(masses):
            
            shower_cutoff=max(self.params_dict['shower_cutoff'], ma)
            
            print(f"\n> Initiating SM showers for ma={ma}, Ecutoff={shower_cutoff}.")
            s = Shower(self.PETITE_home + self.dict_dir, self.shower_material, shower_cutoff)
            photon_dicts = photons_from_beam(self.beam, s, plothisto=plothisto)

            # Always save a new, unique batch file for this run's photons
            self._save_single_mass(photons_dir, ma, photon_dicts, run_id, prefix='phot')
            print(f"Saved {len(photon_dicts)} photons for ma={ma} in batch {run_id}.")

        return True
                    
    
    def gen_primakoff_axions_from_photons(self, masses, photons_dir, out_dir):
        """
        (Stage 2) Fast conversion step. Finds ALL existing photon batches for a given
        mass, converts them to axions, and saves a corresponding axion batch file for each.
        """
        print("\n> Converting All Shower Photons to Primakoff Axions (Fast conversion step)...")
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(photons_dir):
            print(f"  Warning: Photon directory '{photons_dir}' not found. Cannot perform conversion.")
            return False

        for index, ma in enumerate(masses):

            # Find all unprocessed photon files for this mass
            photon_files_to_process = []
            for filename in os.listdir(photons_dir):
                if filename.startswith(f'phot_{float(ma)}_') and filename.endswith('.pkl'):
                    # Check if a corresponding axion file already exists. If not, process it.
                    photon_run_id = filename.replace(f'phot_{float(ma)}_', '').replace('.pkl', '')
                    axion_output_path = os.path.join(out_dir, f'ax_{float(ma)}_{photon_run_id}.pkl')
                    if not os.path.exists(axion_output_path):
                        photon_files_to_process.append(os.path.join(photons_dir, filename))

            if not photon_files_to_process:
                print(f"  No new photon batches to process for ma={ma}.")
                continue

            print(f"  Found {len(photon_files_to_process)} new photon batch(es) to convert for ma={ma}.")

            s = Shower(self.PETITE_home + self.dict_dir, self.shower_material, ma)
            for photon_file_path in photon_files_to_process:
                # Extract the original run_id from the photon filename
                photon_run_id = os.path.basename(photon_file_path).replace(f'phot_{float(ma)}_', '').replace('.pkl', '')

                print(f"    - Converting batch {photon_run_id}...")
                photon_dicts = self._load_dict(photon_file_path).get(ma, [])

                if not photon_dicts: continue

                # Convert this batch
                a_shower, s_shower = convert_phot_to_axions_primakoff(s, photon_dicts, ma, self.params_dict, return_stats=True)

                # Save the axion file with the SAME run_id as its parent photon file
                self._save_single_mass(out_dir, ma, a_shower, photon_run_id, prefix='ax')
                print(f"      Saved corresponding axion batch {photon_run_id}.")
                print(f"\nShower Primakoff Axion Conversion Summary — ma = {ma} GeV - batch no {photon_run_id} ")
                print("=" * 85)
                print(f"{'Source':<15}  {'Converted':>12} {'Invalid weight':>20}")
                print("-" * 85)
                print(f"{'shower ':<16} {s_shower['converted']:>12} "
                          f"{s_shower.get('inv_prod_weight', 0):>20}")
                print("=" * 85 + "\n")
                print()

        return True
    

    ##############
    #RUN METHODS
    ##############                 
                    
    # HELPER 1
    def _data_exists_for_mass(self, ma, check_dirs, prefix):
        """
        Checks if any data batch file exists for a given mass in a list of specified directories.
        This version safely handles None values in the check_dirs list.
        """
        # --- THIS IS THE FIX ---
        # We create a new list that only contains the valid directory paths, filtering out any None values.
        valid_dirs = [d for d in check_dirs if d is not None]

        for directory in valid_dirs:
            # It's still possible a directory hasn't been created yet, so this check remains.
            if not os.path.exists(directory):
                continue

            for filename in os.listdir(directory):
                # Check for a pattern like "ax_0.1_*.pkl"
                if filename.startswith(f'{prefix}_{float(ma)}_') and filename.endswith('.pkl'):
                    return True # Found a file, no need to check further.

        return False # Searched all valid directories and found nothing.

    # HELPER 2
    def _delete_data_for_masses(self, masses_to_delete, target_dirs, prefixes):
        """
        Finds and deletes all data batch files for specific masses within a targeted
        set of directories and for specific file prefixes.

        Args:
            masses_to_delete (list): The list of masses to delete.
            target_dirs (list): The specific directory paths to clean.
            prefixes (list): The file prefixes to look for (e.g., ['ax'], ['vec']).
        """
        if not masses_to_delete or not target_dirs:
            return 0

        files_deleted_count = 0
        # The [d for d in ... if d] safely handles cases where a directory might be None
        for directory in [d for d in target_dirs if d]:
            if not os.path.exists(directory):
                continue

            for filename in os.listdir(directory):
                for ma in masses_to_delete:
                    for prefix in prefixes:
                        if filename.startswith(f'{prefix}_{float(ma)}_') and filename.endswith('.pkl'):
                            file_path = os.path.join(directory, filename)
                            try:
                                os.remove(file_path)
                                files_deleted_count += 1
                            except OSError as e:
                                print(f"[Error] Could not delete file {file_path}: {e}")

        return files_deleted_count     
    
    def run(self, beam, masses, active_processes=None, mode='run', clean=False, primary_only=False, run_id=None):
        """
        Main execution method for the stateless, Two-Stage Generation model.

        This method orchestrates data generation with a clear, mode-driven API that
        handles appending, safe-resuming, and surgical or total overwrites.

        Args:
            beam (list): The primary beam particles for this run.
            masses (list): A list of axion masses to generate or manage.
            active_processes (list, optional): Processes to run, e.g., ['Brem']. Defaults to all.
            mode (str): The execution mode:
                - 'run' (Default): Safe "fill-in" mode. For each process and mass, it first checks
                  if the FINAL axion data already exists. If yes, it does nothing for that
                  process. If no, it runs the entire generation pipeline for it.
                - 'append': Always generates a new, unique batch of data files, adding to
                  any existing statistics. This is the primary mode for increasing data.
                - 'overwrite': Deletes all previous data for the specified masses AND active
                  processes, then generates a new batch.
            clean (bool): Only used with `mode='overwrite'`. If True, deletes the entire
                          data folder for a total reset before starting.
            primary_only (bool): If True, only runs the primary beam component of 'Primakoff'.
            run_id (int): specify a label for the batch. If None, use int(time.time()).
        """
        # =========================================================================
        # --- 1. Initial Setup and Safety Checks ---
        # =========================================================================
        if clean and mode != 'overwrite':
            raise ValueError("The 'clean=True' flag can only be used with 'mode=\"overwrite\"'.")

        self.beam = beam
        if beam[0].get_ids()['PID'] == 11: self.primaries = 'electrons'
        elif beam[0].get_ids()['PID'] == 22: self.primaries = 'photons'
        else: raise ValueError(f"Invalid beam type provided for this run!")

        # if self.params_dict.get('primary_type') is None:
        #     self.params_dict['primary_type'] = self.primaries
        # elif self.params_dict.get('primary_type') != self.primaries:
        #      raise RuntimeError(f"Beam Type Mismatch! Data in '{self.DATA_folder_path}' was created with "
        #                         f"'{self.params_dict.get('primary_type')}' beams. Use a new folder or run with "
        #                         f"mode='overwrite', clean=True.")

        default_processes = ['Ann+Comp', 'Brem', 'Primakoff']
        if active_processes is None: active_processes = default_processes
        else:
            invalid = [p for p in active_processes if p not in default_processes]
            if invalid:
                raise ValueError(
                    f"Invalid process name(s): {invalid}. "
                    f"Allowed names are: {default_processes}. "
                    f"Please correct your 'active_processes' input."
                )

        if primary_only:
            # Check if there are any active processes that are NOT 'Primakoff'
            other_processes = [p for p in active_processes if p != 'Primakoff']
            if other_processes:
                raise ValueError(
                    f"The 'primary_only=True' flag is exclusively for the 'Primakoff' process. "
                    f"You have also included other active processes: {other_processes}. "
                    f"Please either remove the primary_only flag or set active_processes=['Primakoff']."
                )
            # This check for photon beam must still exist, but can be simplified
            if self.primaries != 'photons':
                raise ValueError("The 'primary_only=True' flag can only be used with a photon primary beam.")

        # =========================================================================
        # --- 2. Define Paths and the Process-to-Directory Map ---
        # =========================================================================
        folder_path = self.DATA_folder_path
        all_vec_dir = os.path.join(folder_path, 'dark_vecs')
        ann_axion_dir = os.path.join(folder_path, 'annihilation_axs'); comp_axion_dir = os.path.join(folder_path, 'compton_axs')
        brem_axion_el_dir = os.path.join(folder_path, 'brem_el_axs'); brem_axion_pos_dir = os.path.join(folder_path, 'brem_pos_axs')
        primakoff_out_dir = os.path.join(folder_path, 'primakoff_axs'); primakoff_photons_dir = os.path.join(folder_path, 'primakoff_phot')

        brem_axion_prim_dir, primakoff_axion_prim_dir = None, None
        if self.primaries == 'electrons': brem_axion_prim_dir = os.path.join(folder_path, 'brem_prim_axs')
        if self.primaries == 'photons': primakoff_axion_prim_dir = os.path.join(folder_path, 'primakoff_prim_axs')

        process_to_dirs = {
            'Vectors':   ([all_vec_dir], ['vec']),
            'Ann+Comp':  ([ann_axion_dir, comp_axion_dir], ['ax']),
            'Brem':      ([brem_axion_el_dir, brem_axion_pos_dir, brem_axion_prim_dir], ['ax']),
            'Primakoff': ([primakoff_out_dir, primakoff_photons_dir], ['ax', 'phot']),
            'PrimaryPrimakoff': ([primakoff_axion_prim_dir], ['ax'])
        }

        # =========================================================================
        # --- 3. Handle Deletion/Overwrite Logic ---
        # =========================================================================
        if mode == 'overwrite':
            if clean:
                print(f"[Mode: overwrite, clean=True] Deleting entire data folder: {self.DATA_folder_path}")
                if os.path.exists(self.DATA_folder_path): shutil.rmtree(self.DATA_folder_path)
                os.makedirs(self.DATA_folder_path, exist_ok=True)
            else:
                # Surgical, Process-Aware Overwrite
                print(f"\n[Overwrite] Surgically deleting data for active processes: {active_processes}...")
                total_deleted = 0
                procs_to_delete = set(active_processes)
                if 'Brem' in procs_to_delete or 'Ann+Comp' in procs_to_delete: procs_to_delete.add('Vectors')
                if 'Primakoff' in procs_to_delete:
                    if not primary_only: procs_to_delete.add('Primakoff')
                    else: procs_to_delete.add('PrimaryPrimakoff')

                for process_name in procs_to_delete:
                    if process_name in process_to_dirs:
                        dirs, prefixes = process_to_dirs[process_name]
                        total_deleted += self._delete_data_for_masses(masses, dirs, prefixes)
                print(f"[Overwrite] Deletion complete. {total_deleted} files removed.")

        # =========================================================================
        # --- 4. Main Generation Loop ---
        # =========================================================================
        if run_id is None:
            run_id = int(time.time())
        print(f"\nStarting generation batch with run_id: {run_id} (mode: '{mode}')")

        for ma in masses:
            print(f"\n--- Processing ma = {ma} GeV ---")
            mass_list = [ma]

            # --- A) Vector-Dependent Processes ---
            if 'Brem' in active_processes or 'Ann+Comp' in active_processes:
                # Stage 1: Generate foundational vectors unless mode='run' and they already exist.
                # In append/overwrite modes, this always runs to create a new batch.
                vec_dirs, vec_prefix = process_to_dirs['Vectors']
                if mode == 'run' and self._data_exists_for_mass(ma, vec_dirs, vec_prefix[0]):
                     print(f"  > Foundational vector data exists for ma={ma}. Will use for conversion if needed.")
                else:
                    self.gen_all_vectors(mass_list, all_vec_dir, run_id, active_processes)

            # Stage 2: Convert vectors to axions, respecting 'run' mode for final outputs.
            if 'Ann+Comp' in active_processes:
                dirs, prefix = process_to_dirs['Ann+Comp']
                if mode == 'run' and self._data_exists_for_mass(ma, dirs, prefix[0]):
                    print(f"  > Final Ann+Comp data found. Skipping.")
                else:
                    self.gen_ann_comp_axions_from_vectors(mass_list, all_vec_dir, ann_axion_dir, comp_axion_dir)

            if 'Brem' in active_processes:
                dirs, prefix = process_to_dirs['Brem']
                if mode == 'run' and self._data_exists_for_mass(ma, dirs, prefix[0]):
                    print(f"  > Final Brem data found. Skipping.")
                else:
                    self.gen_brem_axions_from_vectors(mass_list, all_vec_dir, brem_axion_el_dir, brem_axion_pos_dir, out_prim_dir=brem_axion_prim_dir)

            # --- B) Shower Primakoff Process ---
            if 'Primakoff' in active_processes and not primary_only:
                dirs, prefix = process_to_dirs['Primakoff']
                if mode == 'run' and self._data_exists_for_mass(ma, [dirs[0]], prefix[0]):
                    print(f"  > Final Shower Primakoff data found. Skipping.")
                else:
                    # Run the full two-stage process for Primakoff if final data is missing.
                    if mode == 'run' and self._data_exists_for_mass(ma, [dirs[1]], prefix[1]):
                         print(f"> Foundational photons data exists for ma={ma}. Will use for conversion if needed.")
                    else:
                        self.gen_sm_shower_photons(mass_list, primakoff_photons_dir, run_id)
                    
                    self.gen_primakoff_axions_from_photons(mass_list, primakoff_photons_dir, primakoff_out_dir)

            # --- C) Primary-Only Primakoff (No intermediate step) ---
            if 'Primakoff' in active_processes and self.primaries == 'photons':
                dirs, prefix = process_to_dirs['PrimaryPrimakoff']
                if mode == 'run' and self._data_exists_for_mass(ma, dirs, prefix[0]):
                     print(f"  > Final Primary Primakoff data found. Skipping.")
                else:
                    self.gen_primary_primakoff_axions(mass_list, primakoff_axion_prim_dir, run_id)

        # =========================================================================
        # --- 5. Finalization ---
        # =========================================================================
        run_args = {'run_ID': run_id, 'N_primaries': len(beam), 'masses': masses, 'mode': mode, 'clean': clean, 'active_processes': active_processes}
        self.logger.log_run(self, run_args)
        self.logger._save_log_state(self.params_dict) # Save current physics/experiment parameters

        print("\n####################\nAll generation processes completed.\n####################")
        
                    
    
    def run_exp_weights(self, Lpipe=None, l0=None, A_detector=None, Ecut=None, POT=None, gayy_coeff= None, gaee_coeff = None, inelastic_on = None):
        """
        Enforces a set of experimental weights across all existing axion data batches.

        This method is fully automatic and robust against interruption. It scans
        all axion data directories, finds every data batch, and ensures its
        weights are consistent with the parameters provided to this function.
        If the script is interrupted, it can be safely re-run to complete the job.

        Args:
            Lpipe (float): The desired length of the decay pipe.
            l0 (float): The desired distance from target to detector shielding.
            A_detector (float): The desired area of the detector.
            Ecut (float): The desired minimum energy for an axion to be detected.
            POT (float): The desired particles on target.
            gayy_coeff (float): The desired coeff of the axion-photon coupling constant, L_int = gayy_coeff/4 * a_F_Ftilde.
            gaee_coeff (float): The desired coeff of the axion-electron coupling constant, L_int = gaee_coeff * e * a_ebar_Gamma5_e.
        """
        print("\n####################\nStarting Experimental Weighting Run...\n####################")

        # it takes arguments, fills defaults from self.params_dict, and then
        # updates self.params_dict to a final, consistent state.
        if Lpipe is None:
            Lpipe= self.params_dict['Lpipe']
        if l0 is None:
            l0= self.params_dict['l0']
        if A_detector is None:
            A_detector= self.params_dict['A_detector']
        if Ecut is None:
            Ecut= self.params_dict['Ecut']
        if POT is None:
            POT= self.params_dict['POT']
        if gayy_coeff is None:
            gayy_coeff=self.params_dict['gayy_coeff']
        if gaee_coeff is None:
            gaee_coeff=self.params_dict['gaee_coeff']
        if inelastic_on is None:
            inelastic_on=self.params_dict['inelastic_on']
        
        # --- 1. Update the object's parameters. ---
        print("\n--> Step 1: Setting and saving the target parameter state for this run.")
        changes_dict=self._update_params_dict(gayy_coeff= gayy_coeff, gaee_coeff=gaee_coeff, Lpipe=Lpipe, l0=l0, A_detector=A_detector, Ecut=Ecut, POT=POT,inelastic_on=inelastic_on, print_updates=True)

        # --- The rest of this function remains unchanged as its core logic is correct ---
        folder_path = self.DATA_folder_path
        all_axion_dirs = [
            os.path.join(folder_path, 'annihilation_axs'),
            os.path.join(folder_path, 'compton_axs'),
            os.path.join(folder_path, 'brem_el_axs'),
            os.path.join(folder_path, 'brem_pos_axs'),
            os.path.join(folder_path, 'primakoff_axs'),
            os.path.join(folder_path, 'brem_prim_axs'),
            os.path.join(folder_path, 'primakoff_prim_axs')
        ]
        existing_axion_dirs = [d for d in all_axion_dirs if os.path.exists(d)]

        if not existing_axion_dirs:
            print("\n[Warning] No data directories found. Nothing to update.")
            return
        
        electron_based_processes = [
            'annihilation_axs', 'compton_axs', 'brem_el_axs', 
            'brem_pos_axs', 'brem_prim_axs'
        ]

        print("\n--> Step 2: Enforcing state by re-weighting all discovered axion data batches.")
        total_files_updated = 0
        for axion_dir in existing_axion_dirs:
            dir_basename = os.path.basename(axion_dir)
            print(f"--- Scanning Directory: {os.path.basename(axion_dir)} ---")
            
            is_electron_based = dir_basename in electron_based_processes
            process_type = "Electron-based" if is_electron_based else "Primakoff"
            print(f"    Process type detected as: {process_type}")
            
            files_in_dir_updated = 0
            
            files_to_process = [f for f in os.listdir(axion_dir) if f.startswith('ax_') and f.endswith('.pkl')]
            if not files_to_process:
                print("    No axion files found in this directory.")
                continue

            for filename in files_to_process:
                file_path = os.path.join(axion_dir, filename)
                try:
                    axion_data_dict = self._load_dict(file_path, VB=False)
                    if not axion_data_dict: continue

                    mass_key = list(axion_data_dict.keys())[0]
                    axion_list = axion_data_dict[mass_key]
                    updated_axion_list = axions_exp_weights(axion_list, self.params_dict, is_electron_based)
                    self._save_dict({mass_key: updated_axion_list}, file_path, VB=False)
                    files_in_dir_updated += 1
                    
                except Exception as e:
                    print(f"❌ ERROR updating file {filename}: {e}")
            
            if files_in_dir_updated > 0:
                print(f"--> Enforced state and experimental weights for {files_in_dir_updated} batch file(s).")
                total_files_updated += files_in_dir_updated

        self.logger.log_update(self, changes_dict, True, updated_masses=[])
        print(f"\n####################\nState enforcement complete. {total_files_updated} total batch files processed.\n####################")


    ##############
    #PLOTTING METHODS
    ##############  
                    
    def finalize_data(
        self,
        folder_name='PLOT',
        overwrite: bool = True,
        run_ids_to_include=None,
        **exp_params
    ):
        """
        Collects, re-weights, and finalizes data into a per-mass directory structure.

        This is the primary method for preparing data for plotting. It performs two main actions:
        1.  It enforces a consistent set of experimental weights on all raw data files
            by calling `run_exp_weights`. You can pass parameters like `POT`, `Lpipe`, etc.,
            directly to this function.
        2.  It aggregates the re-weighted data, creating a subdirectory for each mass
            (e.g., 'PLOT/0.1/') with a separate file for each process.
        3.  It saves a 'parameters.json' file in the output folder for traceability.

        Args:
            folder_name (str, optional): The name of the main output folder. Defaults to 'PLOT'.
            overwrite (bool, optional): If True (default), existing data files will
                be overwritten. If False, any existing file will be skipped.
            run_ids_to_include (list, optional): A list of run_ids to include. Defaults to all.
            **exp_params: Keyword arguments for experimental parameters to be passed directly
                to `run_exp_weights`. For example: `POT=1e22`, `Lpipe=150`.
        
        Returns:
            dict: A dictionary mapping mass values to their output directories.
        """
        print("\n> Starting data finalization...")
        plot_folder_path = os.path.join(self.DATA_folder_path, folder_name)
        print(f"  Output directory: {os.path.abspath(plot_folder_path)}")
        print(f"  Overwrite existing files: {overwrite}")
        
        # If exp_params is empty, run_exp_weights will use its defaults (the current state).
        # If exp_params has values, they will be used for the re-weighting.
        print("\n> Applying experimental weights to all axions...")
        self.run_exp_weights(**exp_params)

        # --- The rest of the logic remains largely the same ---
        if run_ids_to_include:
            run_ids_to_include = [str(rid) for rid in run_ids_to_include]
            print(f"  Filtering for run_ids: {run_ids_to_include}")

        process_map = {
            'Annihilation': os.path.join(self.DATA_folder_path, 'annihilation_axs'),
            'Compton': os.path.join(self.DATA_folder_path, 'compton_axs'),
            'Brem_el': os.path.join(self.DATA_folder_path, 'brem_el_axs'),
            'Brem_pos': os.path.join(self.DATA_folder_path, 'brem_pos_axs'),
            'Primakoff_shower': os.path.join(self.DATA_folder_path, 'primakoff_axs')
        }
        if self.primaries == 'electrons':
            process_map['Brem_primary'] = os.path.join(self.DATA_folder_path, 'brem_prim_axs')
        elif self.primaries == 'photons':
            process_map['Primakoff_primary'] = os.path.join(self.DATA_folder_path, 'primakoff_prim_axs')

        created_dirs_by_mass = {}

        for process_name, source_dir in process_map.items():
            if not os.path.exists(source_dir): continue

            # Data aggregation logic... (no changes here)
            aggregated_axions_by_mass = {}
            total_primaries_by_mass = {}
            batches_per_mass = {}
            for filename in os.listdir(source_dir):
                if not (filename.startswith('ax_') and filename.endswith('.pkl')): continue
                run_id = filename.split('_')[-1].replace('.pkl', '')
                if run_ids_to_include and run_id not in run_ids_to_include: continue
                batch_data = self._load_dict(os.path.join(source_dir, filename), VB=False)
                for ma, axions in batch_data.items():
                    if not axions: continue
                    if ma not in aggregated_axions_by_mass:
                        aggregated_axions_by_mass[ma] = []
                        total_primaries_by_mass[ma] = 0
                        batches_per_mass[ma] = 0
                    aggregated_axions_by_mass[ma].extend(axions)
                    total_primaries_by_mass[ma] += axions[0].get('N_primaries', 0)
                    batches_per_mass[ma] += 1
            if not aggregated_axions_by_mass: continue
            
            # Print summary and write files... (no changes here)
            print("\n" + "="*80)
            print(f"Finalization Summary for: {process_name}")
            print(f"  > Output structure: {folder_name}/<mass>/{process_name}.pkl")
            print("-"*80)
            print(f"{'Mass':<10} {'Batches':<10} {'Events':<15} {'Primaries':<15} {'Status'}")
            print("-"*80)
            for ma in sorted(aggregated_axions_by_mass.keys()):
                mass_dir_path = os.path.join(plot_folder_path, str(float(ma)))
                final_path = os.path.join(mass_dir_path, f"{process_name}.pkl")
                file_existed_before = os.path.exists(final_path)
                if file_existed_before and not overwrite:
                    print(f"{ma:<10.4g} {'-':<10} {'-':<15} {'-':<15} ✅ Skipping existing file.")
                    continue
                os.makedirs(mass_dir_path, exist_ok=True)
                created_dirs_by_mass[ma] = mass_dir_path
                axion_list = aggregated_axions_by_mass[ma]
                N_total = total_primaries_by_mass[ma]
                w_prim = 1.0 / N_total if N_total > 0 else 0.0
                for axion in axion_list:
                    axion['w_prim'] = w_prim
                with open(final_path, 'wb') as f:
                    pickle.dump(axion_list, f)
                status_msg = "📝 Overwritten file." if file_existed_before else "🆕 Created new file."
                print(f"{ma:<10.4g} {batches_per_mass[ma]:<10} {len(axion_list):<15,d} {N_total:<15,d} {status_msg}")
            print("="*80)

        # --- Save the parameters used for this run to a JSON file ---
        # The self.params_dict was updated by run_exp_weights, so it's our source of truth.
        if created_dirs_by_mass: # Only save if we actually created/updated some data
            params_file_path = os.path.join(plot_folder_path, 'parameters.json')
            print(f"\n> Saving experimental parameters to: {params_file_path}")
            try:
                # Ensure the main plot directory exists
                os.makedirs(plot_folder_path, exist_ok=True)
                with open(params_file_path, 'w') as f:
                    json.dump(self.params_dict, f, indent=4)
                print("  ✅ Parameters saved successfully.")
            except Exception as e:
                print(f"  ❌ ERROR saving parameters file: {e}")

        print("\n✅ All processes finalized.\n")
        return created_dirs_by_mass
    
    def plot_histo_flux(self, masses, folder_name='PLOT', w=5, h=4, legend_title_text=None,
                        bins_per_decade=20, log_scale=True, weights_to_use=None,
                        gaee_factor=None, gayy_factor=None, xlim=None, ylim=None,
                        plot_subprocesses=False, processes_to_plot=None,
                        save_plot=True, plot_filename='Flux.png', plot_title=''):
        """
        Plots a publication-quality axion flux histogram from finalized data.

        This enhanced function can plot data for multiple masses on the same axes and allows
        for plotting only specific, user-defined particle production processes.

        Args:
            masses (float or list of float): A single axion mass or a list of masses to plot.
            folder_name (str): The directory containing the finalized data.
            w, h (float): The target width and height of the axes area in inches.
            legend_title_text (str): Optional title for the plot legend.
            bins_per_decade (int): Number of bins per decade of energy.
            log_scale (bool): Use logarithmic scale for the y-axis if True.
            weights_to_use (list of str): List of weight keys to multiply from the data.
            gaee_factor, gayy_factor (float, optional): Rescaling factors for couplings.
            xlim, ylim (tuple, optional): Manually set the x/y-axis limits.
            plot_subprocesses (bool): If True, plots individual contributions (e.g., Annihilation,
                                      Compton) separately. This is ignored if `processes_to_plot` is set.
            processes_to_plot (list of str, optional): A list of process keys to plot exclusively.
                                                      If None, plots default combinations. Available keys are:
                                                      'ann', 'comp', 'brem_el', 'brem_pos',
                                                      'primakoff_shower', 'brem_primary', 'primakoff_primary'.
            save_plot (bool): If True, saves the plot to a file.
            plot_filename (str): The filename for the saved plot.
            plot_title (str): Additional text to append to the plot title.
        """
        # --- 1. Setup: Normalize Inputs, Styles, and Process Definitions ---
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # Normalize the 'masses' input to always be a list for consistent looping
        if not isinstance(masses, list):
            masses = [masses]

        # Master dictionary defining all known processes, their files, labels, and couplings.
        # This makes the logic cleaner and easier to extend.
        ALL_PROCESS_DEFINITIONS = {
            'ann': {'file': 'Annihilation.pkl', 'label': r'Annihilation', 'coupling': 'gaee'},
            'comp': {'file': 'Compton.pkl', 'label': r'Compton', 'coupling': 'gaee'},
            'brem_el': {'file': 'Brem_el.pkl', 'label': r'Brem, Shower e-', 'coupling': 'gaee'},
            'brem_pos': {'file': 'Brem_pos.pkl', 'label': r'Brem, Shower e+', 'coupling': 'gaee'},
            #'primakoff_shower': {'file': 'Primakoff_shower.pkl', 'label': r'Primakoff, Shower', 'coupling': 'gayy'},
            'primakoff_shower': {'file': 'Primakoff_shower.pkl', 'label': r'', 'coupling': 'gayy'},
            'brem_primary': {'file': 'Brem_primary.pkl', 'label': r'Brem, Primary', 'coupling': 'gaee'},
            'primakoff_primary': {'file': 'Primakoff_primary.pkl', 'label': r'Primakoff, Primary', 'coupling': 'gayy'},
        }

         # --- 2. Build Plotting Recipes ---
        # A "recipe" is a dictionary defining what keys to combine and what label to use.
        recipes = []
        if processes_to_plot:
            # User wants specific processes. `plot_subprocesses` is ignored.
            for key in processes_to_plot:
                if key in ALL_PROCESS_DEFINITIONS:
                    recipes.append({
                        'keys': [key],
                        'label': ALL_PROCESS_DEFINITIONS[key]['label']
                    })
                else:
                    print(f"[Warning] Process key '{key}' not recognized. Skipping.")
        else:
            # Default behavior: use standard groupings or split them if requested.
            if not plot_subprocesses:
                recipes = [
                    {'keys': ['ann', 'comp'], 'label': r'Annihilation + Compton'},
                    {'keys': ['brem_el', 'brem_pos'], 'label': r'Bremsstrahlung, Shower'}, # The requested label
                    {'keys': ['primakoff_shower'], 'label': r'Primakoff, Shower'}
                ]
            else:
                recipes = [
                    {'keys': ['ann'], 'label': r'Annihilation'},
                    {'keys': ['comp'], 'label': r'Compton'},
                    {'keys': ['brem_el'], 'label': r'Brem, Shower e-'},
                    {'keys': ['brem_pos'], 'label': r'Brem, Shower e+'},
                    {'keys': ['primakoff_shower'], 'label': r'Primakoff, Shower'}
                ]

            # Add the primary process recipe regardless of subprocess plotting
            if self.primaries == 'electrons':
                recipes.append({'keys': ['brem_primary'], 'label': r'Brem, Primary'})
            elif self.primaries == 'photons':
                recipes.append({'keys': ['primakoff_primary'], 'label': r'Primakoff, Primary'})

         # --- 3. Pre-scan Data to Determine Global Binning Range ---
        print("Scanning data to determine optimal energy range for bins...")
        all_energies = []
        all_recipe_keys = [key for recipe in recipes for key in recipe['keys']]
        for ma in masses:
            mass_dir_path = os.path.join(self.DATA_folder_path, folder_name, str(ma))
            if not os.path.isdir(mass_dir_path):
                print(f"[Warning] Directory for mass '{ma}' not found. Skipping this mass.")
                continue
            for key in all_recipe_keys:
                file_path = os.path.join(mass_dir_path, ALL_PROCESS_DEFINITIONS[key]['file'])
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        all_energies.extend([d['p'][0] for d in data])
                except FileNotFoundError:
                    continue

        if not all_energies:
            print("\n[Error] No data found for the specified masses and processes. Cannot create plot.")
            return

        E_min, E_max = min(all_energies), max(all_energies)
        common_bins = np.logspace(np.log10(E_min), np.log10(E_max), int(np.ceil((np.log10(E_max) - np.log10(E_min)) * bins_per_decade)))

        # --- 4. Create Plot and Iterate Through Masses and Recipes ---
        fig, ax = plt.subplots(1, 1)
        if weights_to_use is None:
            weights_to_use = ['w_prod_scaled', 'w_angle', 'w_LLP_gone', 'w_Ecut', 'w_prim']

        for ma in masses:
            mass_dir_path = os.path.join(self.DATA_folder_path, folder_name, str(ma))
            if not os.path.isdir(mass_dir_path): continue

            for recipe in recipes:
                combined_axion_list = []
                for key in recipe['keys']:
                    file_path = os.path.join(mass_dir_path, ALL_PROCESS_DEFINITIONS[key]['file'])
                    try:
                        with open(file_path, 'rb') as f:
                            combined_axion_list.extend(pickle.load(f))
                    except FileNotFoundError:
                        continue

                if not combined_axion_list: continue

                # Determine plot label and rescaling factor
                plot_label = recipe['label']
                if len(masses) > 1:
                    #plot_label += fr', $m_a={int(ma*1000)}$ MeV'
                    plot_label += fr'$m_a={int(ma*1000)}$ MeV'

                first_key = recipe['keys'][0]
                coupling_type = ALL_PROCESS_DEFINITIONS[first_key]['coupling']
                rescale_factor = 1.0
                if coupling_type == 'gaee':
                    rescale_factor = (gaee_factor or 1.0)**2
                elif coupling_type == 'gayy':
                    rescale_factor = (gayy_factor or 1.0)**2

                E_values = [d['p'][0] for d in combined_axion_list]
                weights = [np.prod([d.get(wk, 1.0) for wk in weights_to_use]) * rescale_factor for d in combined_axion_list]
                ax.hist(E_values, bins=common_bins, weights=weights, histtype='step', linewidth=2.0, label=plot_label)

        # --- 5. Apply Final Formatting and Layout ---
        ax.set_xlabel(r'Energy $E$ [GeV]')
        ax.set_ylabel(r'Axions $\times f^4$ / POT $[{\rm GeV}^{4}]$')

        if len(masses) == 1:
            ax.set_title(fr'Axion Flux for $m_a = {int(masses[0]*1000)}$ MeV' + plot_title)
        else:
            ax.set_title(r'Axion Flux Comparison' + plot_title)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.8)
        legend = ax.legend(title=legend_title_text, loc='upper center', ncol=2)
        if legend_title_text:
            plt.setp(legend.get_title(), multialignment='left')
        set_size(w, h, ax=ax)

        if save_plot:
            plot_folder_path = os.path.join(self.DATA_folder_path, folder_name)
            final_save_path = os.path.join(plot_folder_path, plot_filename)
            try:
                fig.savefig(final_save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot successfully saved to: {final_save_path}")
            except Exception as e:
                print(f"❌ Error saving plot: {e}")

        plt.show()
    
#     def plot_histo_flux(self, ma, folder_name='PLOT', w=5, h=4, legend_title_text=None,
#                     bins_per_decade=20, log_scale=True, weights_to_use=None,
#                     gaee_factor=None, gayy_factor=None, xlim=None, ylim=None, plot_subprocesses=False, save_plot=True, plot_filename='Flux.png',plot_title=''):
#         """
#         Plots a publication-quality axion flux histogram from finalized data.

#         This function uses a helper to set a precise figure size and applies styling
#         to closely match professional examples. It correctly loads data from the
#         `folder_name/<mass>/<ProcessName>.pkl` structure.

#         Args:
#             ma (float): The specific axion mass to plot.
#             folder_name (str): The directory containing the finalized data.
#             w, h (float): The target width and height of the axes area in inches.
#             legend_title_text (str): Optional title for the plot legend.
#             bins_per_decade (int, optional): Number of bins per decade of energy. Defaults to 20.
#             log_scale (bool): Use logarithmic scale for the y-axis if True.
#             weights_to_use (list of str): List of weight keys to multiply from the data.
#             lim (tuple, optional): A tuple (xmin, xmax) to manually set the x-axis limits.
#             ylim (tuple, optional): A tuple (ymin, ymax) to manually set the y-axis limits.
#             plot_subprocesses (bool): If True, plots individual contributions (e.g., Annihilation,
#                                       Compton) separately instead of combining them. Defaults to False.
#         """
#         # --- 1. Setup Paths, Plotting Style, and Process Mappings ---
#         plt.rc('text', usetex=True)
#         plt.rc('font', family='serif')

#         mass_dir_path = os.path.join(self.DATA_folder_path, folder_name, str(ma))
#         if not os.path.isdir(mass_dir_path):
#             print(f"[Error] Directory for mass '{ma}' not found. Please run `finalize_data()`.")
#             return

#         process_filename_map = {
#             'ann': 'Annihilation.pkl', 'comp': 'Compton.pkl', 'brem_el': 'Brem_el.pkl',
#             'brem_pos': 'Brem_pos.pkl', 'primakoff_shower': 'Primakoff_shower.pkl',
#             'brem_primary': 'Brem_primary.pkl', 'primakoff_primary': 'Primakoff_primary.pkl'
#         }

#         # --- 2. Create Plotting Recipes ---
#         plot_recipes = []
#         # Annihilation and Compton
#         if not plot_subprocesses:
#             plot_recipes.append({'label': r'Annihilation + Compton', 'keys': ['ann', 'comp'], 'rescale': gaee_factor or 1})
#         else:
#             plot_recipes.extend([
#                 {'label': r'Annihilation', 'keys': ['ann'],  'rescale': gaee_factor or 1},
#                 {'label': r'Compton', 'keys': ['comp'],  'rescale': gaee_factor or 1}
#             ])

#         # Bremsstrahlung from shower electrons/positrons
#         if not plot_subprocesses:
#             plot_recipes.append({'label': r'Bremsstrahlung, Shower', 'keys': ['brem_el', 'brem_pos'], 'rescale': gaee_factor or 1})
#         else:
#             plot_recipes.extend([
#                 {'label': r'Brem, Shower e-', 'keys': ['brem_el'],  'rescale': gaee_factor or 1},
#                 {'label': r'Brem, Shower e+', 'keys': ['brem_pos'], 'rescale': gaee_factor or 1}
#             ])

#         # Primakoff from shower photons
#         plot_recipes.append({'label': r'Primakoff, Shower', 'keys': ['primakoff_shower'], 'rescale': gayy_factor or 1})


#         # --- 2. Conditionally Add the Primary Beam Process ---
#         # This is the ONLY part that depends on the initial beam type.

#         if self.primaries == 'electrons':
#             plot_recipes.append({'label': r'Brem, Primary', 'keys': ['brem_primary'], 'rescale': gaee_factor or 1})
#         elif self.primaries == 'photons':
#             plot_recipes.append({'label': r'Primakoff, Primary', 'keys': ['primakoff_primary'], 'rescale': gayy_factor or 1})

#         # --- 3. Load Data and Prepare for Plotting ---
#         data_to_plot = []
#         for recipe in plot_recipes:
#             combined_axion_list = []
#             for key in recipe['keys']:
#                 file_path = os.path.join(mass_dir_path, process_filename_map.get(key, ''))
#                 try:
#                     with open(file_path, 'rb') as f:
#                         combined_axion_list.extend(pickle.load(f))
#                 except FileNotFoundError:
#                     continue  # Silently ignore missing process files

#             if combined_axion_list:
#                 data_to_plot.append({
#                     'recipe': recipe,
#                     'data': combined_axion_list
#                 })

#         if not data_to_plot:
#             print(f"\n[Error] No finalized process files were found for mass ma={ma} in '{mass_dir_path}'.")
#             return

#         # --- 4. Binning and Plot Creation ---
#         all_E = [d['p'][0] for item in data_to_plot for d in item['data']]
#         E_min, E_max = min(all_E), max(all_E)
#         common_bins = np.logspace(np.log10(E_min), np.log10(E_max), int(np.ceil((np.log10(E_max) - np.log10(E_min)) * bins_per_decade)))

#         fig, ax = plt.subplots(1, 1)
#         if weights_to_use is None:
#             weights_to_use = ['w_prod_scaled', 'w_angle', 'w_LLP_gone', 'w_Ecut', 'w_prim']

#         for item in data_to_plot:
#             rescale_factor = item['recipe']['rescale']**2*(gayy_factor or 1)**2
#             E_values = [d['p'][0] for d in item['data']]
#             weights = [np.prod([d.get(wk, 1.0) for wk in weights_to_use])*rescale_factor for d in item['data']]
#             ax.hist(E_values, bins=common_bins, weights=weights, histtype='step', linewidth=2.0, label=item['recipe']['label'])

#         # --- 5. Apply Final Formatting and Layout ---
#         ax.set_xlabel(r'Energy $E$ [GeV]')
#         ax.set_ylabel(r'Axions $\times f^4$ / POT $[{\rm GeV}^{4}]$')
#         ax.set_title(fr'Axion Flux for $m_a = {int(ma*1000)}$ MeV'+plot_title)
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         if xlim: ax.set_xlim(xlim)
#         if ylim: ax.set_ylim(ylim)

#         ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.8)
#         #ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray', alpha=0.4)

#         # Get the handles (the colored lines) and labels created by ax.hist()
#         handles, labels = ax.get_legend_handles_labels()


#         # Create the legend with the combined lists
#         legend = ax.legend(handles, labels,
#                            title=legend_title_text, # This is the user-provided title
#                            handletextpad=0.5,
#                            #bbox_to_anchor=(1.04, 0.5), 
#                            loc='upper center', 
#                            ncol=2)
#                            #borderaxespad=0.)

#         if legend_title_text:
#             plt.setp(legend.get_title(),multialignment='left')
#         set_size(w, h, ax=ax)
        
#         plot_folder_path = os.path.join(self.DATA_folder_path, folder_name)
        
#         if save_plot:
           
#             final_save_path = os.path.join(plot_folder_path, plot_filename)
            
#             try:
#                 fig.savefig(final_save_path, dpi=300, bbox_inches='tight')
#                 print(f"✅ Plot successfully saved to: {final_save_path}")
#             except Exception as e:
#                 print(f"❌ Error saving plot: {e}")
        
#         plt.show()
        
    def compute_and_save_sensitivities(self, masses=None, folder_name='PLOT', sens_folder_name='Sens', compute_combined=True, compute_separate=True):
        """
        Computes and saves sensitivity curves for logically grouped production channels.

        This method defines computational groups (e.g., Ann+Compton, Brem_el+Brem_pos)
        and calculates the sensitivity for each group across all specified masses, saving
        the result to a dedicated pickle file.

        Optionally, if `compute_combined` is True, it will perform a second loop to
        calculate the physically correct total sensitivity and save it to its own file.

        Args:
            masses (list):  A list of mass points to compute for. If None, all available masses will be discovered and used.
                            Defaults to None.
            folder_name (str, optional): The directory containing finalized data. Defaults to 'PLOT'.
            sens_folder_name (str, optional): The directory containing the computed sensitivities. Defaults to 'Sens'.
            compute_combined (bool, optional): If True, also computes and saves the combined
                                               sensitivity. Defaults to True.
            compute_separate (bool, optional): If True, also computes and saves the single channels
                                               sensitivity. Defaults to True.
        """
        print(f"\n> Computing and Saving Sensitivity Curves from '{folder_name}' folder...")
        plot_folder_path = os.path.join(self.DATA_folder_path, folder_name)

        if not os.path.exists(plot_folder_path):
            print(f"❌ ERROR: Finalized data folder not found at '{plot_folder_path}'.")
            return

        # --- Mass Discovery Block ---
        if masses is None:
            print("[Info] No masses provided. Discovering from data folders...")
            discovered_masses = []
            for entry in os.listdir(plot_folder_path):
                entry_path = os.path.join(plot_folder_path, entry)
                if os.path.isdir(entry_path):
                    try:
                        # If the folder name can be cast to a float, it's a mass folder
                        discovered_masses.append(float(entry))
                    except ValueError:
                        # Ignore folders that aren't valid mass numbers (e.g., 'Sens' itself)
                        continue
            
            if not discovered_masses:
                print(f"❌ ERROR: No valid mass folders found in '{plot_folder_path}'. Aborting.")
                return
            
            masses = sorted(discovered_masses)
            print(f"--> Found {len(masses)} masses to process.")
        
        if not masses:
            print("❌ ERROR: The list of masses to process is empty. Aborting.")
            return

        sensitivity_folder_path=self.DATA_folder_path+'/'+folder_name+'/'+sens_folder_name   
        print(f"   Sensitivity results will be saved in: {sensitivity_folder_path}")
        os.makedirs(sensitivity_folder_path, exist_ok=True)

        # Define the logical groups for computation. Each group will have its own output file.
        computation_groups = {
            'Ann_Comp': {
                'sources': ['Annihilation', 'Compton']
            },
            'Brem_Shower': {
                'sources': ['Brem_el', 'Brem_pos']
            },
            'Brem_primary': {
                'sources': ['Brem_primary']
            },
            'Primakoff_shower': {
                'sources': ['Primakoff_shower']
            },
            'Primakoff_primary': {
                'sources': ['Primakoff_primary']
            }
        }

        # --- Block 1: Individual and Grouped Process Calculations ---
        if compute_separate:    
            print("\n--- Calculating Individual and Grouped Channel Sensitivities ---")
            for group_name, group_info in computation_groups.items():
                group_results_with_info = {}
                found_any_data = False

                mass_iterator = sorted(masses)

                for ma in mass_iterator:
                    
                    axions_for_group = []
                    for source_name in group_info['sources']:
                        file_path = os.path.join(plot_folder_path, str(ma), f"{source_name}.pkl")
                        if os.path.exists(file_path):
                            found_any_data = True
                            axions = self._load_dict(file_path, VB=False)
                            if axions:
                                axions_for_group.extend(axions)

                    if not axions_for_group: continue

                    axions_for_calc = copy.deepcopy(axions_for_group)

                    result, approx_result = sensitivity_exact_fast(axions_for_calc, self.params_dict)
                    
                    group_results_with_info[ma] = (result, approx_result, len(axions_for_group))

                if found_any_data:
                    print(f"\nResults for Channel Group: {group_name}")
                    print("="*93)
                    print(f"{'Mass (GeV)':<15} {'Events Found':<18} {'gayy_inf':<15} {'LLP approx':<15} {'gayy_sup':<15} {'dAlambert approx':<15}")
                    print("-"*93)

                    for ma, (res, approx_res, n_events) in sorted(group_results_with_info.items()):
                        g_left_exact_str = f"{res[0]:.4g}" if res and res[0] is not None else "Not Found"
                        g_left_approx_str = f"{approx_res[0]:.4g}" if approx_res and approx_res[0] is not None else "N/A"
                        g_right_exact_str = f"{res[1]:.4g}" if res and res[1] is not None else "Not Found"
                        g_right_approx_str = f"{approx_res[1]:.4g}" if approx_res and approx_res[1] is not None else "N/A"

                        print(f"{ma:<15.4g} {n_events:<18,d} {g_left_exact_str:<15} {g_left_approx_str:<15} {g_right_exact_str:<15} {g_right_approx_str:<15}")
                    print("="*93)

                    output_filename = f"sensitivity_{group_name}.pkl"
                    output_path = os.path.join(sensitivity_folder_path, output_filename)
                    # Save only the result, not the extra event info
                    clean_results = {k: v[0] for k, v in group_results_with_info.items()}
                    self._save_dict(clean_results, output_path)
                    print(f"💾 Results saved to: {os.path.relpath(output_path)}\n")
                else:
                     print(f"\rNo data files found for group '{group_name}' across all specified masses.".ljust(80))

        # --- Block 2: Physically Combined Calculation ---
        if compute_combined:
            print("\n--- Calculating Physically Combined Sensitivity (Sum of All Processes) ---")
            
            all_processes_map = {p: info for group in computation_groups.values() for p, info in zip(group['sources'], [group]*len(group['sources']))}
            combined_results_with_info = {}
            
            mass_iterator_comb = sorted(masses)

            for ma in mass_iterator_comb:
                
                all_axions_combined = []
                for process_name, group_info in all_processes_map.items():
                    
                    # if 'primary' in process_name:
                    #     continue # Skip this iteration if the process is a primary one cause shower brem and primakoff already have primaries!

                    file_path = os.path.join(plot_folder_path, str(ma), f"{process_name}.pkl")
                    if os.path.exists(file_path):
                        axions = self._load_dict(file_path, VB=False)
                        if axions:
                            axions_temp = copy.deepcopy(axions)
                            all_axions_combined.extend(axions_temp)
                
                if not all_axions_combined: continue

                result, approx_result = sensitivity_exact_fast(all_axions_combined, self.params_dict)
                
                combined_results_with_info[ma] = (result, approx_result, len(all_axions_combined))
            
            if combined_results_with_info:
                print(f"\nResults for the Combined Sensitivity")
                print("="*93)
                print(f"{'Mass (GeV)':<15} {'Events Found':<18} {'gayy_inf':<15} {'LLP approx':<15} {'gayy_sup':<15} {'dAlambert approx':<15}")
                print("-"*93)

                for ma, (res, approx_res, n_events) in sorted(combined_results_with_info.items()):
                    g_left_exact_str = f"{res[0]:.4g}" if res and res[0] is not None else "Not Found"
                    g_left_approx_str = f"{approx_res[0]:.4g}" if approx_res and approx_res[0] is not None else "N/A"
                    g_right_exact_str = f"{res[1]:.4g}" if res and res[1] is not None else "Not Found"
                    g_right_approx_str = f"{approx_res[1]:.4g}" if approx_res and approx_res[1] is not None else "N/A"

                    print(f"{ma:<15.4g} {n_events:<18,d} {g_left_exact_str:<15} {g_left_approx_str:<15} {g_right_exact_str:<15} {g_right_approx_str:<15}")
                print("="*93)
                
                output_filename = "sensitivity_Combined.pkl"
                output_path = os.path.join(sensitivity_folder_path, output_filename)
                clean_results = {k: v[0] for k, v in combined_results_with_info.items()}
                self._save_dict(clean_results, output_path)
                print(f"💾 Combined results saved to: {os.path.relpath(output_path)}\n")
        
        print("\n✅ All requested sensitivity computations complete.\n")
        
  
 

    def plot_sensitivities(self, results_path, filenames=None,
                           plot_params_map=None,
                           y_rescale=None,y_label=r'$1/f$ [GeV$^{-1}$]',
                           title='ALPs Sensitivity', malim=None, ylim=None,
                            plot_filename=None,
                           legend_title_text=None,w=6,h=5):
        """
        Loads and plots sensitivity curves from various data formats with enhanced control.

        This upgraded function can handle:
        - Standard filled regions: {mass: [lower_bound, upper_bound]}
        - Single lower-bound lines: {mass: [lower_bound, None]}
        - Single upper-bound lines: {mass: [None, upper_bound]}
        - Complex ordered paths: {'x_values': [...], 'y_values': [...]}

        """
        # --- Block 1: File Discovery ---
        sensitivity_folder_path = os.path.join(self.DATA_folder_path, results_path)
        if not os.path.isdir(sensitivity_folder_path):
            print(f"❌ Error: Results directory not found at '{sensitivity_folder_path}'")
            return
        plot_params_map = plot_params_map or {} # Ensure plot_params_map is a dict
        
        if filenames is None:
            # If no explicit filenames are given, decide which files to plot based on plot_params_map.
            if plot_params_map:
                # If plot_params_map is NOT empty, only plot files corresponding to its keys.
                print(f"ℹ️ `plot_params_map` is provided. Plotting only specified processes: {list(plot_params_map.keys())}")
                filenames_to_plot = sorted([f"sensitivity_{key}.pkl" for key in plot_params_map.keys()])
            else:
                # If plot_params_map IS empty, revert to the original behavior: plot all .pkl files.
                print("ℹ️ `plot_params_map` is empty. Plotting all '.pkl' files found.")
                filenames_to_plot = sorted([f for f in os.listdir(sensitivity_folder_path) if f.endswith('.pkl')])
        else:
            # If a list of filenames is explicitly passed, use it (highest priority).
            filenames_to_plot = filenames

        # --- Block 2: Load Data ---
        couplings_data = {}
        for filename in filenames_to_plot:
            file_path = os.path.join(sensitivity_folder_path, filename)
            if not os.path.exists(file_path): continue
            process_name = filename.replace('sensitivity_', '').replace('.pkl', '')
            couplings_data[process_name] = self._load_dict(file_path)

        # --- Block 3: Plotting Logic ---
        fig=plt.figure(figsize=(w, h))
        plot_params_map = plot_params_map or {}
        default_colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
        lines_to_be_labeled = []
        label_x_pos=[]
        
        # Establish plotting order
        ordered_processes = list(plot_params_map.keys())
        remaining_processes = sorted([p for p in couplings_data.keys() if p not in ordered_processes])
        final_plot_order = ordered_processes + remaining_processes
        print(f"🎨 Plotting order: {final_plot_order}")

        # Main plotting loop
        for process_name in final_plot_order:
            if process_name not in couplings_data: continue
            
            data_dict = couplings_data[process_name]
            params = plot_params_map.get(process_name, {})
            
            if not params.get('plot', True): continue

            # Get styling parameters
            label = params.get('label', process_name.replace('_', ' '))
            color = params.get('color', next(default_colors))
            linestyle = params.get('linestyle', '-')
            marker = params.get('marker', None) # FIX: No more hardcoded markers
            markersize = params.get('markersize', None)
            zorder_line = params.get('zorder', 10)
            zorder_fill = params.get('zorder_fill', params.get('zorder', 2))
            ma_max_cutoff = params.get('ma_max')
            # --- Data Extraction and Plotting ---
            label_for_plot = label

            # Case 1: Handle complex, ordered paths (e.g., ALPINIST)
            if 'x_values' in data_dict and 'y_values' in data_dict:
                print(f"🎨 Plotting '{process_name}' as a continuous, ordered path.")
                x_data = data_dict['x_values']
                y_data = data_dict['y_values']
                line, = plt.plot(x_data, y_data, linestyle=linestyle, marker=marker, markersize=markersize,
                                 label=label, color=color, zorder=zorder_line)
                if params.get('named_line', False):
                    lines_to_be_labeled.append(line)
                    label_x_pos.append(params.get('label_x_pos',None))
                    
                continue # Go to the next file

            # Case 2: Handle standard regions and single lines (keyed by mass)
            all_ma_points = sorted([ma for ma, bounds in data_dict.items() if bounds and (bounds[0] is not None or bounds[1] is not None)])
            if not all_ma_points: continue

            ma_list_low = [ma for ma in all_ma_points if data_dict[ma][0] is not None and not np.isnan(data_dict[ma][0])]
            c_values_dwn = [data_dict[ma][0] for ma in ma_list_low]
            
            ma_list_up = [ma for ma in all_ma_points if data_dict[ma][1] is not None and not np.isnan(data_dict[ma][1])]
            c_values_up = [data_dict[ma][1] for ma in ma_list_up]
            
            # Apply ma_max truncation if specified
            if ma_max_cutoff is not None:
                print(f"✂️ Truncating '{process_name}' at ma = {ma_max_cutoff} GeV.")
                if ma_list_low:
                    filtered_low = [(ma, c) for ma, c in zip(ma_list_low, c_values_dwn) if ma <= ma_max_cutoff]
                    ma_list_low, c_values_dwn = zip(*filtered_low) if filtered_low else ([], [])
                if ma_list_up:
                    filtered_up = [(ma, c) for ma, c in zip(ma_list_up, c_values_up) if ma <= ma_max_cutoff]
                    ma_list_up, c_values_up = zip(*filtered_up) if filtered_up else ([], [])

            # Re-convert tuples from zip back to lists
            ma_list_low, c_values_dwn = list(ma_list_low), list(c_values_dwn)
            ma_list_up, c_values_up = list(ma_list_up), list(c_values_up)

            
            if params.get('close_curve', False):
                if ma_list_low and ma_list_up:
                    print(f"🎨 Plotting '{process_name}' as a closed curve (Robust Method).")
                    x_closed = ma_list_low + ma_list_up[::-1]
                    y_closed = c_values_dwn + c_values_up[::-1]
                    
                    line, = plt.plot(x_closed, y_closed, linestyle=linestyle, marker=marker, markersize=markersize, 
                                     label=label, color=color, zorder=zorder_line)
                    if params.get('named_line', False):
                        lines_to_be_labeled.append(line)
                        label_x_pos.append(params.get('label_x_pos',None))
                    if params.get('fill', True):
                        plt.fill(x_closed, y_closed, alpha=0.15, color=color, zorder=zorder_fill, label='_nolegend_')
                else:
                    print(f"⚠️ Warning: 'close_curve' requested for '{process_name}', but it does not have both upper and lower bounds. Skipping.")
                continue

            
            
            current_ylim = plt.ylim() # Get plot limits for filling
            if params.get('fill_above', False) and c_values_dwn:
                # Exclusion region ABOVE a lower bound line
                # ### --- CHANGE: EXPLICITLY HIDE FILL FROM LEGEND --- ###
                plt.fill_between(ma_list_low, c_values_dwn, current_ylim[1], 
                                 color=color, alpha=0.15, zorder=zorder_fill, linewidth=0, label='_nolegend_')

            elif params.get('fill_below', False) and c_values_up:
                # Region BELOW an upper bound line
                # ### --- CHANGE: EXPLICITLY HIDE FILL FROM LEGEND --- ###
                plt.fill_between(ma_list_up, c_values_up, current_ylim[0], 
                                 color=color, alpha=0.15, zorder=zorder_fill, linewidth=0, label='_nolegend_')
            
            elif params.get('fill', True):
                # Standard region fill BETWEEN two curves
                if ma_list_low and ma_list_up:
                    ma_for_fill = sorted(list(set(ma_list_low) & set(ma_list_up)))
                    c_dwn_map = {ma: c for ma, c in zip(ma_list_low, c_values_dwn)}
                    c_up_map = {ma: c for ma, c in zip(ma_list_up, c_values_up)}
                    c_dwn_for_fill = [c_dwn_map[ma] for ma in ma_for_fill]
                    c_up_for_fill = [c_up_map[ma] for ma in ma_for_fill]
                    # ### --- CHANGE: EXPLICITLY HIDE FILL FROM LEGEND --- ###
                    plt.fill_between(ma_for_fill, c_dwn_for_fill, c_up_for_fill, 
                                     alpha=0.15, color=color, zorder=zorder_fill, linewidth=0, label='_nolegend_')
            
            # --- Plotting the boundary lines (now fully controlled by params) ---
            if params.get('plot_low', True) and ma_list_low:
                line, = plt.plot(ma_list_low, c_values_dwn, linestyle=linestyle, marker=marker, markersize=markersize,
                         label=label_for_plot, color=color, zorder=zorder_line)
                if params.get('named_line', False):
                    lines_to_be_labeled.append(line)
                    label_x_pos.append(params.get('label_x_pos',None))
                label_for_plot = None
                
            if params.get('plot_up', True) and ma_list_up:
                line, = plt.plot(ma_list_up, c_values_up, linestyle=linestyle, marker=marker, markersize=markersize,
                         label=label_for_plot, color=color, zorder=zorder_line)
                # Only add if the lower line wasn't already added (label_for_plot is not None)
                if params.get('named_line', False) and label_for_plot is not None:
                     lines_to_be_labeled.append(line)
                     label_x_pos.append(params.get('label_x_pos',None))
        
        if lines_to_be_labeled:
            print(f"✍️ Placing labels directly on {len(lines_to_be_labeled)} curve(s).")
            labelLinesWithOptionalXvals(lines_to_be_labeled,xvals=label_x_pos,
                       align=False,
                       zorder=20,
                       outline_width=5,         
                       outline_color='white',
                       fontsize=14,             
                       fontweight='bold', 
                       alpha=0.8) 
            
        # --- Final Plotting Setup ---
        if y_rescale is not None: 
            _, new_label = y_rescale
            plt.ylabel(new_label)
        else:
            plt.ylabel(y_label)

        plt.xlabel(r'$m_a$ [GeV]')
        plt.xscale('log')
        plt.yscale('log')
        
        # This will apply the ylim from the function call, which is important for fill_above/below
        if ylim is not None:
            plt.ylim((ylim[0],ylim[1]))
        if malim is not None:
            plt.xlim(left=malim[0],right=malim[1])
            
        plt.title(title)
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        
        if not lines_to_be_labeled:  
            legend = plt.legend(title=legend_title_text, handletextpad=0.5, ncol=1, alignment='center')

            #Check if the legend actually has a title before trying to modify it
            if legend_title_text and legend.get_title():
                plt.setp(legend.get_title(), multialignment='left')

            # Also, check if the legend is empty and remove it if so.
            if not legend.get_texts() and not legend.get_legend_handles():
                legend.remove()
        
        if plot_filename is not None:
            parent_directory = os.path.dirname(sensitivity_folder_path) or '.'
            final_save_path = os.path.join(parent_directory, plot_filename)
            try:
                fig.savefig(final_save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot successfully saved to: {final_save_path}")
            except Exception as e:
                print(f"❌ Error saving plot: {e}")
        
        plt.show()
        
        
    def compute_total_flux(self, folder_name='PLOT', masses=None, weights_to_use=None):
        """
        Computes the total flux for each process across all specified masses.

        The flux for a given process is defined as the sum of the product of weights
        for every axion generated by that process. This method automatically discovers
        mass points if none are provided and aggregates the results into a nested
        dictionary.

        Args:
            folder_name (str, optional): The directory containing the finalized data, 
                                         typically 'PLOT'. Defaults to 'PLOT'.
            masses (list, optional): A list of mass points to compute for. If None, 
                                     all available masses will be discovered and used. 
                                     Defaults to None.
            weights_to_use (list of str, optional): List of weight keys to multiply. 
                                                     If None, a default list from the 
                                                     plotting function is used. 
                                                     Defaults to None.

        Returns:
            dict: A nested dictionary containing the computed flux for each process,
                  keyed by mass. Example: {mass1: {process1: flux1, ...}, ...}
        """
        print(f"\n> Computing total flux from finalized data in '{folder_name}'...")
        
        # --- 1. Setup Paths, Processes, and Weights ---
        data_path = os.path.join(self.DATA_folder_path, folder_name)
        if not os.path.isdir(data_path):
            print(f"[Error] Data directory not found: {data_path}")
            return {}

        process_filename_map = {
            'ann': 'Annihilation.pkl', 'comp': 'Compton.pkl', 'brem_el': 'Brem_el.pkl',
            'brem_pos': 'Brem_pos.pkl', 'primakoff_shower': 'Primakoff_shower.pkl',
            'brem_primary': 'Brem_primary.pkl', 'primakoff_primary': 'Primakoff_primary.pkl'
        }
        
        if weights_to_use is None:
            weights_to_use = ['w_prod_scaled', 'w_angle', 'w_LLP_gone', 'w_Ecut', 'w_prim']
            print(f"[Info] Using default weights: {weights_to_use}")

        # --- 2. Discover Masses if not Provided ---
        if masses is None:
            print("[Info] No masses provided. Discovering from data folders...")
            discovered_masses = []
            for entry in os.listdir(data_path):
                entry_path = os.path.join(data_path, entry)
                if os.path.isdir(entry_path):
                    try:
                        discovered_masses.append(float(entry))
                    except ValueError:
                        continue # Ignore non-numeric folder names
            
            if not discovered_masses:
                print(f"❌ ERROR: No valid mass folders found in '{data_path}'. Aborting.")
                return {}
            
            masses = sorted(discovered_masses)
            print(f"--> Found {len(masses)} masses to process: {masses}")

        # --- 3. Iterate and Compute Flux ---
        flux_results = {}
        for ma in masses:
            mass_dir_path = os.path.join(data_path, str(ma))
            flux_results[ma] = {}
            print(f"\n--- Processing Mass: {ma} GeV ---")

            for process_key, filename in process_filename_map.items():
                file_path = os.path.join(mass_dir_path, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        axion_list = pickle.load(f)
                    
                    if not axion_list:
                        print(f"  - {process_key:<18}: No axions found.")
                        continue
                        
                    # Calculate the sum of the product of weights for all axions
                    total_flux = sum(
                        np.prod([axion.get(w_key, 1.0) for w_key in weights_to_use])
                        for axion in axion_list
                    )
                    
                    flux_results[ma][process_key] = total_flux
                    print(f"  - {process_key:<18}: Flux = {total_flux:.4g}")

                except FileNotFoundError:
                    # This is expected if a process didn't generate axions for a given mass
                    continue
                except Exception as e:
                    print(f"  - {process_key:<18}: Error loading or processing file. {e}")

        print("\n✅ Flux computation complete.")
        return flux_results
    
    
        
###############################################        
###############################################
#LOGGER CLASS (Helper)
###############################################
###############################################

        
class Logger:
    def __init__(self, data_folder_path):
        os.makedirs(data_folder_path, exist_ok=True)
        self.log_file_path = os.path.join(data_folder_path, 'README.md')
        # This hidden file will store the last logged parameter state
        self.state_file_path = os.path.join(data_folder_path, '.log_state.pkl')

    def _load_log_state(self):
        """Loads the state object (the params_dict) from the file."""
        try:
            with open(self.state_file_path, 'rb') as f: return pickle.load(f)
        except (FileNotFoundError, EOFError): return None

    def _save_log_state(self, state_dict):
        """Saves the state object (the params_dict) to the file."""
        with open(self.state_file_path, 'wb') as f: pickle.dump(state_dict, f)
    
    
    def log_run(self, shower_instance, run_args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_logged_params = self._load_log_state()
        current_params = shower_instance.params_dict
        are_params_same = (current_params == last_logged_params)
        
        with open(self.log_file_path, 'a') as f:
            # --- Main Header for the Run ---
            f.write(f"## Run logged on: {timestamp}\n\n")

            # --- Section 1: Core AxionShower Setup ---
            f.write("### AxionShower Setup\n")
            f.write(f"- **Beam Type:** {shower_instance.primaries}\n")
            f.write(f"- **Shower Material:** `{shower_instance.shower_material}`\n")
            f.write(f"- **Masses in this Run:** `{run_args.get('masses')}`\n\n")

            # --- Section 2: Run Configuration ---
            f.write("### Run Configuration\n")
            f.write(f"- **RunID: ** '{run_args.get('run_ID')}'\n")
            f.write(f"- **Number of Primary Particles:** {len(shower_instance.beam)}\n")
            f.write(f"- **Command:** `run(mode='{run_args.get('mode')}', clean={run_args.get('clean')})`\n")
            f.write(f"- **Active Processes:** `{run_args.get('active_processes')}', primary_only={run_args.get('primary_only')}`\n")
            if run_args.get('masses_to_remove'):
                f.write(f"- **Masses Explicitly Removed:** `{run_args.get('masses_to_remove')}`\n\n")

            # --- Section 3: Conditional Parameter Logging ---
            f.write("### Experimental Parameters (`params_dict`)\n")
            if not are_params_same:
                print("[Logger] Parameters have changed since last log. Writing full table.")
                f.write("| Parameter      | Value                     |\n")
                f.write("|:---------------|:--------------------------|\n")
                for key, value in current_params.items():
                    value_str = f"{value:.4g}" if isinstance(value, float) else str(value)
                    f.write(f"| `{key}` | {value_str} |\n")
                # After printing, update the state file to this new state.
                self._save_log_state(current_params)
            else:
                # If they are the same, just write a note.
                f.write("*(Parameters unchanged since last logged event.)*\n")
            
            f.write("\n---\n\n")
        
        print(f"[Logger] Run details appended to {self.log_file_path}")

    
    def log_update(self, shower_instance, changes_dict, reweight_triggered, updated_masses=None):
        """
        Logs a parameter update event, explicitly stating which masses were affected.

        Args:
            shower_instance (AxionShower): The instance of AxionShower being updated.
            changes_dict (dict): A dictionary of the parameter changes.
            reweight_triggered (bool): True if the changes will cause an axion re-weight.
            updated_masses (list, optional): The specific list of masses being re-weighted.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file_path, 'a') as f:
            # --- Main Header for the Update ---
            f.write(f"## Update logged on: {timestamp}\n\n")

            # --- Section 1: Context ---
            f.write("### AxionShower Context\n")
            f.write(f"- **Beam Type:** {shower_instance.primaries}\n")
            f.write(f"- **Shower Material:** `{shower_instance.shower_material}`\n\n")

            # --- Section 2: The Parameter Changes ---
            f.write("### Parameter Changes\n")
            if not changes_dict:
                f.write("No parameter changes were made in this update call.\n")
            else:
                f.write("| Parameter      | Old Value       | New Value       |\n")
                f.write("|:---------------|:----------------|:----------------|\n")
                for key, (old_val, new_val) in changes_dict.items():
                    old_str = f"{old_val:.4g}" if isinstance(old_val, float) else "None" if old_val is None else str(old_val)
                    new_str = f"{new_val:.4g}" if isinstance(new_val, float) else str(new_val)
                    f.write(f"| `{key}` | {old_str} | {new_str} |\n")
            
             # --- Section 3: Conditional Parameter Logging ---
            f.write("\n### Experimental Parameters (`params_dict`)\n")
            if changes_dict:
                f.write("| Parameter      | Value                     |\n")
                f.write("|:---------------|:--------------------------|\n")
                for key, value in shower_instance.params_dict.items():
                    value_str = f"{value:.4g}" if isinstance(value, float) else str(value)
                    f.write(f"| `{key}` | {value_str} |\n")
            else:
                # If they are the same, just write a note.
                f.write("*(Parameters unchanged since last logged event.)*\n")

            # --- CRITICAL: Update the state file with the new parameters ---
            self._save_log_state(shower_instance.params_dict)

            f.write("\n---\n\n")
        
        print(f"[Logger] Update details appended to {self.log_file_path}")
        
################################
#SAMPLING PHOTONS FROM LIST
################################

def process_photons(input_filepath, output_filepath, num_to_select, energy_cutoff_gev, 
                    plot_histograms=False):
    """
    Loads photons, samples them, and optionally plots a comparison of the
    sample's distribution against the full input distribution's shape.
    """
    # --- 1. Load Data and Keep it as the Reference ---
    try:
        all_photons = np.loadtxt(input_filepath)
        print(f"Successfully loaded {len(all_photons)} total photons from '{input_filepath}'.")
    except FileNotFoundError:
        print(f"Error: The INPUT file '{input_filepath}' was not found.")
        return []
    
    # Extract full distributions to use as the reference shape
    all_energies = all_photons[:, 0]
    all_pts = np.sqrt(all_photons[:, 1]**2 + all_photons[:, 2]**2)
    # Filter out zeros for log plotting
    all_energies_gt0 = all_energies[all_energies > 0]
    all_pts_gt0 = all_pts[all_pts > 0]

    # --- 2. Filter and Sample ---
    energy_mask = all_photons[:, 0] > energy_cutoff_gev
    photons_above_cutoff = all_photons[energy_mask]
    num_over_cutoff = len(photons_above_cutoff)
    if num_over_cutoff == 0:
        print(f"Warning: No photons found > {energy_cutoff_gev} GeV.")
        return []

    if num_to_select > num_over_cutoff:
        print(f"Warning: Requested {num_to_select}, but only {num_over_cutoff} available. Selecting all.")
        num_to_select = num_over_cutoff
    
    selected_indices = np.random.choice(num_over_cutoff, size=num_to_select, replace=False)
    selected_photons = photons_above_cutoff[selected_indices]
    
    # --- 3. Create Particle Objects ---
    monte_carlo_weight = (num_over_cutoff / 1e5) * (1 - np.exp(-1))
    pbeam = [Particle(phot, [0, 0, 0], {'PID': 22, 'ID': 0, 'generation_number': 0, 'generation_process': 'Input', 'weight': monte_carlo_weight}) for phot in selected_photons]
    print(f"Randomly selected and created {len(pbeam)} particle objects with weight {monte_carlo_weight}.")
    
    # --- 4. Plotting: Sample vs. Full Dataset ---
    if plot_histograms and pbeam:
        print("Generating histograms: Sample vs. Full Dataset Shape")
        energies_sample = np.array([p.get_p0()[0] for p in pbeam])
        pts_sample = np.array([np.sqrt(p.get_p0()[1]**2+p.get_p0()[2]**2) for p in pbeam])

        energies_parent = photons_above_cutoff[:, 0]
        pts_parent = np.sqrt(photons_above_cutoff[:, 1]**2 + photons_above_cutoff[:, 2]**2)
        pts_parent = pts_parent[pts_parent > 0] # Filter for log scale

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Sample (N={len(pbeam)}) vs. Parent Population (N={num_over_cutoff})', fontsize=24)
        num_bins = 50

        # --- Energy Plot ---
        # Define bins based on the parent population's range
        bins_e = np.logspace(np.log10(energies_parent.min()), np.log10(energies_parent.max()), num_bins)
        
        ax1.hist(energies_sample, bins=bins_e, color='royalblue', alpha=0.8, label=f'Observed in Sample')
        
        # Get histogram of the PARENT POPULATION
        counts_parent_e, _ = np.histogram(energies_parent, bins=bins_e)
        
        # <-- KEY CHANGE: The correct scaling factor -->
        scale_factor = len(pbeam) / num_over_cutoff if num_over_cutoff > 0 else 0
        
        bin_centers_e = (bins_e[:-1] + bins_e[1:]) / 2
        # Plot the scaled line (Expected Counts)
        ax1.plot(bin_centers_e, counts_parent_e * scale_factor, 'o-', color='darkorange', label='Expected Counts (Scaled from Full)', drawstyle='steps-mid', markersize=4)

        ax1.set_title('Energy (E) Distribution', fontsize=22)
        ax1.set_xlabel('Energy [GeV]', fontsize=20)
        ax1.set_ylabel('Counts / bin', fontsize=20)
        ax1.set_xscale('log')
        ax1.set_ylim(bottom=0)
        ax1.grid(True, linestyle='--')
        ax1.legend()
        
        # --- pT Plot ---
        bins_pt = np.logspace(np.log10(all_pts_gt0.min()), np.log10(all_pts_gt0.max()), num_bins)
        ax2.hist(pts_sample, bins=bins_pt, color='seagreen', alpha=0.8, label=f'Observed Counts in Sample')
        counts_full_pt, _ = np.histogram(all_pts_gt0, bins=bins_pt)
        
        counts_parent_pt, _ = np.histogram(pts_parent, bins=bins_pt)
        
        bin_centers_pt = (bins_pt[:-1] + bins_pt[1:]) / 2
        # Use the same, correct scale factor
        ax2.plot(bin_centers_pt, counts_parent_pt * scale_factor, 'o-', color='crimson', label='Expected Counts (Scaled from Full)', drawstyle='steps-mid', markersize=4)

        ax2.set_title('Transverse Momentum ($p_T$) Distribution', fontsize=22)
        ax2.set_xlabel('$p_T$ [GeV]', fontsize=20)
        ax2.set_ylabel('Counts / bin', fontsize=20)
        ax2.set_xscale('log')
        ax2.set_ylim(bottom=0)
        ax2.grid(True, linestyle='--')
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # --- 5. Save and Return ---
    try:
        with open(output_filepath+'/pbeam.pkl', 'wb') as f:
            pickle.dump(pbeam, f)
        print(f"Successfully saved {len(pbeam)} particles to '{output_filepath}'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        
    return pbeam


def plot_flux_comparison(flux_dicts, labels, plot_type='line', 
                         masses_to_plot=None,
                         title=r'Axion Flux Comparison: Shower vs. Primary', 
                         save_path=None, 
                         w=6, h=4.5,
                         legend_title='Flux Source', ylim=None):
    """
    Plots a publication-quality comparison of aggregated "Shower" vs. "Primary" fluxes.

    This function processes flux dictionaries by aggregating fluxes into two categories:
    'Primary' (processes with 'primary' in the name) and 'Shower' (all others).
    It then plots the comparison using LaTeX rendering for a professional appearance.

    Args:
        flux_dicts (list of dict): A list of flux dictionaries from compute_total_flux.
        labels (list of str): A list of labels for each flux dictionary.
        plot_type (str, optional): The type of plot. Can be 'line' (default) or 'stacked_bar'.
        masses_to_plot (list, optional): A specific list of masses to include in the plot.
                                          If None, all available masses are used.
        title (str, optional): The title for the plot (can include LaTeX).
        save_path (str, optional): If provided, the plot is saved to this file path.
        w, h (float, optional): The target width and height of the plot in inches.
        use_latex (bool, optional): If True, use LaTeX for text rendering. Requires a
                                     LaTeX distribution (e.g., MiKTeX, TeX Live) to be installed.
        legend_title (str, optional): The title for the plot's legend.
    """
    if len(flux_dicts) != len(labels):
        raise ValueError("The number of flux dictionaries must match the number of labels.")

    # --- 1. Setup plot styling ---
    fig, ax = plt.subplots(figsize=(w, h))
    
    # --- 2. Process data for each input dictionary (same as before) ---
    processed_data = []
    all_masses_in_scope = set()
    
    col=['C0','C1']
    for i, flux_dict in enumerate(flux_dicts):
        aggregated_flux = {'masses': [], 'primary': [], 'shower': []}
        mass_iterator = sorted(masses_to_plot) if masses_to_plot is not None else sorted(flux_dict.keys())

        for ma in mass_iterator:
            if ma not in flux_dict:
                if masses_to_plot: # Only warn if the user explicitly asked for a mass
                    print(f"[Warning] Mass {ma} not found in the data for '{labels[i]}'. Skipping.")
                continue
                
            all_masses_in_scope.add(ma)
            process_fluxes = flux_dict[ma]
            total_primary_flux = sum(flux for key, flux in process_fluxes.items() if 'primary' in key)
            total_shower_flux = sum(flux for key, flux in process_fluxes.items() if 'primary' not in key)
            
            aggregated_flux['masses'].append(ma)
            aggregated_flux['primary'].append(total_primary_flux)
            aggregated_flux['shower'].append(total_shower_flux)
        
        processed_data.append(aggregated_flux)

    if not all_masses_in_scope:
        print("[Error] No valid mass data to plot after filtering. Aborting plot.")
        plt.close(fig)
        return

    # --- 3. Generate the plot  ---
    if plot_type == 'line':
        for i, data in enumerate(processed_data):
            if not data['masses']: continue
            label_prefix = f"{labels[i]} - " if len(labels) > 1 else ""
            ax.plot(data['masses'], data['primary'], marker='o', linestyle='-', label=f'{label_prefix}Primary',c=col[i])
            ax.plot(data['masses'], data['shower'], marker='x', linestyle='--', label=f'{label_prefix}Shower',c=col[i])
        ax.set_xlabel(r'$m_a$ [GeV]')
        ax.set_xscale('log')

    elif plot_type == 'stacked_bar':
        plot_masses = sorted(list(all_masses_in_scope))
        num_setups = len(processed_data)
        x = np.arange(len(plot_masses))
        width = 0.8 / num_setups
        for i, data in enumerate(processed_data):
            offset = width * i
            bar_positions = x - (0.8 / 2) + offset + (width / 2)
            mass_to_primary_flux = dict(zip(data['masses'], data['primary']))
            mass_to_shower_flux = dict(zip(data['masses'], data['shower']))
            primary_fluxes_ordered = [mass_to_primary_flux.get(m, 0) for m in plot_masses]
            shower_fluxes_ordered = [mass_to_shower_flux.get(m, 0) for m in plot_masses]
            label_prefix = f"{labels[i]} - " if len(labels) > 1 else ""
            ax.bar(bar_positions, shower_fluxes_ordered, width, label=f'{label_prefix}Shower')
            ax.bar(bar_positions, primary_fluxes_ordered, width, bottom=shower_fluxes_ordered, label=f'{label_prefix}Primary')
        ax.set_xlabel(r'$m_a$ [GeV]')
        ax.set_xticks(x)
        ax.set_xticklabels([f'${m}$' for m in plot_masses]) # LaTeX format for tick labels

    else:
        raise ValueError(f"Invalid plot_type: '{plot_type}'. Choose 'line' or 'stacked_bar'.")

    # --- 4. Final plot formatting with professional styling ---
    ax.set_ylabel(r'Axions $\times f^4$ / POT $[{\rm GeV}^{4}]$')
    ax.set_title(title)
    ax.set_yscale('log')
    if ylim: ax.set_ylim(ylim)
    
    
    # Apply specific grid style from the reference plot
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.8)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4, axis='y')
    ax.legend(title=legend_title,  handletextpad=0.5, ncol=2, fancybox=True, loc='upper center')
    
    plt.tight_layout()

    if save_path:
        print(f"✅ Saving plot to: {save_path}")
        # Add bbox_inches='tight' for robust saving
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    

    
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle
import os

def _load_and_validate_csv(csv_path, x_col, y_col, encoding):
    """
    Internal helper function to robustly load and validate a single CSV file.
    It handles file existence, column names, whitespace, and encoding.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at '{csv_path}'")
    
    # The encoding parameter is used here to prevent decoding errors
    df = pd.read_csv(csv_path, encoding=encoding)
    
    # Strip whitespace from column names (e.g., ' coupling' -> 'coupling')
    df.columns = df.columns.str.strip()
    
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV '{csv_path}' is missing required columns '{x_col}' or '{y_col}'. "
                         f"Available columns: {list(df.columns)}")
    
    # Sort by x-axis and remove duplicates to ensure valid interpolation
    return df.sort_values(by=x_col).drop_duplicates(subset=[x_col])


def process_boundaries_to_dict(output_path,
                               lower_csv_path=None,
                               upper_csv_path=None,
                               encoding='utf-8',
                               num_points=500,
                               y_scale_factor=1.0,
                               y_transform=None,          # <─ NEW
                               csv_x_col='mass',
                               csv_y_col='coupling',
                               save_file=True,
                               verbose=True):
    """
    Reads one or two CSV files for boundaries, interpolates them onto a smart
    grid, optionally rescales or arbitrarily transforms the y–values, and
    returns / saves a mass → [lower, upper] dictionary.

    Parameters
    ----------
    …
    y_scale_factor : float or callable, optional
        Old behaviour – multiply all y by this factor (or by
        y_scale_factor(x) if it is a function of x).
    y_transform : callable, optional
        New behaviour – arbitrary transformation.  Must accept
        y_new = y_transform(x, y_old).  Ignored if None.
    """

    if not lower_csv_path and not upper_csv_path:
        raise ValueError("At least one boundary csv must be given.")

    # ------------------------------------------------------------------ helpers
    def apply_y(x, y_old):
        """
        Decide which y modification to apply.
        Priority   1) y_transform(x, y_old)
                   2) y_old * y_scale_factor(x)   (callable)
                   3) y_old * y_scale_factor      (float/int)
        """
        if y_old is None:
            return None
        if callable(y_transform):
            return float(y_transform(x, float(y_old)))
        if callable(y_scale_factor):
            return float(y_old) * float(y_scale_factor(x))
        return float(y_old) * float(y_scale_factor)

    # ------------------------------------------------------------------ loading
    try:
        interp_lower = interp_upper = None
        all_original_x = np.array([])

        if lower_csv_path:
            if verbose: print(f"📄 Loading lower boundary from '{lower_csv_path}'...")
            df_lower = _load_and_validate_csv(lower_csv_path, csv_x_col, csv_y_col, encoding)
            interp_lower = interp1d(df_lower[csv_x_col], df_lower[csv_y_col],
                                    bounds_error=False, fill_value=None)
            all_original_x = np.union1d(all_original_x, df_lower[csv_x_col].unique())

        if upper_csv_path:
            if verbose: print(f"📄 Loading upper boundary from '{upper_csv_path}'...")
            df_upper = _load_and_validate_csv(upper_csv_path, csv_x_col, csv_y_col, encoding)
            interp_upper = interp1d(df_upper[csv_x_col], df_upper[csv_y_col],
                                    bounds_error=False, fill_value=None)
            all_original_x = np.union1d(all_original_x, df_upper[csv_x_col].unique())

        if verbose: print("✅ Data loaded successfully.")

        # ------------------------------------------------------------------ grid
        min_mass, max_mass = all_original_x.min(), all_original_x.max()
        log_grid = np.logspace(np.log10(min_mass), np.log10(max_mass), num_points)
        final_mass_grid = np.union1d(all_original_x, log_grid)
        if verbose: print(f"✨ Masterful grid created with {len(final_mass_grid)} points.")

        # ----------------------------------------------------------- dictionary
        sensitivity_dict = {}
        for mass in final_mass_grid:
            lower_raw = interp_lower(mass) if interp_lower is not None else None
            upper_raw = interp_upper(mass) if interp_upper is not None else None

            if lower_raw is None and upper_raw is None:
                continue  # skip empty entries

            scaled_lower = apply_y(mass, lower_raw)
            scaled_upper = apply_y(mass, upper_raw)

            sensitivity_dict[float(mass)] = [scaled_lower, scaled_upper]

        if verbose: print(f"✅ Generated dictionary with {len(sensitivity_dict)} points.")

        # --------------------------------------------------------------- saving
        if save_file:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as fh:
                pickle.dump(sensitivity_dict, fh)
            if verbose: print(f"💾 Dictionary saved to '{output_path}'.")

        return sensitivity_dict

    except (FileNotFoundError, ValueError, Exception) as e:
        if verbose: print(f"❌ AN ERROR OCCURRED: {e}")
        return None
    
    
def labelLinesWithOptionalXvals(lines, xvals=None, **kwargs):
    """
    Wrapper for labellines.labelLines that allows None in xvals
    to mean 'auto-place this label'.
    
    Parameters
    ----------
    lines : list of matplotlib lines
    xvals : list of floats or None
        Same length as lines. If element is a float, it fixes
        label at that x. If None, automatic placement is used.
    kwargs : passed to labellines.labelLines
    """
    if xvals is None:
        # normal usage: all automatic
        return labelLines(lines, **kwargs)
    
    # filter out the manual ones
    manual_lines = []
    manual_xvals = []
    auto_lines = []
    
    for line, xv in zip(lines, xvals):
        if xv is None:
            auto_lines.append(line)
        else:
            manual_lines.append(line)
            manual_xvals.append(xv)
    
    # call once for manual positions
    if manual_lines:
        labelLines(manual_lines, xvals=manual_xvals, **kwargs)
    # call again for automatic placement of remaining
    if auto_lines:
        labelLines(auto_lines, **kwargs)