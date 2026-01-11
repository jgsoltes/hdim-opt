import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# constants
epsilon = 1e-12 # to avoid mathematical singularities

# visualization parameters
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['lines.color'] = 'white'

def apply_noise(signal, noise):
    # adding random gaussian noise within 1% of discrete amplitudes
    stdev = np.std(np.abs(signal))
    signal_noisy = signal + noise * np.random.normal(loc=0, scale=stdev, size=signal.shape)

    return signal_noisy

def e1_waveform(peak_Vm=50e3, rise_s=5e-9, decay_s=200e-9, 
                              sample_Hz=10e9, duration_s=1e-6, noise=0.0):
    '''
    Objective:
        - Generates time-domain double exponential H-EMP E1 waveform, per MIL-STD / IEC specifications.
        - Pulse waveform: E(t) = E0 * k * (exp(-alpha*t) - exp(-beta*t))

    Inputs:
        - E_peak_V_m: Target EMP amplitude (V/m).
        - E_rise_time_ns: Target EMP risetime (ns).
        - E_decay_time_ns: Target decay time of the pulse (FWHM).
        - sampling_rate: Sampling rate of pulse (Hz).
        - duration_ns: Duration of pulse (ns).
        
    Outputs:
        - time_s: Time array of pulse (s).
        - E_t: Electric field amplitude array at each time (V/m).
        
    '''
    
    dt_s = 1 / sample_Hz
    time_s = np.arange(0, duration_s, dt_s)

    # alpha controls the decay and broadband shape
    # input E_decay_time_ns = desired pulse FWHM
    alpha = np.log(2) / decay_s # derived from from FWHM = ln(2) / FWHM

    # beta controls the rise time and high-frequency content
    beta = 2.0035 / rise_s # # approximate relationship: beta ~= 2.2 / (rise_time) (in seconds^-1)

    # calculate time of pulse peak for k_norm calculation
    t_peak_s = np.log(beta / alpha) / (beta - alpha)

    # 'k_norm' factor normalizes pulse peak to input E_peak_V_m (50,000 V/m)
    denominator = (np.exp(-alpha * t_peak_s) - np.exp(-beta * t_peak_s))
    if np.isclose(denominator, 0):
        k_norm = peak_Vm 
    else:
        k_norm = peak_Vm / denominator

    # generate the E1 waveform
    E_t = k_norm * (np.exp(-alpha * time_s) - np.exp(-beta * time_s))
    
    # adding random gaussian noise
    E_t = apply_noise(E_t, noise)
    
    return time_s, E_t
    
def e2_waveform(E_peak=100, tr_us=1.5, tf_ms=1.0, sample_rate=1e6, duration_s=0.01, noise=0.0):
    '''Generates E2 H-EMP pulse waveform (lightning-like).'''
    t = np.arange(0, duration_s, 1/sample_rate)
    alpha = 1 / (tf_ms * 1e-3)
    beta = 1 / (tr_us * 1e-6)
    E_t = E_peak * 1.1 * (np.exp(-alpha * t) - np.exp(-beta * t))
    
    # adding random gaussian noise
    E_t = apply_noise(E_t, noise)
    
    return t, E_t

def e3_waveform(E_peak=40, t_peak_s=10, sample_rate=10, duration_s=500, noise=0.0):
    '''Generates E3 H-EMP pulse waveform (geostorm-like; magnetohydrodynamic).'''
    t = np.arange(0, duration_s, 1/sample_rate)
    
    # simplified IEC 61000-2-9 E3 waveform
    E_t = E_peak * (np.exp(-t/120) - np.exp(-t/20)) 
    
    # adding random gaussian noise
    E_t = apply_noise(E_t, noise)
    
    return t, E_t

def calculate_rise_time(time_array, pulse_array):
    '''
    Objective:
        - Calculates the 10%-90% rise time of a double exponential pulse waveform.

    Inputs:
        - time_array: Pulse time array (s).
        - pulse_array: Pulse waveform array (V/m).

    Outputs: 
        - rise_time: Rise time (10%-90%) of pulse waveform (ns).
        - t_90_percent: Time at rising 90% peak amplitude (ns).
        - t_10_percent: Time at rising 10% amplitude (ns).
        
    '''
    
    peak_amplitude = np.max(pulse_array)
    
    # calculate 10% and 90% thresholds
    threshold_10_percent = 0.1 * peak_amplitude
    threshold_90_percent = 0.9 * peak_amplitude

    # find indices where the pulse first crosses the 10% threshold (rising side)
    idx_10 = np.where(pulse_array >= threshold_10_percent)[0]
    if len(idx_10) == 0:
        return None
    idx_10_first = idx_10[0]

    # find indices where the pulse first crosses the 90% threshold (rising side)
    idx_90 = np.where(pulse_array >= threshold_90_percent)[0]
    if len(idx_90) == 0:
        return None
    idx_90_first = idx_90[0]

    # interpolate to find the exact time points at 10% and 90% thresholds
    # time at 10% threshold
    t_10_percent = time_array[idx_10_first-1] + (threshold_10_percent - pulse_array[idx_10_first-1]) * \
                   (time_array[idx_10_first] - time_array[idx_10_first-1]) / \
                   (pulse_array[idx_10_first] - pulse_array[idx_10_first-1])
    
    # time at 90% threshold
    t_90_percent = time_array[idx_90_first-1] + (threshold_90_percent - pulse_array[idx_90_first-1]) * \
                   (time_array[idx_90_first] - time_array[idx_90_first-1]) / \
                   (pulse_array[idx_90_first] - pulse_array[idx_90_first-1])

    # calculate risetime
    rise_time = t_90_percent - t_10_percent
    
    return rise_time, t_90_percent, t_10_percent
    
def calculate_fwhm(time_array, pulse_array):
    '''
    Objective:
        - Calculates the full width at half maximum (FWHM) of a double exponential pulse waveform.

    Inputs: 
        - time_array: Pulse time array (s).
        - pulse_array: Pulse waveform array (V/m).

    Outputs:
        - fwhm: Full-width half-max of pulse waveform (ns).
        - t_fwhm2: Time at rising half-max (ns).
        - t_fwhm1: Time at decaying half-max (ns).
        
    '''
    
    # find the peak value of the pulse
    peak_amplitude = np.max(pulse_array)
    half_max = peak_amplitude / 2.0

    # find indices where the pulse is above half_max
    indices_above_half_max = np.where(pulse_array >= half_max)[0]

    # find the first and last points where the pulse crosses half_max
    idx1 = indices_above_half_max[0]
    idx2 = indices_above_half_max[-1]

    # first FWHM crossing point (rising side)
    if idx1 == 0: # pulse starts above half_max
        t_fwhm1 = time_array[idx1]
    else:
        # interpolate to find first time point at half_max
        t_fwhm1 = time_array[idx1-1] + (half_max - pulse_array[idx1-1]) * \
                  (time_array[idx1] - time_array[idx1-1]) / \
                  (pulse_array[idx1] - pulse_array[idx1-1])

    # second FWHM crossing point (decaying side)
    if idx2 == len(pulse_array) - 1: # pulse ends above half_max
        t_fwhm2 = time_array[idx2]
    else:
        # interpolate to find second time point at half_max
        t_fwhm2 = time_array[idx2] + (half_max - pulse_array[idx2]) * \
                  (time_array[idx2+1] - time_array[idx2]) / \
                  (pulse_array[idx2+1] - pulse_array[idx2])

    # calculate FWHM
    fwhm = t_fwhm2 - t_fwhm1
    
    return fwhm, t_fwhm2, t_fwhm1

def apply_shielding(f, shielding_dB, rolloff_hf=500e6, rolloff_lf=1e3):
    '''
    Combines complex transfer function math with real-world 
    LF (Magnetic) and HF (Leakage) rolloff physics.
    '''
    # base linear gain
    base_gain = 10**(-shielding_dB/20) 

    # HF rolloff
    h_hf = 1 / (1 + 1j * (f / rolloff_hf))

    # LF rolloff
    h_lf = 1 / (1 + (rolloff_lf / (f + 1e-12)))

    # total complex shielding function
    h_total = base_gain * h_hf * h_lf
    
    return h_total
    
def analyze_waveform(x=None, y=None, sample_rate=None, domain='time', method='complex', 
                     tf_function=None, tf_kwargs=None, noise=0.0, verbose=True):
    '''
    Decomposes & analyzes the given signal waveform.
    Outputs:
        - df_time: DataFrame of time-domain data (N rows)
        - df_freq: DataFrame of positive frequency-domain data (N/2+1 rows)
        - metrics: Dictionary of scalar results
    '''

    # clean
    x = np.array(x)
    y = np.array(y).flatten()
    domain = 'time' if domain.lower() in ['t', 'time'] else 'freq'
    
    # apply transfer function
    if tf_function:
        y = apply_transfer_function(x, y, tf_function, domain=domain, **(tf_kwargs or {}))
    
    # apply gaussian noise
    y = apply_noise(y, noise)

    # extract parameters
    if domain == 'time':
        n = len(y)
        fs = 1 / (x[1] - x[0])
        if method == 'complex':
            y_f = np.fft.fft(y) / n
            freqs = np.fft.fftfreq(n, 1/fs)
        elif method == 'real':
            y_f = np.fft.rfft(y) / n
            freqs = np.fft.rfftfreq(n, 1/fs)

        y_t, t = y, x
    else:
        n = len(y)
        fs = sample_rate if sample_rate else x[len(x)//2] * 2
        if method == 'complex':
            y_t = np.fft.ifft(y).real * n
        elif method == 'real':
            y_t = np.fft.irfft(y).real * n
        t = np.arange(0, n) / fs
        y_f, freqs = y, x

    # positive half of frequencies
    mask = freqs >= 0
    f_pos = freqs[mask]
    yf_pos = y_f[mask]

    # energy calculation
    esd = (np.abs(yf_pos)**2) * (2 / fs)
    
    # handle DC
    esd = (np.abs(yf_pos)**2) * (2 / fs)
    esd[0] = esd[0] / 2  # DC only exists once
    
    # if real, last bin (nyquist) only exists once
    if method == 'real' and len(esd) > 0:
        esd[-1] = esd[-1] / 2
        
    df_freq = f_pos[1] - f_pos[0]
    cumul_energy = np.cumsum(esd) * df_freq
    total_energy = cumul_energy[-1]

    # convert to dataframes:
    # time domain
    temporal = {
        'time_s': t,
        'amplitude': y_t
    }

    # frequency domain
    spectral = {
        'freq': f_pos,
        'signal': yf_pos,
        'esd': esd,
        'energy': cumul_energy
    }

    # max gradient
    dv_dt = np.diff(y_t) * fs
    
    # action integral
    action_integral = np.trapezoid(y_t**2, t)
    
    # calculate rise time/decay time
    rise_s, _, _ = calculate_rise_time(t, np.abs(y_t))
    rise_ns = 1e9 * rise_s
    fwhm_s, _, _ = calculate_fwhm(t, np.abs(y_t))
    fwhm_ns = 1e9 * fwhm_s
    
    # scalar metrics
    metrics = {
        'peak_t': np.max(np.abs(y_t)),
        'peak_f': np.max(np.abs(yf_pos)),
        'total_energy': total_energy,
        'action_integral': action_integral,
        'max_dv_dt': np.max(np.abs(dv_dt)),
        'bw_90_hz': f_pos[np.where(cumul_energy >= 0.9 * total_energy)[0][0]],
        'center_freq_hz': np.sum(f_pos * esd) / (total_energy + 1e-12),
        'papr_db': 10 * np.log10(np.max(y_t**2) / np.mean(y_t**2)),
        'sample_rate': fs,
        'rise90_ns': rise_ns,
        'fwhm_ns': fwhm_ns,
    }
    
    # plot
    if verbose:
        plot_diagnostic_dashboard(temporal, spectral, metrics)

    return temporal, spectral, metrics

def apply_transfer_function(x, y, tf_function, domain='freq', **kwargs):
    '''
    x: Time or frequency array.
    y: Input signal.
    tf_function: The transfer function.
    domain: 'freq' or 'time'.
    **kwargs: Arguments passed to the transfer function.
    '''

    if domain == 'freq':
        H_f = tf_function(np.abs(x), **kwargs)
        output = y * H_f
    elif domain == 'time':
        h_t = tf_function(x, **kwargs) 
        output = np.convolve(y, h_t, mode='same')
        output *= (x[1] - x[0]) # normalize by dt
    else:
        raise ValueError('Unrecognized signal domain.')
        
    return output

def plot_diagnostic_dashboard(temporal, spectral, metrics):
    '''
    6-plot diagnostic dashboard:
    Time Domain | Frequency Magnitude
    Phase Spectrum | Energy Spectral Density (ESD)
    Cumulative Energy | Spectrogram
    '''

    # 3 rows, 2 columns
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax_time = fig.add_subplot(gs[0, 0])
    ax_freq = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_esd  = fig.add_subplot(gs[1, 1])
    ax_cum  = fig.add_subplot(gs[2, 0])
    ax_spec = fig.add_subplot(gs[2, 1])

    # mask for positive frequencies to ensure dimensions match
    pos_mask = spectral['freq'] >= 0
    f_mhz = spectral['freq'][pos_mask] / 1e6
    mag_pos = np.abs(spectral['signal'])[pos_mask]

    # time domain
    ax_time.plot(temporal['time_s']*1e6, temporal['amplitude'], color='cyan', label=f'Peak: {metrics['peak_t']:.2f}')
    ax_time.set_title(f'Time-Domain Signal')
    ax_time.set_xlabel(r'Time ($\mu s$)')
    ax_time.set_ylabel('Amplitude')
    ax_time.legend()

    # frequency domain
    ax_freq.semilogy(f_mhz, np.abs(spectral['signal']), color='violet')
    ax_freq.set_title('Frequency-Domain Signal')
    ax_freq.set_xlabel('Frequency (MHz)')
    max_f = metrics['sample_rate'] / 2e6 # limit no higher than nyquist
    ax_freq.set_xlim(0, min(1000, max_f))
    ax_freq.grid(alpha=0.2, which='both')

    # phase spectrum
    ax_phase.plot(f_mhz, np.angle(spectral['signal']), color='limegreen', linewidth=0.5)
    ax_phase.set_title('Phase Spectrum')
    ax_phase.set_xlabel('Frequency (MHz)')
    ax_phase.set_ylabel('Phase (rad)')
    ax_phase.set_xlim(ax_freq.get_xlim())

    # energy spectral density (ESD)
    ax_esd.semilogy(f_mhz, spectral['esd'], color='gold')
    ax_esd.set_title('Energy Spectral Density (ESD)')
    ax_esd.set_ylabel(r'$V^2 \cdot s / Hz$')
    ax_esd.set_xlabel('Frequency (MHz)')
    ax_esd.set_xlim(ax_freq.get_xlim())
    ax_esd.set_xlim(0, min(1000, max_f))

    # cumulative energy distribution
    ax_cum.plot(f_mhz, spectral['energy'], color='gold', linewidth=2)
    ax_cum.fill_between(f_mhz, spectral['energy'], color='gold', alpha=0.2)
    ax_cum.axvline(metrics['bw_90_hz'], color='red', linestyle='--', 
                   label=f'90% Band: {metrics['bw_90_hz']:.1f} MHz')
    ax_cum.set_title('Cumulative Energy')
    ax_cum.set_ylabel('Normalized Energy')
    ax_cum.set_xlabel('Frequency (MHz)')
    ax_cum.set_xlim(ax_freq.get_xlim())
    ax_cum.legend(fontsize='small')

    # spectrogram
    # ensure amplitude is real for spectrogram
    sample_rate = metrics['sample_rate']
    y_signal = np.real(temporal['amplitude'])
    window_duration_s = 0.01*(temporal['time_s'].max() - temporal['time_s'].min())
    nperseg = int(window_duration_s * sample_rate)
    
    raw_n = (1/1.15) * (window_duration_s * sample_rate)
    
    # powers of 2
    if raw_n <= 1:
        nperseg = 16 # minimum window
    else:
        # round to nearest power of 2
        nperseg = int(2**np.round(np.log2(raw_n)))
        
        # reinforce minimum
        nperseg = max(nperseg, 16)
    
    # nonoverlap always less than num per segment
    noverlap = nperseg // 2
    f, t_spec, Sxx = signal.spectrogram(y_signal, fs=sample_rate, window='hann', 
                                        nperseg=nperseg, noverlap=nperseg//2)
    
    im = ax_spec.pcolormesh(t_spec*1e6, f, 10*np.log10(Sxx + epsilon), 
                            shading='gouraud', cmap='plasma')
    
    ax_spec.set_yscale('log')
    ax_spec.set_ylim(np.abs(spectral['freq']).min()+epsilon, np.abs(spectral['freq'].max()/sample_rate))

    plt.show()