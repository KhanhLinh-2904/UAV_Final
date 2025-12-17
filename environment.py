import math
import numpy as np
from gymnasium.utils import seeding
from gymnasium import spaces

class RadioEnvironment:
    def __init__(self, seed=None):
        super().__init__()
        
        # --- 1. (NETWORK PHYSICS) ---
        self.fc_hz = 4.7e9           
        self.cell_radius = 150.0   
        self.inter_site_distance = 225.0
        
        # (Base Stations) ****
        self.x_bs_1, self.y_bs_1 = 0, 0
        self.x_bs_2, self.y_bs_2 = self.inter_site_distance, 0

        # --- 2. power
        self.max_tx_power = 46.0        
        self.tx_power_min = 1.0         # 
        self.max_tx_power_interference = 46.0
        
        self.sinr_target = 3.0          #  (0 dB)
        self.sinr_min = 0.0            
        self.sinr_max = 16.0            # (16 dB)
        
        self.prob_LOS = 0.9             #  (Line-of-Sight)
        self.gain_no_beamforming = 11.0 #  Beamforming (dBi)

        # --- 3. BEAMFORMING ---
        self.use_beamforming = False
        self.num_antennas = 1           # num_antennas (1 = SISO)
        self.oversample_factor = 1      # k_oversample
        self.Np = 15
        
        #  Beamforming (Codebook)
        self._init_beamforming_codebook()

        # --- 4. RL (REWARD & SPACES) ---
        self.reward_min = -20 
        self.reward_max = 100
        self.num_actions = 16               # 16 actions
        
        self._init_gym_spaces()
        
        # --- 5. initial state ---
        self.seed(seed)
        self.state = None

    def _init_beamforming_codebook(self):
        """(Steering Vectors)"""
      
        total_beams = self.oversample_factor * self.num_antennas
        
        self.F = np.zeros([self.num_antennas, total_beams], dtype=complex)
        
        self.theta_n = math.pi * np.linspace(0, 1, total_beams, endpoint=False)
        
        for n in range(total_beams):
            self.F[:, n] = self._compute_bf_vector(self.theta_n[n])
 
        self.f_n_bs1 = None
        self.f_n_bs2 = None

    def _init_gym_spaces(self):
       
        # Action Space
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation Space:
        # State: [x_ue1, y_ue1, x_ue2, y_ue2, Power_BS1, Power_BS2]
        
    
        low_bounds = np.array([
            -self.cell_radius,              # x1
            -self.cell_radius,              # y1
            self.inter_site_distance - self.cell_radius,    # x2 
            -self.cell_radius,              # y2
            1.0,                            # Power 1 Min
            1.0                             # Power 2 Min
        ], dtype=np.float32)

       
        high_bounds = np.array([
            self.cell_radius,               # x1
            self.cell_radius,               # y1
            self.inter_site_distance + self.cell_radius,    # x2
            self.cell_radius,               # y2
            self.max_tx_power,              # Power 1 Max
            self.max_tx_power               # Power 2 Max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low_bounds, high_bounds, dtype=np.float32)
        
    def seed(self, seed_value=None):
        self.np_random, used_seed = seeding.np_random(seed_value)
        return [used_seed]

    def _path_loss_mmWave(self, x_ue, y_ue, path_loss_exponent, x_bs=0, y_bs=0):
        
        """
        Path loss for mmWave 28GHz
        FSPL + path loss gain for beamforming + shadowing
        """
        #parameters from paper about channel model
        C_LIGHT = 3e8
        wavelength = C_LIGHT / self.fc_hz
        
        BF_GAIN_FACTOR = 0.0671
        SHADOWING_STD = 9.1
        num_antennas = self.num_antennas
        
        #distance
        dist = math.sqrt((x_ue - x_bs)**2 + (y_ue - y_bs)**2)
        dist = max(dist, 1.0)
        
        #fspl for vacuum environment
        fspl_db = 20 * np.log10((4 * math.pi * dist) / wavelength)
        
        #plus PLE (Path Loss Exponent), but reduced by Beamforming gain
        beamforming_gain_correction = (1 - BF_GAIN_FACTOR * np.log2(num_antennas))
        env_loss_db = 10 * path_loss_exponent * np.log10(dist) * beamforming_gain_correction
        #shadowing by houses, trees, ... objects
        shadowing_loss_db = self.np_random.normal(0, SHADOWING_STD)
        total_path_loss_db = fspl_db + env_loss_db + shadowing_loss_db
        
        return total_path_loss_db
    
    def _path_loss_sub6(self, x_ue, y_ue, x_bs=0, y_bs=0):
        """
        path loss via to model COST 231 Hata (for freq 1.5 ~ 2 GHz)
        metropolitan
        """    
        
        f_mhz = self.fc_hz / 1e6 #Hz --> MHz
        d_meters = math.sqrt((x_ue - x_bs)**2 + (y_ue - y_bs)**2)
        d_meters = max(d_meters, 1.0)
        d_km = d_meters / 1000.0
        
        h_bs = 20.0
        h_ue = 1.5
        correction_factor_ue = (1.1 * np.log10(f_mhz) - 0.7) * h_ue - (1.56 * np.log10(f_mhz) - 0.8)
        env_offset_C = 3.0
        
        path_loss_db = (
                46.3 
                + 33.9 * np.log10(f_mhz) 
                - 13.82 * np.log10(h_bs) 
                - correction_factor_ue 
                + (44.9 - 6.55 * np.log10(h_bs)) * np.log10(d_km) 
                + env_offset_C
            )
        
        return path_loss_db
    
    def _compute_bf_vector(self, theta_rad):
            """
            steering vectors
            """
            C_LIGHT = 3e8                   
            wavelength = C_LIGHT / self.fc_hz # lambda
            
            d_spacing = wavelength / 2.0  
            
            # (Wave number) k = 2*pi / lambda
            wavenumber = 2.0 * np.pi / wavelength
            
            # Phase Shift
            # Indices n = [0, 1, 2, ..., M-1]
            antenna_indices = np.arange(self.num_antennas)
            
            #  phi = k * d * cos(theta) * n
            #  cos(theta)
            phase_shifts = 1j * wavenumber * d_spacing * np.cos(theta_rad) * antenna_indices
            
            normalization_factor = 1.0 / np.sqrt(self.num_antennas)
            
            steering_vector = normalization_factor * np.exp(phase_shifts)
            
            return steering_vector
    

    def _effective_sinr(self, raw_sinr_db):
        MAX_CODING_GAIN_DB = 3.0
        MIN_RATE = 0.001
        MAX_RATE = 0.999
        
        normalized_rate = (raw_sinr_db - self.sinr_min) / (self.sinr_max - self.sinr_min)
        
        code_rate = np.clip(normalized_rate, MIN_RATE, MAX_RATE)
        #R >> small -> negative log >> large --> positive gain >> large
        calculated_gain = -10 * np.log(code_rate)
        # not excess 3dbB
        final_gain = min(calculated_gain, MAX_CODING_GAIN_DB)
        
        return raw_sinr_db + final_gain
    
    def _compute_channel(self, x_ue, y_ue, x_bs, y_bs):
        PLE_LOS = 2.0
        PLE_NLOS = 4.0
        
        if self.use_beamforming:
            g_ant_dbi = 3.0
        else:
            g_ant_dbi = self.gain_no_beamforming
            
        antenna_norm_factor = 10 ** (g_ant_dbi / 10.0)
        
        #path loss
        is_mmWave = (self.fc_hz > 25e9)
        if is_mmWave:
            pl_db_loss = self._path_loss_mmWave(x_ue, y_ue, PLE_LOS, x_bs, y_bs)
            pl_db_nloss = self._path_loss_mmWave(x_ue, y_ue, PLE_NLOS, x_bs, y_bs)
        else:
            pl_db_loss = self._path_loss_sub6(x_ue, y_ue, x_bs, y_bs)
            pl_db_nloss = self._path_loss_sub6(x_ue, y_ue, x_bs, y_bs)
            
        # Linear Gain = 1 / Linear Loss
        gain_linear_los = 1.0 / (10 ** (pl_db_loss / 10.0)) 
        gain_linear_nlos = 1.0 / (10 ** (pl_db_nloss / 10.0))
        
        is_los = (self.np_random.binomial(1, self.prob_LOS) == 1)
        
        if is_los:
            num_paths = 1
            alpha = np.zeros(num_paths, dtype=complex)
            alpha[0] = math.sqrt(gain_linear_los)
        else:
            num_paths = self.Np
            # Fading Rayleigh (Complex Gaussian)
            real_part = self.np_random.normal(size=num_paths)
            imag_part = self.np_random.normal(size=num_paths)
            fading = (real_part + 1j * imag_part)
            alpha = fading * math.sqrt(gain_linear_nlos)
            
        # SPATIAL CHANNEL
        thetas = self.np_random.uniform(low=0, high=math.pi, size=num_paths)
        h_vector = np.zeros(self.num_antennas, dtype=complex)
        
        for i in range(num_paths):
            steering_vec = self._compute_bf_vector(thetas[i])
            h_vector += (alpha[i] / antenna_norm_factor) *steering_vec.T
        
        h_vector *= math.sqrt(self.num_antennas)
        return h_vector
    
    def _calc_power_gain(self, h_channel, tx_power, beam_idx=None):
        if self.use_beamforming and beam_idx is not None:
            # Beamforming: P_rx = P_tx * |h* . w|^2  
            # w: vector beamforming (precoder)
            precoder = self.F[:, beam_idx]
            channel_gain = abs(np.dot(h_channel.conj(), precoder)) ** 2
        else:
            # (SISO): P_rx = P_tx * ||h||^2
            channel_gain = np.linalg.norm(h_channel, ord=2) ** 2
        return tx_power * channel_gain

    def _compute_rf(self, x_ue, y_ue, pt_bs1, pt_bs2, is_ue_2=False):
        TEMP_KELVIN = 290
        BANDWIDTH = 15000 # Hz
        K_BOLTZMANN = 1.38e-23
        EPSILON = 1e-20
        
        # (Thermal Noise)
        noise_power = K_BOLTZMANN * TEMP_KELVIN * BANDWIDTH 

        if not is_ue_2:
            # UE 1: BS1 serve, BS2 interfere
            bs_srv_pos = (self.x_bs_1, self.y_bs_1)
            bs_int_pos = (self.x_bs_2, self.y_bs_2)
            pt_srv = pt_bs1
            pt_int = pt_bs2
            beam_srv = self.f_n_bs1
            beam_int = self.f_n_bs2
        else:
            # UE 2: BS2 interfere, BS1 serve
            bs_srv_pos = (self.x_bs_2, self.y_bs_2)
            bs_int_pos = (self.x_bs_1, self.y_bs_1)
            pt_srv = pt_bs2
            pt_int = pt_bs1
            beam_srv = self.f_n_bs2
            beam_int = self.f_n_bs1

        # (Channel Vector h)
        # (Signal Path): Serving BS -> UE
        h_signal = self._compute_channel(x_ue, y_ue, x_bs=bs_srv_pos[0], y_bs=bs_srv_pos[1])
        
        # (Interference Path): Interfering BS -> UE
        h_interf = self._compute_channel(x_ue, y_ue, x_bs=bs_int_pos[0], y_bs=bs_int_pos[1])

        # 4. calculate power
        received_power = self._calc_power_gain(h_signal, pt_srv, beam_srv)
        interference_power = self._calc_power_gain(h_interf, pt_int, beam_int)

        # 5. SINR
        total_interference = interference_power + noise_power + EPSILON
        
        if received_power <= 0 or total_interference <= 0:
            received_sinr_db = -np.inf
        else:
            # SINR (dB) = 10 * log10( Signal / (Interference + Noise) )
            received_sinr_db = 10 * np.log10(received_power / total_interference)

        return [received_power, interference_power, received_sinr_db]

    def step(self, action):
        x_ue_1, y_ue_1, x_ue_2, y_ue_2, p_srv, p_int = self.state
        POWER_LEVELS = [-3, -1, 1, 3]
        step_reward = 0
        
        if action == -1:
            pass
        elif 0<= action < self.num_actions:
            idx_int = (action & 0b1100) >> 2
            idx_srv = (action & 0b0011) 
            
            change_db_srv = POWER_LEVELS[idx_srv] 
            change_db_int = POWER_LEVELS[idx_int]
            
            # change from dB to Linear
            p_srv *= 10 ** (change_db_srv / 10.0)
            p_int *= 10 ** (change_db_int / 10.0)
            
        else:
            return np.array(self.state, dtype=np.float32), 0, False, {}
        
        # UAV moving
        speed_ms = 2.0 * (5.0 / 18.0) # km/h -> m/s
        
        theta_1 = self.np_random.uniform(-math.pi, math.pi)
        x_ue_1 += speed_ms * math.cos(theta_1)
        y_ue_1 += speed_ms * math.sin(theta_1)
        
        theta_2 = self.np_random.uniform(-math.pi, math.pi)
        x_ue_2 += speed_ms * math.cos(theta_2)
        y_ue_2 += speed_ms * math.sin(theta_2)
        
        # SINR
        _, _, sinr_1 = self._compute_rf(x_ue_1, y_ue_1, p_srv, p_int, is_ue_2=False)
        _, _, sinr_2 = self._compute_rf(x_ue_2, y_ue_2, p_srv, p_int, is_ue_2=True)
        
        #coding gain
        sinr_1 = self._effective_sinr(sinr_1)
        sinr_2 = self._effective_sinr(sinr_2)
        
        self.received_sinr_dB = sinr_1 
        self.received_ue2_sinr_dB = sinr_2
        self.serving_transmit_power_dBm = 10*np.log10(p_srv*1e3)
        self.interfering_transmit_power_dBm = 10*np.log10(p_int*1e3)
        #success : good sinr and ranged power
        
        is_success = (
            (sinr_1 >= self.sinr_min) and (sinr_2 >= self.sinr_min) and
            (p_srv <= self.max_tx_power) and (p_srv >= 0) and
            (p_int <= self.max_tx_power_interference) and (p_int >= 0) 
            and (sinr_1 <= self.sinr_max) and (sinr_2 <= self.sinr_max)
        )
 
        is_failure = (
            (p_srv > self.max_tx_power) or (p_int > self.max_tx_power_interference) or
            (sinr_1 < self.sinr_min) or (sinr_2 < self.sinr_min) or
            # (sinr_1 > self.sinr_max) or (sinr_2 > self.sinr_max) or
            # (sinr_1 < self.sinr_target) or (sinr_2 < self.sinr_target) or
            np.isnan(sinr_1) or np.isnan(sinr_2)
        )
        current_sinr = (sinr_1 + sinr_2) / 2.0
        ratio = (current_sinr - self.sinr_min) / (self.sinr_max - self.sinr_min)
        ratio = np.clip(ratio, 0.0, 1.0)
        done = False
        info = False
        
        if is_success:
            step_reward = ratio * self.reward_max
            done = True
            
        else:
            step_reward = self.reward_min
            done = True
            info = True
        
        self.state = (x_ue_1, y_ue_1, x_ue_2, y_ue_2, p_srv, p_int)
        self.received_sinr_dB = sinr_1
        self.received_ue2_sinr_dB = sinr_2
        return np.array(self.state, dtype=np.float32), step_reward, done, info
    
    def reset(self):
        self.f_n_bs1 = self.np_random.integers(low=0, high=self.num_antennas)
        self.f_n_bs2 = self.np_random.integers(low=0, high=self.num_antennas) 
        
        ue1_x = self.np_random.uniform(-self.cell_radius, self.cell_radius)
        ue1_y = self.np_random.uniform(-self.cell_radius, self.cell_radius)
        
        ue2_center_x = self.inter_site_distance
        ue2_x = self.np_random.uniform(ue2_center_x - self.cell_radius, ue2_center_x + self.cell_radius)
        ue2_y = self.np_random.uniform(-self.cell_radius, self.cell_radius)
        
        p_serving = self.np_random.uniform(1.0, self.max_tx_power / 2.0)
        p_interferer = self.np_random.uniform(1.0, self.max_tx_power_interference / 2.0)
        
        self.state = np.array([
                ue1_x, ue1_y,       # position UE 1
                ue2_x, ue2_y,       # position UE 2
                p_serving,          # power serving
                p_interferer        # power interfering
            ], dtype=np.float32)
            
        return self.state