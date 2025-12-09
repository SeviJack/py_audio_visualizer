import ctypes
import pygame, sys
import numpy as np
import sounddevice as sd
import win32gui, win32con
#todo
# custom window frame
# - transparent background button
# - click to toggle topmost
# - click and drag to move window
# - resize handle
# fix the damn logscale for bass
# add icon

#Constants 
WM_NCLBUTTONDOWN = 0x00A1
HTCAPTION = 0x02
LOW_BLOCK = 8192
HIGH_BLOCK = 1024
SPLIT_FREQ = 300  # Hz
last_size = (800, 400)
user32 = ctypes.windll.user32
ATTACK = 0.6   # 0..1, higher = snappier rise
DECAY  = 0.85  # 0..1, lower = faster fall

input_device = None
# display parameters
w, h = 32, 32
margin_x, margin_y = 6, 2
led_w, led_h = 12, 4
padding_x, padding_y_top, padding_y_bottom = 20, 20, 30

cell_x, cell_y = led_w + margin_x, led_h + margin_y
total_w = w * cell_x - margin_x + 2 * padding_x
total_h = h * cell_y - margin_y + padding_y_top + padding_y_bottom + 4
clock = pygame.time.Clock()

# audio analysis parameters
fmin, fmax = 40, 10000  # Hz range
SCALE_VALUE = 1  # adjust as needed  # 0.1 = small gain_nodes, 1.0 = full height
GATE = 0.02         # adjust as needed  # 0.02 = ignore <2% of max power
DC_CUTTOFF = 0

# audio parameters
samplerate = 48000
blocksize = 8192  # ~0.16s latency
# smoothing factor (lower = slower)
alpha = 0.15 #4
peak_gain_nodes = np.zeros

# freq_bins = np.geomspace(fmin, fmax, w + 1)
# freq_labels = [(freq_bins[i] + freq_bins[i+1]) / 2 for i in range(w)]

centers = np.array([
    40,   50,   63,   80,   100,  120,  125,  160,  180,  200,  250,
    315,  400,  500,  630,  800,  1000, 1250, 1600, 2000, 2200, 2500,
    3150, 3600, 4000, 5000, 6300, 7000, 7500, 8500, 9500, 10000
])


# Clamp to your fmin/fmax range
centers = centers[(centers >= fmin) & (centers <= fmax)]
w = len(centers)  # make w follow the band count

# Bin edges are geometric midpoints between centers
edges = np.zeros(w + 1)
edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
edges[0] = fmin
edges[-1] = fmax

freq_bins = edges
freq_labels = centers



# visualization parameters

gain_nodes = np.zeros(w)
smoothed = np.zeros(w)

# VFD color palette
core = (0, 255, 180)
edge = (0, 120, 90)
core_red = (255, 60, 40)
edge_red = (120, 20, 10)
bg = (5, 10, 10)
label_color = (100, 100, 100)

# Pygame setup
pygame.display.set_caption("Audio Visualizer Emulator")
pygame.init()
font = pygame.font.SysFont("Consolas", 10)  # small, clear font

def set_opacity(alpha: float):
    """alpha: 0.0-1.0"""
    a = max(0, min(255, int(alpha * 255)))
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, a, win32con.LWA_ALPHA)

def enable_clickthrough():
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    style |= (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)

def disable_clickthrough():
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    style &= ~win32con.WS_EX_TRANSPARENT
    style |= win32con.WS_EX_LAYERED
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)

def move_window_bottom_right():
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    ww, wh = screen.get_size()
    x = sw - ww
    y = sh - wh
    win32gui.SetWindowPos(hwnd, 0, x, y, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER)

# find default output and its loopback
def connect_loopback_input():
    global input_device
    default_output = sd.default.device[1]
    output_name = sd.query_devices(default_output)['name']

    # look for "(loopback)" version of that device (maybe thread this?)
    devices = sd.query_devices()
    loopback_index = None
    for i, d in enumerate(devices):
        if "Loopback" in d['name'] and output_name.split(' (')[0] in d['name']:
            loopback_index = i
            break
    if loopback_index is None:
        loopback_index = sd.default.device[0]

    input_device = loopback_index

def make_top_level_window():
    hwnd = pygame.display.get_wm_info()["window"]
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOPMOST,
        0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )
def move_window_bottom_right():
    hwnd = pygame.display.get_wm_info()["window"]
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # current pygame window size
    win_w, win_h = screen.get_size()

    x = screen_w - win_w
    y = screen_h - win_h - 48  # taskbar offset

    user32.SetWindowPos(hwnd, None, x, y, 0, 0,
                        win32con.SWP_NOSIZE | win32con.SWP_NOZORDER)

def audio_callback(indata, frames, time, status):
    global gain_nodes, smoothed, fmin, fmax
    
    # stereo → mono
    mono = np.mean(indata, axis=1)
    if np.max(np.abs(mono)) < 1e-6:
        gain_nodes.fill(0.0)
        smoothed.fill(0.0)
        return
    # --- Dual-FFT hybrid for low + high frequencies ---

    # long FFT for low end hz 118
    low_window = np.hanning(min(len(mono), LOW_BLOCK))
    low_fft = np.fft.rfft(mono[:LOW_BLOCK] * low_window)
    mag_low = np.abs(low_fft)
    freqs_low = np.fft.rfftfreq(LOW_BLOCK, 1.0 / samplerate)

    # short FFT for highs
    high_window = np.hanning(min(len(mono), HIGH_BLOCK))
    high_fft = np.fft.rfft(mono[:HIGH_BLOCK] * high_window)
    mag_high = np.abs(high_fft)
    freqs_high = np.fft.rfftfreq(HIGH_BLOCK, 1.0 / samplerate)

    # merge around crossover frequency
    split = SPLIT_FREQ  # Hz
    mask_low = freqs_low < split
    mask_high = freqs_high >= split
    # Slight bass compensation
    # bass_boost_freq = 150.0
    # bass_mask = freqs_low < bass_boost_freq
    # mag_low[bass_mask] *= 1.2

    freqs = np.concatenate((freqs_low[mask_low], freqs_high[mask_high]))
    mag = np.concatenate((mag_low[mask_low], mag_high[mask_high]))


    # magnitude spectrum
    # Mild compressive mapping (visual, not physical)
    mag = np.sqrt(mag + 1e-12)

    # # Slight bass compensation


    # restrict to frequency window
    mask = (freqs >= fmin) & (freqs <= fmax)
    mag = mag[mask]
    freqs = freqs[mask]

    # --- LOG-SPACED BINS ---  (vectorized, safe)
    # use precomputed global freq_bins and w
    idx = np.digitize(freqs, freq_bins) - 1
    idx = np.clip(idx, 0, w-1)

    
    # sum and count per bin
    counts = np.bincount(idx, minlength=w)
    sums   = np.bincount(idx, weights=mag, minlength=w)

    values = np.zeros(w, dtype=float)
    nonzero = counts > 0
    values[nonzero] = sums[nonzero] / counts[nonzero]
    
    # normalize + noise gate
    peak = np.percentile(values, 95) + 1e-6
    values = values / peak
    values = np.clip(values, 0, 1)
    values[values < GATE] = 0
    # visual gamma
    gamma = 1.2  # <1 makes low levels more visible
    values = np.power(values, gamma)

    # frequency tilt compensation to keep bass and highs visually even
    tilt = np.power(freq_labels / freq_labels[0], 0.22)
    values *= tilt
    values /= np.max(values + 1e-6)
    values = np.clip(values, 0.0, 1.0)

    #calculate height
    gain_nodes = values * h * SCALE_VALUE


# Pygame window setup
screen = pygame.display.set_mode((total_w, total_h), pygame.RESIZABLE | pygame.NOFRAME) #noframe
base_surface = pygame.Surface((total_w, total_h))  # offscreen render
fade_surface = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
hwnd = pygame.display.get_wm_info()['window']

set_opacity(1.0)
make_top_level_window()
#start
connect_loopback_input()
move_window_bottom_right()

 # draw vertical frequency label for every bar
label_surfaces = []
for x in range(w):
    label_val = round(freq_labels[x], -1)
    surf = font.render(str(int(label_val)), True, label_color)
    surf = pygame.transform.rotate(surf, -90)
    label_surfaces.append(surf)

stream = sd.InputStream(device=input_device,
                        samplerate=samplerate,
                        channels=2,
                        blocksize=blocksize,
                        callback=audio_callback)
stream.start()

hovered = False

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            stream.stop(); stream.close(); sys.exit()

        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                user32.ReleaseCapture()
                user32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0)

        elif e.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(e.size, pygame.RESIZABLE | pygame.NOFRAME)
            pygame.event.pump()
            hwnd = pygame.display.get_wm_info()["window"]
            make_top_level_window()
            move_window_bottom_right()
            set_opacity(1.0)
            hovered = False

    cx, cy = win32gui.GetCursorPos()
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    inside = (left <= cx < right) and (top <= cy < bottom)
    if inside and not hovered:
        set_opacity(0.5)
        enable_clickthrough()
        hovered = True
    elif not inside and hovered:
        set_opacity(1.0)
        disable_clickthrough()
        hovered = False
        
    # clear base surface    
    fade_surface.fill((0, 0, 0, 40))
    base_surface.blit(fade_surface, (0, 0))

    # exponential smoothing
    smoothed = alpha * gain_nodes + (1 - alpha) * smoothed

    # Hard floor to kill residual bars
    smoothed[smoothed < 0.01] = 0.0

    # draw gain_nodes and labels
    for x in range(w):
        height = int(smoothed[x])
        X = padding_x + x * cell_x
        for y in range(height):
            Y = total_h - padding_y_bottom - (y + 1) * cell_y
            pygame.draw.rect(base_surface, edge, (X, Y, led_w, led_h))
            pygame.draw.rect(base_surface, core, (X+1, Y+1, led_w-2, led_h-2))
        # red base
        Y_base = total_h - padding_y_bottom - cell_y
        pygame.draw.rect(base_surface, edge_red, (X, Y_base, led_w, led_h))
        pygame.draw.rect(base_surface, core_red, (X+1, Y_base+1, led_w-2, led_h-2))
        # draw label
        base_surface.blit(label_surfaces[x], (X, total_h - padding_y_bottom))

    
    # after all drawing done on base_surface:
    cur_size = screen.get_size()
    if cur_size == last_size:
        screen.blit(base_surface, (0,0))
    else:
        base_scaled = pygame.transform.smoothscale(base_surface, cur_size)
        screen.blit(base_scaled, (0,0))
        last_size = cur_size

    pygame.display.flip()
    clock.tick(30)
