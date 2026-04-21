"""
Garmin RSD Texture Mapper
-------------------------------------------------------------------
Optimized for high-quality sidescan mosaic output with:
- TVG (Time Varied Gain) correction in dB space
- Slant-to-ground range correction with nadir masking
- Empirical Gain Normalization (EGN)
- Despeckle filtering
- Float32 pipeline (avoids cumulative uint8 quantization)
- Per-ping metadata (range, sample count, pixel size)
- Auto-detect UTM projection (no Web Mercator distortion)
- Interpolated gap filling between pings
- Port/starboard merge into single mosaic

Outputs:
1. port_intensity.tif / star_intensity.tif: Individual side backscatter
2. intensity.tif: Merged port+starboard backscatter mosaic
3. texture.tif: Merged local-std texture map (High = rough/hard, Low = smooth/soft)
"""

import numpy as np
import pandas as pd
import os
import sys
import math
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
from pyproj import CRS, Transformer


# === Configuration ===
# Point INPUT_PATH to a single .RSD file, or a directory containing multiple .RSD files.
RSD_FILE = os.environ.get("GARMIN_RSD_FILE", r"path_to_rsd_file")

RSD_BASENAME = os.path.splitext(os.path.basename(RSD_FILE))[0]
RSD_PARENT_DIR = os.path.dirname(RSD_FILE)
OUTPUT_BASE_DIR = os.path.join(RSD_PARENT_DIR, f"garmin_output_{RSD_BASENAME}")
META_DIR = os.path.join(OUTPUT_BASE_DIR, "meta")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "processed")
ALL_META_FILE = os.path.join(META_DIR, "All-Garmin-Sonar-MetaData.csv")

# Processing Parameters
OUTPUT_RESOLUTION = 0.05    # 5cm per pixel
TEXTURE_WINDOW_SIZE = 15    # Texture analysis window (~1.5m at 5cm res)
MAX_RANGE_FALLBACK = 15.0   # Fallback if per-ping max_range not in metadata
PORT_MAX_RANGE_OVERRIDE_M = 15.0       # Force port swath width; set to None to use per-ping metadata
STARBOARD_MAX_RANGE_OVERRIDE_M = 15  # Force starboard swath width; set to None to use per-ping metadata
PORT_RANGE_SCALE = 1.0                 # Additional multiplicative range tuning for port
STARBOARD_RANGE_SCALE = 1.0            # Additional multiplicative range tuning for starboard

# Radiometric Corrections
APPLY_TVG = True
TVG_SPREADING_DB = 30.0     # Spreading loss compensation (20-40 dB typical for sidescan)
ABSORPTION_COEFF = 0.05     # dB/m (0.03-0.06 freshwater, 0.1-0.2 seawater at 455kHz)
APPLY_DESPECKLE = True
DESPECKLE_SIZE = 3           # Median filter kernel (3x3)
APPLY_LEE_FILTER = True      # Adaptive speckle suppression on waterfall
LEE_WINDOW_SIZE = 7          # Local window size for Lee filter

# Navigation
HEADING_SMOOTH_WINDOW = 100  # Heading smoothing window (pings)
MIN_SPEED_MS = 0.3           # Skip pings below this speed (m/s), 0 to disable

# Gap filling
FILL_GAPS = True
MAX_FILL_DISTANCE = 2.0     # Max interpolation distance between pings (meters)
OVERLAP_POLICY = "first"    # "first" keeps earliest painted pixel, "last" overwrites with newest

# Nadir zone
NADIR_MASK_BINS = 2          # Zero out first N ground-range bins (nadir artifact)
NADIR_ALTITUDE_FACTOR = 0.0  # Extra nadir mask from altitude: bins = altitude*factor/resolution

# Downscan-based nadir fill (paints bottom-return intensity along the boat track
# to close the sidescan nadir gap with real acoustic data)
APPLY_DOWNSCAN_NADIR_FILL = True
DOWNSCAN_META_NAME = "B001_ds_hifreq_meta.csv"  # alt: "B004_ds_vhifreq_meta.csv"
DOWNSCAN_STRIP_WIDTH_M = 0.6       # cross-track paint width centered on boat
DOWNSCAN_BOTTOM_WINDOW_M = 0.25    # vertical averaging window around bottom return
DOWNSCAN_PAYLOAD_MODE_OVERRIDE = "auto"  # "auto" or specific mode

# Raster gap fill (post-processing to close single-pixel gaps from scan line divergence)
GAP_FILL_PASSES = 3          # Iterative neighbor-mean fill passes (0 to disable, 2-3 typical)

# Contrast stretch & Color Calibration
STRETCH_LOW_PCT = 2          # Lower percentile for output contrast stretch (if dynamic)
STRETCH_HIGH_PCT = 98        # Upper percentile for output contrast stretch (if dynamic)
APPLY_LOG_COMPRESSION = True  # Log-compress before percentile stretch
LOG_COMPRESSION_SCALE = 6.0   # Higher = stronger highlight compression

# Set to fixed floats (e.g. LO=0.0, HI=0.8) to lock brightness consistently across different passes.
FIXED_STRETCH_LO = None       
FIXED_STRETCH_HI = None

# Record/payload decoding
PAYLOAD_PROBE_PINGS = 150     # Number of pings used to auto-select payload decoding mode
FALLBACK_SAMPLE_COUNT = 2048  # Fallback sample count when metadata ping_cnt is missing
PAYLOAD_MODES = [
    "u8_tail", "u8_even", "u8_odd", "u16_le", "u16_be", "u16_le_hi", "u16_le_lo",
    "son_u8", "son_even", "son_odd", "son_u16_le", "son_u16_be",
]
PAYLOAD_EXTRA_MODES = [
    "u8_tail_s16", "u8_even_s16", "u8_odd_s16", "u16_le_s16", "u16_be_s16",
    "u8_tail_s23", "u8_even_s23", "u8_odd_s23", "u16_le_s23", "u16_be_s23",
    "u8_tail_s24", "u8_even_s24", "u8_odd_s24", "u16_le_s24", "u16_be_s24",
    "u8_tail_s32", "u8_even_s32", "u8_odd_s32", "u16_le_s32", "u16_be_s32",
]
PAYLOAD_MODE_OVERRIDE_PORT = "son_u16_le"       # "auto" or one of PAYLOAD_MODES
PAYLOAD_MODE_OVERRIDE_STARBOARD = "son_u16_le"  # "auto" or one of PAYLOAD_MODES

# Side-specific sample ordering
REVERSE_PORT_SAMPLES = False
REVERSE_STARBOARD_SAMPLES = False
PORT_HEADING_OFFSET_DEG = 0.0    # Additional heading correction per side
STARBOARD_HEADING_OFFSET_DEG = 0.0
USE_PORT_CHANNEL_OVERRIDE = True         # Use channel override for normal port processing
PORT_CHANNEL_OVERRIDE_ID = 1             # Port channel_id in All-Garmin-Sonar-MetaData.csv

# EGN stabilization
EGN_SMOOTH_WINDOW = 250       # Smoothing width for beam-pattern estimate
EGN_GAIN_MIN = 0.45           # Prevent over-darkening
EGN_GAIN_MAX = 2.20           # Prevent noise amplification

# Ping-to-ping leveling
APPLY_ROW_BALANCE = False     # Remove ping gain jitter before EGN




# === Metadata Generation ===
def ensure_metadata_exists(rsd_file, meta_dir, force_regenerate=False):
    """
    Ensures metadata CSV files exist. If not, generates them using pingverter.

    Args:
        rsd_file: Path to RSD file
        meta_dir: Directory where metadata CSVs should be stored
        force_regenerate: If True, regenerate even if files exist

    Returns:
        True if metadata is ready, False if generation failed
    """
    port_meta = os.path.join(meta_dir, "B002_ss_port_meta.csv")
    star_meta = os.path.join(meta_dir, "B003_ss_star_meta.csv")

    if not force_regenerate and os.path.exists(port_meta) and os.path.exists(star_meta):
        print(f"  Metadata files found in {meta_dir}")
        return True

    print(f"Generating metadata from RSD file...")
    print(f"  RSD: {rsd_file}")
    print(f"  Output: {meta_dir}")
    print("  This may take a few minutes for large files...")

    try:
        from pingverter import gar2pingmapper

        parent_dir = os.path.dirname(meta_dir)
        os.makedirs(parent_dir, exist_ok=True)

        sonar_object = gar2pingmapper(rsd_file, parent_dir)

        if os.path.exists(port_meta) and os.path.exists(star_meta):
            print("  Metadata generation complete!")
            return True
        else:
            print("  ERROR: Metadata files not found after generation")
            return False

    except ImportError:
        print("  ERROR: pingverter library not found")
        print("  Install with: pip install pingverter")
        return False
    except Exception as e:
        print(f"  ERROR generating metadata: {e}")
        return False


# === Processing Functions ===

def apply_tvg_correction(intensities, slant_ranges_m, tvg_db=30.0, absorption=0.05):
    """
    Apply Time Varied Gain in dB space.

    TVG(dB) = tvg_db * log10(R) + 2 * absorption * R

    The first term compensates for geometric spreading loss.
    The second term compensates for frequency-dependent absorption.
    Returns float32 to preserve dynamic range.
    """
    n = len(intensities)
    if n == 0:
        return intensities.astype(np.float32)

    safe_ranges = np.maximum(slant_ranges_m, 0.1)

    spreading = tvg_db * np.log10(safe_ranges)
    absorption_loss = 2.0 * absorption * safe_ranges
    tvg_total_db = spreading + absorption_loss

    # Convert dB gain to linear multiplier and apply
    gain_linear = np.power(10.0, tvg_total_db / 20.0)
    corrected = intensities.astype(np.float32) * gain_linear

    return corrected


def slant_range_correction(intensities, altitude_m, max_range_m, res_m, nadir_mask_bins=5):
    """
    Converts slant range (time) to ground range (distance).
    Removes water column and corrects geometric distortion.
    Uses binned averaging (anti-aliasing) instead of naive linear interpolation
    to prevent "static TV" noise when heavily downsampling acoustic samples to grid cells.
    Returns float32.
    """
    n_samples = len(intensities)
    if n_samples == 0:
        return np.array([], dtype=np.float32)

    slant_ranges = np.linspace(0, max_range_m, n_samples)

    # Mask blind zone (slant range < altitude = inside water column)
    valid_mask = slant_ranges >= altitude_m
    if not np.any(valid_mask):
        return np.array([], dtype=np.float32)

    # Ground range: Rg = sqrt(Rs^2 - H^2)
    sr_valid = slant_ranges[valid_mask]
    ground_ranges = np.sqrt(sr_valid ** 2 - altitude_m ** 2)
    valid_intensities = intensities[valid_mask]

    if len(ground_ranges) < 2:
        return np.array([], dtype=np.float32)

    max_bin = int(max_range_m / res_m)
    if max_bin == 0:
        return np.zeros(1, dtype=np.float32)

    # Map each valid ground range to its grid bin
    bins = (ground_ranges / res_m).astype(np.int32)
    
    valid_bins = bins[bins < max_bin]
    valid_vals = valid_intensities[bins < max_bin]

    if len(valid_bins) == 0:
        return np.zeros(max_bin, dtype=np.float32)

    # Fast averaging of all acoustic samples that fall within pixel width
    counts = np.bincount(valid_bins, minlength=max_bin)[:max_bin]
    sums = np.bincount(valid_bins, weights=valid_vals, minlength=max_bin)[:max_bin]

    corrected_line = np.zeros(max_bin, dtype=np.float32)
    mask = counts > 0
    corrected_line[mask] = sums[mask] / counts[mask]

    # Fill empty bins using linear interpolation from neighbors
    empty = ~mask
    if np.any(empty) and np.any(mask):
        valid_indices = np.flatnonzero(mask)
        empty_indices = np.flatnonzero(empty)
        # only interpolate within the min-max valid range
        min_idx, max_idx = valid_indices[0], valid_indices[-1]
        empty_to_fill = empty_indices[(empty_indices > min_idx) & (empty_indices < max_idx)]
        if len(empty_to_fill) > 0:
            corrected_line[empty_to_fill] = np.interp(
                empty_to_fill, valid_indices, corrected_line[valid_indices]
            )

    # Mask nadir zone (first few bins are geometrically unstable)
    dynamic_nadir = int((max(0.0, altitude_m) * NADIR_ALTITUDE_FACTOR) / max(res_m, 1e-6))
    effective_nadir_bins = max(nadir_mask_bins, dynamic_nadir)
    if effective_nadir_bins > 0 and len(corrected_line) > effective_nadir_bins:
        corrected_line[:effective_nadir_bins] = 0

    return corrected_line


def compute_egn_curve(waterfall):
    """
    Calculates Empirical Gain Normalization (EGN) curve.
    Uses column-wise mean of non-zero pixels, double-smoothed
    with a wide kernel to isolate the low-frequency beam pattern.
    """
    egn_curve = np.zeros(waterfall.shape[1], dtype=np.float64)
    for c in range(waterfall.shape[1]):
        col = waterfall[:, c]
        valid = col[col > 0]
        if len(valid) == 0:
            continue
        # Use median instead of mean to avoid outlier-driven beam estimates.
        egn_curve[c] = np.median(valid)

    # Fill holes in curve, then smooth heavily to retain only low-frequency beam shape.
    valid_idx = np.flatnonzero(egn_curve > 0)
    if len(valid_idx) < 2:
        return np.where(egn_curve > 0, egn_curve, 1.0)
    missing_idx = np.flatnonzero(egn_curve <= 0)
    if len(missing_idx) > 0:
        egn_curve[missing_idx] = np.interp(missing_idx, valid_idx, egn_curve[valid_idx])

    kernel_size = min(EGN_SMOOTH_WINDOW, len(egn_curve) // 2)
    if kernel_size > 1:
        kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
        smoothed = np.convolve(egn_curve, kernel, mode='same')
        egn_curve = np.convolve(smoothed, kernel, mode='same')

    return np.where(egn_curve > 1e-6, egn_curve, 1.0)


def apply_egn(waterfall, egn_curve):
    """
    Applies the inverse beam pattern for even illumination.
    Returns float32 (no uint8 clipping to preserve dynamic range).
    """
    global_mean = np.median(egn_curve[egn_curve > 0])
    safe_curve = np.where(egn_curve < 1e-6, 1e-6, egn_curve)
    gain_factors = global_mean / safe_curve
    gain_factors = np.clip(gain_factors, EGN_GAIN_MIN, EGN_GAIN_MAX)
    return (waterfall * gain_factors).astype(np.float32)


def lee_despeckle(image, window_size=7):
    """
    Adaptive Lee filter for speckle suppression while preserving edges.
    """
    img = image.astype(np.float32)
    mean_local = uniform_filter(img, size=window_size)
    mean_sq_local = uniform_filter(img * img, size=window_size)
    var_local = np.clip(mean_sq_local - mean_local * mean_local, 0.0, None)

    positive_var = var_local[var_local > 0]
    if len(positive_var) == 0:
        return img
    noise_var = np.percentile(positive_var, 15)

    weight = var_local / (var_local + noise_var + 1e-6)
    filtered = mean_local + weight * (img - mean_local)
    return filtered.astype(np.float32)


def log_compress(data, scale=6.0):
    """
    Log compression to tame bright outliers and reveal seabed structure.
    """
    out = data.astype(np.float32).copy()
    valid = out > 0
    if not np.any(valid):
        return out
    v = out[valid]
    p99 = np.percentile(v, 99.5)
    if p99 <= 0:
        return out
    x = np.clip(v / p99, 0.0, None)
    out[valid] = np.log1p(scale * x) / np.log1p(scale)
    return out


def balance_ping_levels(waterfall):
    """
    Remove row-wise gain jitter using robust per-row medians.
    """
    w = waterfall.astype(np.float32).copy()
    row_medians = np.zeros(w.shape[0], dtype=np.float32)
    for i in range(w.shape[0]):
        row = w[i]
        valid = row[row > 0]
        if len(valid) > 0:
            row_medians[i] = np.median(valid)

    valid_medians = row_medians[row_medians > 0]
    if len(valid_medians) == 0:
        return w
    target = np.median(valid_medians)

    for i in range(w.shape[0]):
        med = row_medians[i]
        if med <= 0:
            continue
        # Clamp prevents dim/noisy rows from being amplified unbounded,
        # which otherwise inject bright streaks that EGN can't undo.
        scale = np.clip(target / med, 0.5, 2.0)
        valid = w[i] > 0
        w[i, valid] *= scale

    return w


def match_side_levels(port_data, star_data):
    """
    Rescale port and starboard waterfalls so their non-zero medians match.
    Removes the along-track seam created by per-side independent EGN.
    """
    if port_data is None or star_data is None:
        return port_data, star_data

    port_valid = port_data[port_data > 0]
    star_valid = star_data[star_data > 0]
    if len(port_valid) == 0 or len(star_valid) == 0:
        return port_data, star_data

    port_med = float(np.median(port_valid))
    star_med = float(np.median(star_valid))
    if port_med <= 0 or star_med <= 0:
        return port_data, star_data

    target = 0.5 * (port_med + star_med)
    port_scale = np.clip(target / port_med, 0.5, 2.0)
    star_scale = np.clip(target / star_med, 0.5, 2.0)

    print(f"  Side level match: port x{port_scale:.3f}, star x{star_scale:.3f}")
    return (port_data.astype(np.float32) * port_scale,
            star_data.astype(np.float32) * star_scale)


def calculate_texture(image, window_size=15):
    """
    Computes local standard deviation as a texture proxy.
    High values = rough/hard texture (reef, shell, rock)
    Low values = smooth texture (sand, mud, silt)

    Uses the identity: std = sqrt(E[x^2] - E[x]^2) with uniform_filter,
    which is ~100x faster than generic_filter with np.std.
    """
    print("  Computing texture analysis...")
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img ** 2, size=window_size)
    variance = np.clip(mean_sq - mean ** 2, 0, None)
    return np.sqrt(variance).astype(np.float32)


def circular_lerp(a_deg, b_deg, t):
    """Linearly interpolate between two angles (degrees) along the shortest arc."""
    diff = ((b_deg - a_deg + 180) % 360) - 180
    return (a_deg + t * diff) % 360


def smooth_and_filter_nav(nav_data, waterfall_rows, transformer=None):
    """
    Compute COG headings from projected (UTM) coordinates and smooth them.
    The instrument heading from the Garmin is too noisy for scan line painting,
    so we derive heading entirely from GPS track using a wide step.
    """
    if len(nav_data) < 10:
        return nav_data, waterfall_rows

    x_vals = np.array([n[0] for n in nav_data])
    y_vals = np.array([n[1] for n in nav_data])

    # Project to UTM for accurate distance/bearing computation
    if transformer is not None:
        px, py = transformer.transform(x_vals, y_vals)
    else:
        px, py = x_vals.copy(), y_vals.copy()

    # Compute COG from projected coordinates with a wide step
    # 50-ping step ≈ 2-4m, enough for stable bearing estimation
    step = 50
    cog_headings = np.zeros(len(nav_data), dtype=np.float64)
    for i in range(len(nav_data)):
        p_prev = max(0, i - step)
        p_next = min(len(nav_data) - 1, i + step)

        if p_prev == p_next:
            cog_headings[i] = nav_data[i][2]  # fallback to instrument heading
            continue

        dx = px[p_next] - px[p_prev]
        dy = py[p_next] - py[p_prev]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 0.5:
            cog_headings[i] = nav_data[i][2]
        else:
            cog_headings[i] = np.degrees(np.arctan2(dx, dy)) % 360

    # Circular moving average to further smooth any GPS jitter
    if HEADING_SMOOTH_WINDOW > 1:
        window = HEADING_SMOOTH_WINDOW
        if window % 2 == 0:
            window += 1
        half = window // 2

        heading_smooth = np.copy(cog_headings)
        for i in range(half, len(cog_headings) - half):
            local = cog_headings[i - half:i + half + 1]
            sin_mean = np.mean(np.sin(np.radians(local)))
            cos_mean = np.mean(np.cos(np.radians(local)))
            heading_smooth[i] = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360

        # Extend edge smoothing (use nearest smoothed value instead of raw heading)
        heading_smooth[:half] = heading_smooth[half]
        heading_smooth[-half:] = heading_smooth[-(half + 1)]
    else:
        heading_smooth = cog_headings

    smoothed_nav = [(x_vals[i], y_vals[i], heading_smooth[i])
                    for i in range(len(nav_data))]
    return smoothed_nav, waterfall_rows


def determine_utm_epsg(lats, lons):
    """
    Auto-detect the appropriate UTM EPSG code from data centroid.
    Returns EPSG code as integer (e.g. 32615 for UTM zone 15N).
    """
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)

    zone_number = int((center_lon + 180) / 6) + 1

    if center_lat >= 0:
        epsg = 32600 + zone_number  # Northern hemisphere
    else:
        epsg = 32700 + zone_number  # Southern hemisphere

    return epsg


def compute_next_offsets(meta_df):
    """
    Compute next record offsets from metadata.
    Reading between consecutive offsets avoids bleeding into adjacent records.
    """
    next_offsets = np.full(len(meta_df), np.nan, dtype=np.float64)
    if 'index' not in meta_df.columns or len(meta_df) < 2:
        return next_offsets

    offsets = pd.to_numeric(meta_df['index'], errors='coerce').to_numpy(dtype=np.float64)
    for i in range(len(offsets) - 1):
        curr_off = offsets[i]
        next_off = offsets[i + 1]
        if np.isfinite(curr_off) and np.isfinite(next_off) and next_off > curr_off:
            next_offsets[i] = next_off
    return next_offsets


def read_ping_record(rsd_handle, file_size, offset, data_size, next_offset):
    """
    Read one ping record robustly.
    Prefer [offset, next_offset) span when available; fallback to metadata size.
    """
    if offset < 0 or offset >= file_size:
        return b""

    ping_header_len = 37
    fallback_read = max(0, data_size + ping_header_len)

    if np.isfinite(next_offset) and next_offset > offset:
        read_len = int(next_offset - offset)
    else:
        read_len = fallback_read

    if read_len <= 0:
        return b""

    rsd_handle.seek(offset)
    return rsd_handle.read(read_len)


def extract_payload_candidates(record_data, sample_count, son_offset=None):
    """
    Build candidate payload interpretations from the tail of a variable-length record.
    """
    candidates = {}
    if sample_count <= 0 or not record_data:
        return candidates

    n = int(sample_count)

    def add_shifted(shift_bytes):
        end = len(record_data) - shift_bytes
        if end <= 0:
            return

        suffix = "" if shift_bytes == 0 else f"_s{shift_bytes}"

        if end >= n:
            tail_u8 = np.frombuffer(record_data[end - n:end], dtype=np.uint8).astype(np.float32)
            candidates[f"u8_tail{suffix}"] = tail_u8
            # Some channels encode useful intensity only in one nibble.
            candidates[f"u8_tail_hi4{suffix}"] = np.floor(tail_u8 / 16.0) * 17.0
            candidates[f"u8_tail_lo4{suffix}"] = np.mod(tail_u8, 16.0) * 17.0

        if end >= (2 * n):
            tail_2n = record_data[end - 2 * n:end]
            raw_u8 = np.frombuffer(tail_2n, dtype=np.uint8)
            u16_le = np.frombuffer(tail_2n, dtype="<u2")
            u16_be = np.frombuffer(tail_2n, dtype=">u2")

            candidates[f"u8_even{suffix}"] = raw_u8[0::2].astype(np.float32)
            candidates[f"u8_odd{suffix}"] = raw_u8[1::2].astype(np.float32)
            candidates[f"u8_even_hi4{suffix}"] = np.floor(raw_u8[0::2].astype(np.float32) / 16.0) * 17.0
            candidates[f"u8_even_lo4{suffix}"] = np.mod(raw_u8[0::2].astype(np.float32), 16.0) * 17.0
            candidates[f"u8_odd_hi4{suffix}"] = np.floor(raw_u8[1::2].astype(np.float32) / 16.0) * 17.0
            candidates[f"u8_odd_lo4{suffix}"] = np.mod(raw_u8[1::2].astype(np.float32), 16.0) * 17.0
            candidates[f"u16_le{suffix}"] = u16_le.astype(np.float32)
            candidates[f"u16_be{suffix}"] = u16_be.astype(np.float32)

            # High/low byte variants often reveal where Garmin packs effective intensity.
            if shift_bytes == 0:
                candidates["u16_le_hi"] = ((u16_le >> 8) & 0xFF).astype(np.float32)
                candidates["u16_le_lo"] = (u16_le & 0xFF).astype(np.float32)

    def add_anchored(start, prefix):
        if start is None or start < 0:
            return
        if len(record_data) >= (start + n):
            start_u8 = np.frombuffer(record_data[start:start + n], dtype=np.uint8).astype(np.float32)
            candidates[f"{prefix}_u8"] = start_u8
            candidates[f"{prefix}_u8_hi4"] = np.floor(start_u8 / 16.0) * 17.0
            candidates[f"{prefix}_u8_lo4"] = np.mod(start_u8, 16.0) * 17.0
        if len(record_data) >= (start + 2 * n):
            block = record_data[start:start + 2 * n]
            raw_u8 = np.frombuffer(block, dtype=np.uint8)
            candidates[f"{prefix}_even"] = raw_u8[0::2].astype(np.float32)
            candidates[f"{prefix}_odd"] = raw_u8[1::2].astype(np.float32)
            candidates[f"{prefix}_even_hi4"] = np.floor(raw_u8[0::2].astype(np.float32) / 16.0) * 17.0
            candidates[f"{prefix}_even_lo4"] = np.mod(raw_u8[0::2].astype(np.float32), 16.0) * 17.0
            candidates[f"{prefix}_odd_hi4"] = np.floor(raw_u8[1::2].astype(np.float32) / 16.0) * 17.0
            candidates[f"{prefix}_odd_lo4"] = np.mod(raw_u8[1::2].astype(np.float32), 16.0) * 17.0
            candidates[f"{prefix}_u16_le"] = np.frombuffer(block, dtype="<u2").astype(np.float32)
            candidates[f"{prefix}_u16_be"] = np.frombuffer(block, dtype=">u2").astype(np.float32)

    add_shifted(0)
    add_shifted(16)
    add_shifted(23)
    add_shifted(24)
    add_shifted(32)
    if son_offset is not None:
        add_anchored(int(son_offset), "son")

    return candidates


def score_payload_line(samples):
    """
    Per-ping detail score. Higher = richer structure.
    Entropy is intentionally NOT rewarded: uniform-distribution entropy is
    maximized by random bytes, which is exactly the wrong-decode signature.
    """
    if samples is None or len(samples) < 16:
        return -1e9

    s = samples.astype(np.float64)
    if np.allclose(s, s[0]):
        return -1e9

    p99, p01 = np.percentile(s, [99, 1])
    spread = p99 - p01
    if spread <= 1e-6:
        return -1e9

    s_norm = np.clip((s - p01) / spread, 0.0, 1.0)

    contrast = float(np.std(s_norm))
    high_freq = float(np.std(np.diff(s_norm)))

    return contrast + 0.75 * high_freq


def mode_simplicity_bonus(mode):
    """
    Nudge toward canonical modes. The top cluster (e.g. u8_odd + u8_odd_s16 +
    u8_odd_s24 + u8_odd_s32) reads the same underlying bytes at different
    offsets and scores within ~0.02 of each other; without a decisive bonus
    the winner flips arbitrarily between runs.
    """
    bonus = 0.0
    if "_s" not in mode:
        bonus += 0.05
    if mode in ("u8_tail", "u8_odd", "u8_even", "u16_le", "u16_be"):
        bonus += 0.02
    return bonus


def range_profile_score(stack):
    """
    Reward stable range-dependent structure averaged across probes.
    Correct decode: per-ping range profile has a nadir bump + decay at
    roughly consistent sample indices, so the mean profile retains
    low-frequency shape after smoothing. Wrong decode: each row is ~random,
    so the mean profile is near-flat and loses most of its std when smoothed.

    Returns smoothed_std / raw_std ∈ [0, 1]. Noise ≈ 1/sqrt(window) (~0.2);
    real signal ≈ 0.7+.
    """
    if stack.ndim != 2 or stack.shape[0] < 2 or stack.shape[1] < 16:
        return 0.0

    s = stack.astype(np.float64)
    p99 = np.percentile(s, 99, axis=1, keepdims=True)
    p01 = np.percentile(s, 1, axis=1, keepdims=True)
    spread = np.maximum(p99 - p01, 1e-6)
    sn = np.clip((s - p01) / spread, 0.0, 1.0)

    mean_profile = np.mean(sn, axis=0)
    raw_std = float(np.std(mean_profile))
    if raw_std < 1e-6:
        return 0.0

    window = max(3, min(21, len(mean_profile) // 20))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(mean_profile, kernel, mode='same')
    return float(np.std(smoothed) / raw_std)


def choose_payload_mode(rsd_handle, meta_df, next_offsets, file_size, side_name):
    """
    Auto-select payload decoding mode.

    Scoring combines three independent signals:
      - per-ping detail (contrast + high-frequency variation)
      - adjacent-ping correlation (true backscatter is correlated 0.7-0.95
        between neighbors; random bytes are uncorrelated)
      - aggregate range-profile structure (correct decode produces a
        stable nadir bump + decay; noise produces a flat mean profile)

    A small simplicity bonus breaks near-ties toward canonical modes.
    """
    mode_names = list(PAYLOAD_MODES) + list(PAYLOAD_EXTRA_MODES)
    per_ping_scores = {m: [] for m in mode_names}
    adjacent_corrs = {m: [] for m in mode_names}
    sample_stacks = {m: [] for m in mode_names}

    has_ping_cnt = 'ping_cnt' in meta_df.columns
    has_speed = 'speed_ms' in meta_df.columns
    has_son_offset = 'son_offset' in meta_df.columns

    # Speed filter excludes docked/idling pings at trip endpoints that
    # otherwise degrade the probe set with weak signal.
    valid_positions = []
    for i, row in meta_df.iterrows():
        if pd.isna(row.get('index')) or pd.isna(row.get('data_size')):
            continue
        if has_speed and MIN_SPEED_MS > 0:
            speed = row.get('speed_ms')
            if pd.notna(speed) and speed < MIN_SPEED_MS:
                continue
        valid_positions.append(i)

    if len(valid_positions) < 2:
        return "u8_tail"

    # Each probe needs a following ping for adjacent-ping correlation.
    probe_ceiling = len(valid_positions) - 1
    probe_count = min(PAYLOAD_PROBE_PINGS, probe_ceiling)
    probe_idx = np.linspace(0, probe_ceiling - 1, probe_count, dtype=np.int32)

    def read_candidates(row_pos):
        row = meta_df.iloc[row_pos]
        offset = int(row['index'])
        data_size = int(row['data_size'])
        next_off = next_offsets[row_pos]
        if has_ping_cnt and pd.notna(row.get('ping_cnt')) and row['ping_cnt'] > 0:
            n_samples = int(row['ping_cnt'])
        else:
            n_samples = FALLBACK_SAMPLE_COUNT
        record_data = read_ping_record(rsd_handle, file_size, offset, data_size, next_off)
        if len(record_data) < 32:
            return None
        son_offset = int(row['son_offset']) if (has_son_offset and pd.notna(row.get('son_offset'))) else None
        return extract_payload_candidates(record_data, n_samples, son_offset=son_offset)

    for k in probe_idx:
        cand_a = read_candidates(valid_positions[k])
        cand_b = read_candidates(valid_positions[k + 1])
        if cand_a is None or cand_b is None:
            continue

        for mode in mode_names:
            a = cand_a.get(mode)
            b = cand_b.get(mode)
            if a is None or b is None or len(a) < 16:
                continue

            per_ping_scores[mode].append(score_payload_line(a))
            sample_stacks[mode].append(a)

            n = min(len(a), len(b), 1024)
            if n >= 32:
                ar = a[:n].astype(np.float64) - np.mean(a[:n])
                br = b[:n].astype(np.float64) - np.mean(b[:n])
                denom = (np.linalg.norm(ar) * np.linalg.norm(br)) + 1e-6
                adjacent_corrs[mode].append(float(np.dot(ar, br) / denom))

    score_details = {}
    for mode in mode_names:
        raw_scores = per_ping_scores[mode]
        attempted = len(raw_scores)
        valid = [s for s in raw_scores if s > -1e8]
        # Require ≥25% of probes to pass the per-ping filter; a mode that
        # survives aggregation but fails on most pings produces a black mosaic.
        min_required = max(3, attempted // 4) if attempted else 0
        if not valid or len(valid) < min_required:
            score_details[mode] = {
                "total": -1e9, "per_ping": -1e9, "corr": 0.0,
                "profile": 0.0, "bonus": 0.0, "valid": len(valid),
            }
            continue

        per_ping = float(np.mean(valid))
        corr_mean = float(np.mean(adjacent_corrs[mode])) if adjacent_corrs[mode] else 0.0

        stack = sample_stacks[mode]
        profile = 0.0
        if len(stack) >= 2:
            min_len = min(len(s) for s in stack)
            if min_len >= 16:
                arr = np.stack([s[:min_len] for s in stack], axis=0)
                profile = range_profile_score(arr)

        bonus = mode_simplicity_bonus(mode)
        total = per_ping + 1.5 * corr_mean + 1.0 * profile + bonus
        score_details[mode] = {
            "total": total, "per_ping": per_ping, "corr": corr_mean,
            "profile": profile, "bonus": bonus, "valid": len(valid),
        }

    best_mode = max(score_details, key=lambda m: score_details[m]["total"])

    ranked = sorted(score_details.items(), key=lambda x: -x[1]["total"])
    top = ranked[:5]
    print(f"  Auto-selected payload mode for {side_name}: {best_mode}")
    print(f"    Top candidates (total = per_ping + 1.5*corr + 1.0*profile + bonus):")
    for mode, d in top:
        print(
            f"      {mode:>18s}: total={d['total']:.3f}  "
            f"per_ping={d['per_ping']:.3f}  corr={d['corr']:.3f}  "
            f"profile={d['profile']:.3f}  bonus={d['bonus']:.3f}  valid={d['valid']}"
        )
    if len(ranked) >= 2:
        margin = ranked[0][1]["total"] - ranked[1][1]["total"]
        if margin < 0.02:
            print(f"    WARNING: top two modes within {margin:.3f} — decode is ambiguous")

    return best_mode


def extract_nav_point(row):
    """
    Extract navigation tuple in whichever coordinate columns exist.
    """
    x_coord, y_coord = None, None
    if pd.notna(row.get('lon')) and pd.notna(row.get('lat')):
        x_coord = float(row['lon'])
        y_coord = float(row['lat'])
    elif pd.notna(row.get('e')) and pd.notna(row.get('n')):
        x_coord = float(row['e'])
        y_coord = float(row['n'])

    if x_coord is None:
        return None

    heading = float(row['instr_heading']) if pd.notna(row.get('instr_heading')) else 0.0
    return x_coord, y_coord, heading


def process_side(rsd_handle, meta_df, side_name, sign, transformer=None, payload_mode_override="auto",
                 reverse_samples=False, heading_offset_deg=0.0):
    """
    Reads and processes one side (port or starboard) of sidescan data.

    Pipeline order: TVG -> Slant Range Correction -> Stack -> EGN -> Despeckle

    Uses per-ping metadata (max_range, pixM, ping_cnt) when available
    instead of hardcoded global values.
    """
    print(f"\n--- Processing {side_name} Side ---")

    waterfall_rows = []
    nav_data = []
    meta_df = meta_df.reset_index(drop=True)
    next_offsets = compute_next_offsets(meta_df)

    file_size = os.path.getsize(RSD_FILE)

    # Check which per-ping metadata columns are available
    has_max_range = 'max_range' in meta_df.columns
    has_son_offset = 'son_offset' in meta_df.columns
    has_speed = 'speed_ms' in meta_df.columns
    has_ping_cnt = 'ping_cnt' in meta_df.columns

    if has_max_range:
        print(f"  Using per-ping max_range from metadata")
    else:
        print(f"  Using fallback max_range = {MAX_RANGE_FALLBACK}m")

    if payload_mode_override and payload_mode_override != "auto":
        payload_mode = payload_mode_override
        print(f"  Forced payload mode for {side_name}: {payload_mode}")
    else:
        payload_mode = choose_payload_mode(rsd_handle, meta_df, next_offsets, file_size, side_name)

    skipped_slow = 0
    skipped_no_nav = 0
    skipped_decode = 0

    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Reading"):
        if pd.isna(row['index']) or pd.isna(row['data_size']):
            continue
        offset = int(row['index'])
        if offset >= file_size:
            continue

        # Skip near-stationary pings (GPS pileup)
        if has_speed and MIN_SPEED_MS > 0:
            speed = row['speed_ms']
            if pd.notna(speed) and speed < MIN_SPEED_MS:
                skipped_slow += 1
                continue

        rsd_handle.seek(offset)
        try:
            body_size = int(row['data_size'])
            next_off = next_offsets[idx]
            record_data = read_ping_record(rsd_handle, file_size, offset, body_size, next_off)

            if len(record_data) < 32:
                continue

            if has_ping_cnt and pd.notna(row['ping_cnt']) and row['ping_cnt'] > 0:
                n_samples = int(row['ping_cnt'])
            else:
                n_samples = FALLBACK_SAMPLE_COUNT

            son_offset = int(row['son_offset']) if ('son_offset' in meta_df.columns and pd.notna(row.get('son_offset'))) else None
            candidates = extract_payload_candidates(record_data, n_samples, son_offset=son_offset)
            raw_intensities = candidates.get(payload_mode)
            if raw_intensities is None and candidates:
                # Fallback to first available mode for this ping if chosen mode is unavailable.
                raw_intensities = next(iter(candidates.values()))
            if raw_intensities is None:
                skipped_decode += 1
                continue

            if len(raw_intensities) == 0:
                continue
            if reverse_samples:
                raw_intensities = raw_intensities[::-1].copy()

            # Per-ping range from metadata, or fallback
            if has_max_range and pd.notna(row['max_range']):
                ping_max_range = float(row['max_range'])
            else:
                ping_max_range = MAX_RANGE_FALLBACK

            # Per-side range overrides (useful when metadata reports wrong swath width).
            if sign < 0 and PORT_MAX_RANGE_OVERRIDE_M is not None:
                ping_max_range = float(PORT_MAX_RANGE_OVERRIDE_M)
            elif sign > 0 and STARBOARD_MAX_RANGE_OVERRIDE_M is not None:
                ping_max_range = float(STARBOARD_MAX_RANGE_OVERRIDE_M)

            if sign < 0:
                ping_max_range *= PORT_RANGE_SCALE
            else:
                ping_max_range *= STARBOARD_RANGE_SCALE

            if ping_max_range <= 0:
                continue

            # Compute slant ranges for this ping's samples
            slant_ranges = np.linspace(0, ping_max_range, len(raw_intensities))

            # TVG correction (in dB space, returns float32)
            if APPLY_TVG:
                corrected_intensities = apply_tvg_correction(
                    raw_intensities, slant_ranges, TVG_SPREADING_DB, ABSORPTION_COEFF)
            else:
                corrected_intensities = raw_intensities.astype(np.float32)

            # Slant range -> ground range correction with nadir masking
            depth = row['inst_dep_m'] if pd.notna(row['inst_dep_m']) else 1.0
            corrected = slant_range_correction(
                corrected_intensities, depth, ping_max_range,
                OUTPUT_RESOLUTION, NADIR_MASK_BINS)

            if len(corrected) > 0:
                nav_point = extract_nav_point(row)
                if nav_point is None:
                    # Keep intensity/nav arrays strictly aligned to avoid row shifts in georeferencing.
                    skipped_no_nav += 1
                    continue
                waterfall_rows.append(corrected)
                x_nav, y_nav, h_nav = nav_point
                nav_data.append((x_nav, y_nav, (h_nav + heading_offset_deg) % 360.0))

        except Exception:
            continue

    if skipped_slow > 0:
        print(f"  Skipped {skipped_slow} near-stationary pings (speed < {MIN_SPEED_MS} m/s)")
    if skipped_no_nav > 0:
        print(f"  Skipped {skipped_no_nav} pings with missing navigation")
    if skipped_decode > 0:
        print(f"  Skipped {skipped_decode} pings that could not decode payload")

    if not waterfall_rows:
        print("  No valid data found.")
        return None, None, None, None

    if len(nav_data) != len(waterfall_rows):
        aligned_len = min(len(nav_data), len(waterfall_rows))
        nav_data = nav_data[:aligned_len]
        waterfall_rows = waterfall_rows[:aligned_len]

    # Smooth headings and filter outliers
    print("  Smoothing navigation data...")
    nav_data, waterfall_rows = smooth_and_filter_nav(nav_data, waterfall_rows, transformer=transformer)

    if not waterfall_rows:
        print("  No valid data after filtering.")
        return None, None, None, None

    # Stack into waterfall (float32 to preserve TVG dynamic range)
    print("  Stacking waterfall...")
    max_len = max(len(r) for r in waterfall_rows)
    waterfall = np.zeros((len(waterfall_rows), max_len), dtype=np.float32)
    for i, r in enumerate(waterfall_rows):
        waterfall[i, :len(r)] = r

    if APPLY_ROW_BALANCE:
        print("  Balancing ping levels...")
        waterfall = balance_ping_levels(waterfall)

    # EGN (beam pattern normalization) - applied BEFORE despeckle
    print("  Applying Gain Normalization...")
    egn_curve = compute_egn_curve(waterfall)
    waterfall = apply_egn(waterfall, egn_curve)

    # Despeckle AFTER EGN (preserves beam pattern statistics for EGN)
    if APPLY_DESPECKLE:
        print("  Applying despeckle filter...")
        waterfall = median_filter(waterfall, size=DESPECKLE_SIZE).astype(np.float32)

    # Texture BEFORE Lee: Lee suppresses the local variance that
    # calculate_texture measures. Median despeckle has already removed
    # impulse noise, so residual variance is substrate-driven.
    texture_img = calculate_texture(waterfall, TEXTURE_WINDOW_SIZE)

    if APPLY_LEE_FILTER:
        print("  Applying adaptive Lee filter...")
        waterfall = lee_despeckle(waterfall, window_size=LEE_WINDOW_SIZE)

    return waterfall, texture_img, nav_data, 1 if sign == 1 else -1


def process_downscan_nadir(rsd_handle, meta_df):
    """
    Extract bottom-return intensity per ping from the downscan channel.
    The sample at slant_range ≈ instrument depth is the seafloor return
    directly below the boat. Returns list of (x, y, heading, intensity).

    Bottom detection: seeded from `inst_dep_m`, refined by local-maximum
    search to handle small metadata errors and sample reversal.
    """
    print(f"\n--- Processing Downscan (nadir fill) ---")

    meta_df = meta_df.reset_index(drop=True)
    next_offsets = compute_next_offsets(meta_df)
    file_size = os.path.getsize(RSD_FILE)

    has_speed = 'speed_ms' in meta_df.columns
    has_ping_cnt = 'ping_cnt' in meta_df.columns
    has_max_range = 'max_range' in meta_df.columns
    has_son_offset = 'son_offset' in meta_df.columns

    if DOWNSCAN_PAYLOAD_MODE_OVERRIDE and DOWNSCAN_PAYLOAD_MODE_OVERRIDE != "auto":
        payload_mode = DOWNSCAN_PAYLOAD_MODE_OVERRIDE
        print(f"  Forced downscan payload mode: {payload_mode}")
    else:
        payload_mode = choose_payload_mode(rsd_handle, meta_df, next_offsets, file_size, "Downscan")

    results = []
    skipped_slow = 0
    skipped_decode = 0
    skipped_no_nav = 0
    skipped_depth = 0

    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Downscan"):
        if pd.isna(row['index']) or pd.isna(row['data_size']):
            continue
        offset = int(row['index'])
        if offset >= file_size:
            continue

        if has_speed and MIN_SPEED_MS > 0:
            speed = row.get('speed_ms')
            if pd.notna(speed) and speed < MIN_SPEED_MS:
                skipped_slow += 1
                continue

        try:
            body_size = int(row['data_size'])
            next_off = next_offsets[idx]
            record_data = read_ping_record(rsd_handle, file_size, offset, body_size, next_off)
            if len(record_data) < 32:
                continue

            if has_ping_cnt and pd.notna(row['ping_cnt']) and row['ping_cnt'] > 0:
                n_samples = int(row['ping_cnt'])
            else:
                n_samples = FALLBACK_SAMPLE_COUNT

            son_offset = int(row['son_offset']) if (has_son_offset and pd.notna(row.get('son_offset'))) else None
            candidates = extract_payload_candidates(record_data, n_samples, son_offset=son_offset)
            raw = candidates.get(payload_mode)
            if raw is None and candidates:
                raw = next(iter(candidates.values()))
            if raw is None or len(raw) == 0:
                skipped_decode += 1
                continue

            depth = float(row['inst_dep_m']) if pd.notna(row.get('inst_dep_m')) else 0.0
            ping_max_range = float(row['max_range']) if (has_max_range and pd.notna(row['max_range'])) else MAX_RANGE_FALLBACK
            if depth <= 0 or ping_max_range <= 0:
                skipped_depth += 1
                continue

            # Seed bottom index from metadata depth, then refine via local peak.
            expected = int(np.clip((depth / ping_max_range) * len(raw), 0, len(raw) - 1))
            search_half = max(4, int(0.15 * len(raw)))
            lo_s = max(int(0.03 * len(raw)), expected - search_half)
            hi_s = min(len(raw), expected + search_half)
            if lo_s < hi_s:
                bottom_idx = lo_s + int(np.argmax(raw[lo_s:hi_s]))
            else:
                bottom_idx = expected

            # Average over a vertical window of real seafloor samples.
            win = max(1, int((DOWNSCAN_BOTTOM_WINDOW_M / ping_max_range) * len(raw) / 2))
            lo = max(0, bottom_idx - win)
            hi = min(len(raw), bottom_idx + win + 1)
            if lo >= hi:
                skipped_decode += 1
                continue

            bottom_intensity = float(np.mean(raw[lo:hi]))
            if bottom_intensity <= 0:
                continue

            nav_point = extract_nav_point(row)
            if nav_point is None:
                skipped_no_nav += 1
                continue
            x, y, h = nav_point
            results.append((x, y, h, bottom_intensity))
        except Exception:
            skipped_decode += 1
            continue

    if skipped_slow:
        print(f"  Skipped {skipped_slow} downscan pings below speed threshold")
    if skipped_decode:
        print(f"  Skipped {skipped_decode} downscan pings with decode errors")
    if skipped_no_nav:
        print(f"  Skipped {skipped_no_nav} downscan pings with missing navigation")
    if skipped_depth:
        print(f"  Skipped {skipped_depth} downscan pings with invalid depth/range")
    print(f"  Extracted {len(results)} nadir samples")
    return results


def match_downscan_levels(downscan_nadir, port_data, star_data):
    """
    Rescale downscan intensities so their median matches the combined
    port+star median. Skips the match if either side is missing.
    """
    if not downscan_nadir:
        return downscan_nadir

    ds_vals = np.array([v for (_, _, _, v) in downscan_nadir], dtype=np.float64)
    ds_vals = ds_vals[ds_vals > 0]
    if len(ds_vals) == 0:
        return downscan_nadir

    target_parts = []
    if port_data is not None:
        pv = port_data[port_data > 0]
        if pv.size:
            target_parts.append(pv.astype(np.float64))
    if star_data is not None:
        sv = star_data[star_data > 0]
        if sv.size:
            target_parts.append(sv.astype(np.float64))
    if not target_parts:
        return downscan_nadir

    ds_med = float(np.median(ds_vals))
    target_med = float(np.median(np.concatenate(target_parts)))
    if ds_med <= 0 or target_med <= 0:
        return downscan_nadir

    scale = target_med / ds_med
    print(f"  Downscan level match: x{scale:.3f} (ds_med={ds_med:.1f}, target={target_med:.1f})")
    return [(x, y, h, v * scale) for (x, y, h, v) in downscan_nadir]


def paint_nadir_strip(intensity, ux, uy, head, strip_width_m, pixel_size,
                      min_x, max_y, width, height, raster, raster_filled):
    """
    Paint a short perpendicular strip centered on (ux, uy) with uniform intensity.
    Uses "first" overlap policy so already-painted sidescan pixels are preserved
    and only the nadir gap gets filled.
    """
    if strip_width_m <= 0:
        return

    n_bins = max(2, int(strip_width_m / pixel_size) + 1)
    half_w = strip_width_m / 2.0

    # Perpendicular to heading; extend -half_w to +half_w
    angle_rad = np.radians(head) + (np.pi / 2)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    r_bins = np.linspace(-half_w, half_w, n_bins)
    p_x = ux + r_bins * sin_a
    p_y = uy + r_bins * cos_a

    idx_x = ((p_x - min_x) / pixel_size).astype(np.int32)
    idx_y = ((max_y - p_y) / pixel_size).astype(np.int32)

    valid = (idx_x >= 0) & (idx_x < width) & (idx_y >= 0) & (idx_y < height)
    if not np.any(valid):
        return

    pix_x = idx_x[valid]
    pix_y = idx_y[valid]

    flat_idx = pix_y.astype(np.int64) * width + pix_x.astype(np.int64)
    _, keep_idx = np.unique(flat_idx, return_index=True)
    pix_x = pix_x[keep_idx]
    pix_y = pix_y[keep_idx]

    writable = ~raster_filled[pix_y, pix_x]
    if not np.any(writable):
        return
    pix_x = pix_x[writable]
    pix_y = pix_y[writable]

    raster[pix_y, pix_x] = intensity
    raster_filled[pix_y, pix_x] = True


def percentile_stretch(data, low_pct=2, high_pct=98):
    """
    Percentile-based contrast stretch to uint8.
    Maps [low_percentile, high_percentile] -> [1, 255], with 0 reserved for nodata.
    """
    stretch_input = log_compress(data, LOG_COMPRESSION_SCALE) if APPLY_LOG_COMPRESSION else data
    valid = stretch_input[stretch_input > 0]
    if len(valid) == 0:
        return np.zeros_like(stretch_input, dtype=np.uint8)

    if FIXED_STRETCH_LO is not None and FIXED_STRETCH_HI is not None:
        lo = FIXED_STRETCH_LO
        hi = FIXED_STRETCH_HI
    else:
        lo = np.percentile(valid, low_pct)
        hi = np.percentile(valid, high_pct)

    if hi <= lo:
        hi = lo + 1

    stretched = (stretch_input - lo) / (hi - lo) * 254.0 + 1.0
    stretched = np.where(stretch_input > 0, stretched, 0)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def fill_raster_gaps(raster, passes=3):
    """
    Iteratively fill single-pixel gaps using the mean of non-zero neighbors.
    Each pass fills gaps that have at least 3 of 8 neighbors with data,
    so N passes can close gaps up to ~N pixels wide.
    """
    filled = raster.copy()
    for _ in range(passes):
        gaps = filled == 0
        if not np.any(gaps):
            break

        # Compute neighborhood mean excluding zeros
        neighbor_sum = uniform_filter(filled, size=3, mode='constant', cval=0.0)
        neighbor_count = uniform_filter((filled > 0).astype(np.float32), size=3, mode='constant', cval=0.0)

        # Fill where gap pixel has at least ~3 of 9 window cells with data
        fillable = gaps & (neighbor_count >= 0.33)
        safe_count = np.where(neighbor_count > 0, neighbor_count, 1.0)
        neighbor_mean = neighbor_sum / safe_count

        filled[fillable] = neighbor_mean[fillable]

    return filled


def paint_scan_line(line_data, ux, uy, head, side_sign, pixel_size,
                    min_x, max_y, width, height, raster, raster_filled,
                    overlap_policy="first"):
    """
    Paint a single scan line into the raster grid.

    Overlap policies:
      "first" — keep earliest painted pixel.
      "last"  — overwrite with newest painted pixel.
    """
    angle_rad = np.radians(head) + (np.pi / 2 * side_sign)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    # Double sample density along the scan line to ensure no grid projection gaps
    dense_len = len(line_data) * 2
    if dense_len == 0:
        return
    dense_idx = np.linspace(0, len(line_data) - 1, dense_len)
    dense_data = np.interp(dense_idx, np.arange(len(line_data)), line_data)

    r_bins = dense_idx * pixel_size
    p_x = ux + r_bins * sin_a
    p_y = uy + r_bins * cos_a

    idx_x = ((p_x - min_x) / pixel_size).astype(np.int32)
    idx_y = ((max_y - p_y) / pixel_size).astype(np.int32)

    valid = (idx_x >= 0) & (idx_x < width) & (idx_y >= 0) & (idx_y < height) & (dense_data > 0)
    if not np.any(valid):
        return

    pix_x = idx_x[valid]
    pix_y = idx_y[valid]
    pix_vals = dense_data[valid]
    flat_idx = pix_y.astype(np.int64) * width + pix_x.astype(np.int64)

    if overlap_policy == "first":
        _, keep_idx = np.unique(flat_idx, return_index=True)
    elif overlap_policy == "last":
        _, rev_idx = np.unique(flat_idx[::-1], return_index=True)
        keep_idx = (len(flat_idx) - 1) - rev_idx
    else:
        raise ValueError(f"Unsupported overlap policy: {overlap_policy}")

    pix_x = pix_x[keep_idx]
    pix_y = pix_y[keep_idx]
    pix_vals = pix_vals[keep_idx]

    if overlap_policy == "first":
        writable = ~raster_filled[pix_y, pix_x]
        if not np.any(writable):
            return
        pix_x = pix_x[writable]
        pix_y = pix_y[writable]
        pix_vals = pix_vals[writable]

    raster[pix_y, pix_x] = pix_vals
    raster_filled[pix_y, pix_x] = True


def save_geotiff(data, nav_data, side_sign, pixel_size, filename,
                 raster_crs=None, transformer=None):
    """
    Projects waterfall strips into a georeferenced raster.
    Uses interpolated gap filling while preserving the chosen overlap policy.
    """
    if data is None or not nav_data:
        return
    print(f"  Mapping to {filename}...")

    # Project nav_data if transformer provided (geographic -> UTM)
    if transformer is not None:
        projected_nav = []
        for x_lon, y_lat, head in nav_data:
            px, py = transformer.transform(x_lon, y_lat)
            projected_nav.append((px, py, head))
        nav_data = projected_nav

    all_x = [n[0] for n in nav_data]
    all_y = [n[1] for n in nav_data]
    if not all_x:
        return

    buffer = 20
    min_x, max_x = min(all_x) - buffer, max(all_x) + buffer
    min_y, max_y = min(all_y) - buffer, max(all_y) + buffer

    width = int((max_x - min_x) / pixel_size)
    height = int((max_y - min_y) / pixel_size)

    raster = np.zeros((height, width), dtype=np.float32)
    raster_filled = np.zeros((height, width), dtype=bool)

    print(f"  Grid: {width} x {height} pixels")

    for i, (ux, uy, head) in enumerate(tqdm(nav_data, desc="Georeferencing")):
        if i >= len(data):
            break
        line_data = data[i].astype(np.float32)

        # Paint current ping
        paint_scan_line(line_data, ux, uy, head, side_sign, pixel_size,
                        min_x, max_y, width, height, raster, raster_filled,
                        overlap_policy=OVERLAP_POLICY)

        # Interpolated gap filling between adjacent pings
        if FILL_GAPS and i > 0:
            prev_ux, prev_uy, prev_head = nav_data[i - 1]
            ping_dist = np.sqrt((ux - prev_ux) ** 2 + (uy - prev_uy) ** 2)

            if 0 < ping_dist <= MAX_FILL_DISTANCE:
                n_fills = max(1, int(np.ceil(ping_dist / pixel_size)))
                prev_data = data[i - 1].astype(np.float32)

                # Ensure both lines are same length for interpolation
                max_len = max(len(prev_data), len(line_data))
                if len(prev_data) < max_len:
                    prev_data = np.pad(prev_data, (0, max_len - len(prev_data)))
                if len(line_data) < max_len:
                    line_data = np.pad(line_data, (0, max_len - len(line_data)))

                for k in range(1, n_fills):
                    t = k / n_fills
                    interp_x = prev_ux + t * (ux - prev_ux)
                    interp_y = prev_uy + t * (uy - prev_uy)
                    interp_head = circular_lerp(prev_head, head, t)
                    # Use nearest ping's data (avoids intensity blending artifacts)
                    fill_data = line_data if t >= 0.5 else prev_data

                    paint_scan_line(fill_data, interp_x, interp_y, interp_head,
                                    side_sign, pixel_size,
                                    min_x, max_y, width, height, raster, raster_filled,
                                    overlap_policy=OVERLAP_POLICY)

    # Fill remaining single-pixel gaps from scan line divergence at far range
    if GAP_FILL_PASSES > 0:
        raster = fill_raster_gaps(raster, GAP_FILL_PASSES)

    # Percentile contrast stretch to uint8
    output = percentile_stretch(raster, STRETCH_LOW_PCT, STRETCH_HIGH_PCT)

    transform = from_origin(min_x, max_y, pixel_size, pixel_size)

    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=raster_crs,
        transform=transform,
        nodata=0,
        compress='lzw'
    ) as dst:
        dst.write(output, 1)


def save_merged_geotiff(port_data, port_nav, star_data, star_nav,
                        pixel_size, filename, raster_crs=None, transformer=None,
                        downscan_nadir=None):
    """
    Merges port and starboard into a single georeferenced mosaic.
    Overlap is resolved by the configured paint policy instead of blending layers.
    If downscan_nadir is supplied, bottom-return intensities are painted as
    narrow along-track strips into the remaining nadir gap (gap-only, never
    overwrites sidescan pixels).
    """
    if port_data is None and star_data is None:
        return
    print(f"  Mapping merged mosaic to {filename}...")

    # Project all nav data
    def project_nav(nav):
        if nav is None or transformer is None:
            return nav
        return [(transformer.transform(x, y)[0], transformer.transform(x, y)[1], h)
                for x, y, h in nav]

    p_nav = project_nav(port_nav) if port_nav else []
    s_nav = project_nav(star_nav) if star_nav else []

    # Combined bounding box
    all_x = [n[0] for n in p_nav] + [n[0] for n in s_nav]
    all_y = [n[1] for n in p_nav] + [n[1] for n in s_nav]
    if not all_x:
        return

    buffer = 20
    min_x, max_x = min(all_x) - buffer, max(all_x) + buffer
    min_y, max_y = min(all_y) - buffer, max(all_y) + buffer

    width = int((max_x - min_x) / pixel_size)
    height = int((max_y - min_y) / pixel_size)

    raster = np.zeros((height, width), dtype=np.float32)
    raster_filled = np.zeros((height, width), dtype=bool)

    print(f"  Grid: {width} x {height} pixels")

    # Paint both sides into the same grid
    for side_data, side_nav, side_sign, side_name in [
        (port_data, p_nav, -1, "Port"),
        (star_data, s_nav, 1, "Starboard")
    ]:
        if side_data is None or not side_nav:
            continue

        for i, (ux, uy, head) in enumerate(tqdm(side_nav, desc=f"Merging {side_name}")):
            if i >= len(side_data):
                break
            line_data = side_data[i].astype(np.float32)

            paint_scan_line(line_data, ux, uy, head, side_sign, pixel_size,
                            min_x, max_y, width, height, raster, raster_filled,
                            overlap_policy=OVERLAP_POLICY)

            # Interpolated gap fill
            if FILL_GAPS and i > 0:
                prev_ux, prev_uy, prev_head = side_nav[i - 1]
                ping_dist = np.sqrt((ux - prev_ux) ** 2 + (uy - prev_uy) ** 2)

                if 0 < ping_dist <= MAX_FILL_DISTANCE:
                    n_fills = max(1, int(np.ceil(ping_dist / pixel_size)))
                    prev_data = side_data[i - 1].astype(np.float32)

                    max_len = max(len(prev_data), len(line_data))
                    if len(prev_data) < max_len:
                        prev_data = np.pad(prev_data, (0, max_len - len(prev_data)))
                    if len(line_data) < max_len:
                        line_data = np.pad(line_data, (0, max_len - len(line_data)))

                    for k in range(1, n_fills):
                        t = k / n_fills
                        interp_x = prev_ux + t * (ux - prev_ux)
                        interp_y = prev_uy + t * (uy - prev_uy)
                        interp_head = circular_lerp(prev_head, head, t)
                        fill_data = line_data if t >= 0.5 else prev_data

                        paint_scan_line(fill_data, interp_x, interp_y, interp_head,
                                        side_sign, pixel_size,
                                        min_x, max_y, width, height, raster, raster_filled,
                                        overlap_policy=OVERLAP_POLICY)

    # Downscan nadir fill: paint bottom-return intensity along the boat track
    # into gaps only (raster_filled blocks overwrite of real sidescan data).
    if downscan_nadir:
        print("  Painting downscan nadir fill...")
        if transformer is not None:
            ds_proj = []
            for x, y, h, v in downscan_nadir:
                px, py = transformer.transform(x, y)
                ds_proj.append((px, py, h, v))
        else:
            ds_proj = list(downscan_nadir)

        prev = None
        for entry in tqdm(ds_proj, desc="Downscan nadir"):
            ux, uy, head, intensity = entry
            paint_nadir_strip(intensity, ux, uy, head,
                              DOWNSCAN_STRIP_WIDTH_M, pixel_size,
                              min_x, max_y, width, height, raster, raster_filled)

            # Along-track densification so the strip is continuous between pings
            if prev is not None:
                px, py, ph, pv = prev
                dist = np.sqrt((ux - px) ** 2 + (uy - py) ** 2)
                if 0 < dist <= MAX_FILL_DISTANCE:
                    n_fills = max(1, int(np.ceil(dist / pixel_size)))
                    for k in range(1, n_fills):
                        t = k / n_fills
                        ix = px + t * (ux - px)
                        iy = py + t * (uy - py)
                        ih = circular_lerp(ph, head, t)
                        iv = pv + t * (intensity - pv)
                        paint_nadir_strip(iv, ix, iy, ih,
                                          DOWNSCAN_STRIP_WIDTH_M, pixel_size,
                                          min_x, max_y, width, height, raster, raster_filled)
            prev = entry

    if GAP_FILL_PASSES > 0:
        raster = fill_raster_gaps(raster, GAP_FILL_PASSES)

    output = percentile_stretch(raster, STRETCH_LOW_PCT, STRETCH_HIGH_PCT)

    transform = from_origin(min_x, max_y, pixel_size, pixel_size)

    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=raster_crs,
        transform=transform,
        nodata=0,
        compress='lzw'
    ) as dst:
        dst.write(output, 1)


def process_single_rsd(file_path):
    global RSD_FILE, RSD_BASENAME, RSD_PARENT_DIR, OUTPUT_BASE_DIR, META_DIR, OUTPUT_DIR, ALL_META_FILE
    
    RSD_FILE = file_path
    RSD_BASENAME = os.path.splitext(os.path.basename(RSD_FILE))[0]
    RSD_PARENT_DIR = os.path.dirname(RSD_FILE)
    OUTPUT_BASE_DIR = os.path.join(RSD_PARENT_DIR, f"garmin_output_{RSD_BASENAME}")
    META_DIR = os.path.join(OUTPUT_BASE_DIR, "meta")
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "processed")
    ALL_META_FILE = os.path.join(META_DIR, "All-Garmin-Sonar-MetaData.csv")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Processing RSD: {RSD_BASENAME}")
    print("=" * 60)
    print(f"  Resolution:  {OUTPUT_RESOLUTION * 100:.0f} cm/pixel")
    print(f"  TVG:         {APPLY_TVG} (spreading={TVG_SPREADING_DB} dB, absorption={ABSORPTION_COEFF} dB/m)")
    print(f"  Despeckle:   {APPLY_DESPECKLE} ({DESPECKLE_SIZE}x{DESPECKLE_SIZE} median)")
    print(f"  Lee filter:  {APPLY_LEE_FILTER} ({LEE_WINDOW_SIZE}x{LEE_WINDOW_SIZE})")
    print(f"  Texture:     {TEXTURE_WINDOW_SIZE}px window (~{TEXTURE_WINDOW_SIZE * OUTPUT_RESOLUTION:.2f}m)")
    print(f"  Nadir mask:  {NADIR_MASK_BINS} bins ({NADIR_MASK_BINS * OUTPUT_RESOLUTION:.2f}m)")
    print(f"  Nadir depth: factor={NADIR_ALTITUDE_FACTOR}")
    print(f"  Gap fill:    {FILL_GAPS} (interpolation, max {MAX_FILL_DISTANCE}m)")
    print(f"  Overlap:     {OVERLAP_POLICY}")
    print(f"  Min speed:   {MIN_SPEED_MS} m/s")
    if FIXED_STRETCH_LO is not None:
        print(f"  Stretch:     FIXED LO={FIXED_STRETCH_LO}, HI={FIXED_STRETCH_HI}")
    else:
        print(f"  Stretch:     DYNAMIC {STRETCH_LOW_PCT}-{STRETCH_HIGH_PCT} percentile")
    print(f"  Log comp:    {APPLY_LOG_COMPRESSION} (scale={LOG_COMPRESSION_SCALE})")
    print(f"  Row balance: {APPLY_ROW_BALANCE}")
    print()

    # Ensure metadata exists
    print("Checking metadata files...")
    if not ensure_metadata_exists(RSD_FILE, META_DIR):
        print(f"\nFailed to generate or find metadata files for {RSD_BASENAME}. Skipping.")
        return
    print()

    print("Loading metadata...")
    ss_port = pd.read_csv(os.path.join(META_DIR, "B002_ss_port_meta.csv"))
    ss_star = pd.read_csv(os.path.join(META_DIR, "B003_ss_star_meta.csv"))

    if USE_PORT_CHANNEL_OVERRIDE and os.path.exists(ALL_META_FILE):
        all_meta = pd.read_csv(ALL_META_FILE)
        channel_vals = pd.to_numeric(all_meta.get('channel_id'), errors='coerce')
        ss_port_alt = all_meta[channel_vals == float(PORT_CHANNEL_OVERRIDE_ID)].copy()
        ss_port_alt = ss_port_alt.reset_index(drop=True)

        star_channel = None
        if 'channel_id' in ss_star.columns:
            ch_vals = pd.to_numeric(ss_star['channel_id'], errors='coerce').dropna()
            if len(ch_vals) > 0:
                star_channel = int(ch_vals.iloc[0])

        if star_channel is not None and star_channel == PORT_CHANNEL_OVERRIDE_ID:
            print(f"  WARNING: Port channel override {PORT_CHANNEL_OVERRIDE_ID} "
                  f"matches starboard's channel_id — skipping override to avoid "
                  f"mirrored mosaic.")
        elif len(ss_port_alt) > 0:
            ss_port = ss_port_alt
            print(f"  Port source override: channel_id={PORT_CHANNEL_OVERRIDE_ID} from All-Garmin-Sonar-MetaData.csv")
        else:
            print(f"  WARNING: Port channel override {PORT_CHANNEL_OVERRIDE_ID} had no rows; using default B002 metadata")

    # Detect coordinate type and auto-determine UTM CRS
    coord_type = 'projected'
    output_crs = None
    transformer = None
    cols = ss_port.columns

    if 'lon' in cols and 'lat' in cols:
        coord_type = 'geographic'
        all_lats = pd.concat([ss_port['lat'], ss_star['lat']]).dropna()
        all_lons = pd.concat([ss_port['lon'], ss_star['lon']]).dropna()
        epsg = determine_utm_epsg(all_lats.values, all_lons.values)
        output_crs = CRS.from_epsg(epsg)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        print(f"  Coordinate type: Geographic (lat/lon)")
        print(f"  Auto-detected UTM zone: EPSG:{epsg}")
    elif 'utm_zone' in cols:
        try:
            zone = int(ss_port['utm_zone'].iloc[0])
            output_crs = CRS.from_epsg(32600 + zone)
            coord_type = 'projected'
            print(f"  Coordinate type: Projected (UTM zone {zone})")
        except Exception:
            output_crs = None
            print(f"  Coordinate type: Projected (unknown CRS)")
    else:
        print(f"  Coordinate type: Projected (unknown CRS)")

    print(f"  Port pings:  {len(ss_port)}")
    print(f"  Star pings:  {len(ss_star)}")

    with open(RSD_FILE, 'rb') as f:
        wf_p, tex_p, nav_p, sign_p = process_side(
            f, ss_port, "Port", -1, transformer=transformer,
            payload_mode_override=PAYLOAD_MODE_OVERRIDE_PORT,
            reverse_samples=REVERSE_PORT_SAMPLES,
            heading_offset_deg=PORT_HEADING_OFFSET_DEG,
        )
        wf_s, tex_s, nav_s, sign_s = process_side(
            f, ss_star, "Starboard", 1, transformer=transformer,
            payload_mode_override=PAYLOAD_MODE_OVERRIDE_STARBOARD,
            reverse_samples=REVERSE_STARBOARD_SAMPLES,
            heading_offset_deg=STARBOARD_HEADING_OFFSET_DEG,
        )

    # Save individual side GeoTIFFs
    print("\n--- Saving Individual GeoTIFFs ---")
    save_geotiff(
        wf_p, nav_p, sign_p, OUTPUT_RESOLUTION,
        os.path.join(OUTPUT_DIR, "port_intensity.tif"),
        raster_crs=output_crs,
        transformer=transformer if coord_type == 'geographic' else None,
    )
    save_geotiff(
        wf_s, nav_s, sign_s, OUTPUT_RESOLUTION,
        os.path.join(OUTPUT_DIR, "star_intensity.tif"),
        raster_crs=output_crs,
        transformer=transformer if coord_type == 'geographic' else None,
    )

    # Save merged port+starboard mosaic
    print("\n--- Saving Merged Mosaic ---")
    wf_p_matched, wf_s_matched = match_side_levels(wf_p, wf_s)

    downscan_nadir = None
    if APPLY_DOWNSCAN_NADIR_FILL:
        ds_meta_path = os.path.join(META_DIR, DOWNSCAN_META_NAME)
        if os.path.exists(ds_meta_path):
            ds_meta = pd.read_csv(ds_meta_path)
            with open(RSD_FILE, 'rb') as f:
                downscan_nadir = process_downscan_nadir(f, ds_meta)
            downscan_nadir = match_downscan_levels(downscan_nadir, wf_p_matched, wf_s_matched)
        else:
            print(f"  Downscan metadata not found at {ds_meta_path}, skipping nadir fill")

    save_merged_geotiff(
        wf_p_matched, nav_p, wf_s_matched, nav_s,
        OUTPUT_RESOLUTION,
        os.path.join(OUTPUT_DIR, "intensity.tif"),
        raster_crs=output_crs,
        transformer=transformer if coord_type == 'geographic' else None,
        downscan_nadir=downscan_nadir,
    )
    save_merged_geotiff(
        tex_p, nav_p, tex_s, nav_s,
        OUTPUT_RESOLUTION,
        os.path.join(OUTPUT_DIR, "texture.tif"),
        raster_crs=output_crs,
        transformer=transformer if coord_type == 'geographic' else None,
    )

    print("\n" + "=" * 60)
    print("Done! Output files saved to:", OUTPUT_DIR)
    print("=" * 60)



# === Execution ===
if __name__ == "__main__":
    print("=" * 60)
    print("Garmin Sidescan Mosaic Generator")
    print("=" * 60)

    if not os.path.exists(RSD_FILE):
        print(f"Input RSD file does not exist: {RSD_FILE}")
        print("Set GARMIN_RSD_FILE environment variable or edit RSD_FILE in the script.")
        sys.exit(1)

    process_single_rsd(RSD_FILE)


