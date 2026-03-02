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
import math
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
from pyproj import CRS, Transformer


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


# === Configuration ===
RSD_FILE = r"C:\Users\jason\Downloads\Side.RSD"

RSD_BASENAME = os.path.splitext(os.path.basename(RSD_FILE))[0]
RSD_PARENT_DIR = os.path.dirname(RSD_FILE)
OUTPUT_BASE_DIR = os.path.join(RSD_PARENT_DIR, f"garmin_output_{RSD_BASENAME}")
META_DIR = os.path.join(OUTPUT_BASE_DIR, "meta")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "processed")

# Processing Parameters
OUTPUT_RESOLUTION = 0.05    # 5cm per pixel
TEXTURE_WINDOW_SIZE = 15    # Texture analysis window (~1.5m at 5cm res)
MAX_RANGE_FALLBACK = 30.0   # Fallback if per-ping max_range not in metadata

# Radiometric Corrections
APPLY_TVG = True
TVG_SPREADING_DB = 30.0     # Spreading loss compensation (20-40 dB typical for sidescan)
ABSORPTION_COEFF = 0.05     # dB/m (0.03-0.06 freshwater, 0.1-0.2 seawater at 455kHz)
APPLY_DESPECKLE = True
DESPECKLE_SIZE = 3           # Median filter kernel (3x3)

# Navigation
HEADING_SMOOTH_WINDOW = 100  # Heading smoothing window (pings)
FILTER_OUTLIERS = False      # Remove heading/position outlier pings
MIN_SPEED_MS = 0.3           # Skip pings below this speed (m/s), 0 to disable

# Gap filling
FILL_GAPS = True
MAX_FILL_DISTANCE = 2.0     # Max interpolation distance between pings (meters)

# Nadir zone
NADIR_MASK_BINS = 5          # Zero out first N ground-range bins (nadir artifact)

# Raster gap fill (post-processing to close single-pixel gaps from scan line divergence)
GAP_FILL_PASSES = 3          # Iterative neighbor-mean fill passes (0 to disable, 2-3 typical)

# Contrast stretch
STRETCH_LOW_PCT = 2          # Lower percentile for output contrast stretch
STRETCH_HIGH_PCT = 98        # Upper percentile for output contrast stretch

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    Masks nadir zone bins where the correction is most unstable.
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
    ground_ranges = np.sqrt(slant_ranges[valid_mask] ** 2 - altitude_m ** 2)
    valid_intensities = intensities[valid_mask]

    if len(ground_ranges) < 2:
        return np.array([], dtype=np.float32)

    max_ground_range = np.max(ground_ranges)
    grid_samples = int(max_ground_range / res_m)
    if grid_samples == 0:
        return np.zeros(1, dtype=np.float32)

    grid_ranges = np.linspace(0, max_ground_range, grid_samples)

    f = interp1d(ground_ranges, valid_intensities, kind='linear',
                 bounds_error=False, fill_value=0)
    corrected_line = f(grid_ranges).astype(np.float32)

    # Mask nadir zone (first few bins are geometrically unstable)
    if nadir_mask_bins > 0 and len(corrected_line) > nadir_mask_bins:
        corrected_line[:nadir_mask_bins] = 0

    return corrected_line


def compute_egn_curve(waterfall):
    """
    Calculates Empirical Gain Normalization (EGN) curve.
    Uses column-wise mean of non-zero pixels, double-smoothed
    with a wide kernel to isolate the low-frequency beam pattern.
    """
    col_sums = waterfall.sum(axis=0)
    col_counts = (waterfall != 0).sum(axis=0).astype(np.float64) + 1e-6
    egn_curve = col_sums / col_counts

    kernel_size = min(200, len(egn_curve) // 2)
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(egn_curve, kernel, mode='same')
        return np.convolve(smoothed, kernel, mode='same')
    return egn_curve


def apply_egn(waterfall, egn_curve):
    """
    Applies the inverse beam pattern for even illumination.
    Returns float32 (no uint8 clipping to preserve dynamic range).
    """
    global_mean = np.mean(egn_curve)
    safe_curve = np.where(egn_curve < 1e-6, 1e-6, egn_curve)
    gain_factors = global_mean / safe_curve
    return (waterfall * gain_factors).astype(np.float32)


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


def smooth_and_filter_nav(nav_data, waterfall_rows):
    """
    Smooth headings and filter GPS/heading outliers.
    Uses COG computed from projected coordinates with a wide window,
    then applies circular moving-average smoothing.
    """
    if len(nav_data) < 10:
        return nav_data, waterfall_rows

    x_vals = np.array([n[0] for n in nav_data])
    y_vals = np.array([n[1] for n in nav_data])
    headings = np.array([n[2] for n in nav_data])

    # Recompute COG from projected coordinates with a wider window
    # to reduce GPS jitter (pingverter uses consecutive points only)
    cog_headings = np.zeros_like(headings)
    step = 5
    for i in range(len(headings)):
        p_prev = max(0, i - step)
        p_next = min(len(headings) - 1, i + step)

        if p_prev == p_next:
            cog_headings[i] = headings[i]
            continue

        dx = x_vals[p_next] - x_vals[p_prev]
        dy = y_vals[p_next] - y_vals[p_prev]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 0.5:
            cog_headings[i] = headings[i]
        else:
            cog_headings[i] = np.degrees(np.arctan2(dx, dy)) % 360

    # Circular moving average
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
    else:
        heading_smooth = cog_headings

    # Outlier detection
    heading_diff = np.abs(headings - heading_smooth)
    heading_diff = np.minimum(heading_diff, 360 - heading_diff)

    dx = np.diff(x_vals, prepend=x_vals[0])
    dy = np.diff(y_vals, prepend=y_vals[0])
    distances = np.sqrt(dx ** 2 + dy ** 2)
    median_dist = np.median(distances[distances > 0])

    heading_threshold = 30.0
    position_threshold = max(5 * median_dist, 2.0)
    valid_mask = (heading_diff < heading_threshold) & (distances < position_threshold)
    valid_mask[:5] = True

    removed = np.sum(~valid_mask)

    if FILTER_OUTLIERS:
        if removed > 0:
            print(f"    Removed {removed} outlier pings ({100 * removed / len(nav_data):.1f}%)")
        filtered_nav = [(x_vals[i], y_vals[i], heading_smooth[i])
                        for i in range(len(nav_data)) if valid_mask[i]]
        filtered_wf = [waterfall_rows[i]
                       for i in range(len(waterfall_rows)) if valid_mask[i]]
        return filtered_nav, filtered_wf
    else:
        if removed > 0:
            print(f"    [Info] Potential outliers detected: {removed} (filtering disabled)")
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


def process_side(rsd_handle, meta_df, side_name, sign):
    """
    Reads and processes one side (port or starboard) of sidescan data.

    Pipeline order: TVG -> Slant Range Correction -> Stack -> EGN -> Despeckle

    Uses per-ping metadata (max_range, pixM, ping_cnt) when available
    instead of hardcoded global values.
    """
    print(f"\n--- Processing {side_name} Side ---")

    waterfall_rows = []
    nav_data = []

    file_size = os.path.getsize(RSD_FILE)

    # Check which per-ping metadata columns are available
    has_max_range = 'max_range' in meta_df.columns
    has_son_offset = 'son_offset' in meta_df.columns
    has_speed = 'speed_ms' in meta_df.columns

    if has_max_range:
        print(f"  Using per-ping max_range from metadata")
    else:
        print(f"  Using fallback max_range = {MAX_RANGE_FALLBACK}m")

    skipped_slow = 0

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
            ping_size = int(row['data_size'])
            record_data = rsd_handle.read(ping_size)

            # Determine header size: use son_offset from metadata if available
            if has_son_offset and pd.notna(row['son_offset']):
                header_size = int(row['son_offset'])
            else:
                header_size = 113

            if len(record_data) < header_size:
                continue

            raw_intensities = np.frombuffer(record_data[header_size:], dtype=np.uint8)
            if len(raw_intensities) == 0:
                continue

            # Per-ping range from metadata, or fallback
            if has_max_range and pd.notna(row['max_range']):
                ping_max_range = float(row['max_range'])
            else:
                ping_max_range = MAX_RANGE_FALLBACK

            # Compute slant ranges for this ping's samples
            n_samples = len(raw_intensities)
            slant_ranges = np.linspace(0, ping_max_range, n_samples)

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
                waterfall_rows.append(corrected)

                # Extract coordinates
                utmx, utmy = None, None
                if 'lon' in meta_df.columns and 'lat' in meta_df.columns \
                        and pd.notna(row['lon']) and pd.notna(row['lat']):
                    utmx = row['lon']
                    utmy = row['lat']
                elif 'e' in meta_df.columns and 'n' in meta_df.columns \
                        and pd.notna(row['e']) and pd.notna(row['n']):
                    utmx = row['e']
                    utmy = row['n']

                if utmx is not None:
                    heading = row['instr_heading']
                    nav_data.append((utmx, utmy, heading))

        except Exception:
            continue

    if skipped_slow > 0:
        print(f"  Skipped {skipped_slow} near-stationary pings (speed < {MIN_SPEED_MS} m/s)")

    if not waterfall_rows:
        print("  No valid data found.")
        return None, None, None, None

    # Smooth headings and filter outliers
    print("  Smoothing navigation data...")
    nav_data, waterfall_rows = smooth_and_filter_nav(nav_data, waterfall_rows)

    if not waterfall_rows:
        print("  No valid data after filtering.")
        return None, None, None, None

    # Stack into waterfall (float32 to preserve TVG dynamic range)
    print("  Stacking waterfall...")
    max_len = max(len(r) for r in waterfall_rows)
    waterfall = np.zeros((len(waterfall_rows), max_len), dtype=np.float32)
    for i, r in enumerate(waterfall_rows):
        waterfall[i, :len(r)] = r

    # EGN (beam pattern normalization) - applied BEFORE despeckle
    print("  Applying Gain Normalization...")
    egn_curve = compute_egn_curve(waterfall)
    waterfall = apply_egn(waterfall, egn_curve)

    # Despeckle AFTER EGN (preserves beam pattern statistics for EGN)
    if APPLY_DESPECKLE:
        print("  Applying despeckle filter...")
        waterfall = median_filter(waterfall, size=DESPECKLE_SIZE).astype(np.float32)

    # Texture analysis on corrected waterfall
    texture_img = calculate_texture(waterfall, TEXTURE_WINDOW_SIZE)

    return waterfall, texture_img, nav_data, 1 if sign == 1 else -1


def percentile_stretch(data, low_pct=2, high_pct=98):
    """
    Percentile-based contrast stretch to uint8.
    Maps [low_percentile, high_percentile] -> [1, 255], with 0 reserved for nodata.
    """
    valid = data[data > 0]
    if len(valid) == 0:
        return np.zeros_like(data, dtype=np.uint8)

    lo = np.percentile(valid, low_pct)
    hi = np.percentile(valid, high_pct)

    if hi <= lo:
        hi = lo + 1

    stretched = (data - lo) / (hi - lo) * 254.0 + 1.0
    stretched = np.where(data > 0, stretched, 0)
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
                    min_x, max_y, width, height, raster_sum, raster_count):
    """Paint a single scan line into the accumulator grid."""
    angle_rad = np.radians(head) + (np.pi / 2 * side_sign)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    r_bins = np.arange(len(line_data)) * pixel_size
    p_x = ux + r_bins * sin_a
    p_y = uy + r_bins * cos_a

    idx_x = ((p_x - min_x) / pixel_size).astype(np.int32)
    idx_y = ((max_y - p_y) / pixel_size).astype(np.int32)

    valid = (idx_x >= 0) & (idx_x < width) & (idx_y >= 0) & (idx_y < height) & (line_data > 0)

    raster_sum[idx_y[valid], idx_x[valid]] += line_data[valid]
    raster_count[idx_y[valid], idx_x[valid]] += 1


def save_geotiff(data, nav_data, side_sign, pixel_size, filename,
                 raster_crs=None, transformer=None):
    """
    Projects waterfall strips into a georeferenced raster.
    Uses overlap averaging and interpolated gap filling.
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

    raster_sum = np.zeros((height, width), dtype=np.float64)
    raster_count = np.zeros((height, width), dtype=np.uint16)

    print(f"  Grid: {width} x {height} pixels")

    for i, (ux, uy, head) in enumerate(tqdm(nav_data, desc="Georeferencing")):
        if i >= len(data):
            break
        line_data = data[i].astype(np.float32)

        # Paint current ping
        paint_scan_line(line_data, ux, uy, head, side_sign, pixel_size,
                        min_x, max_y, width, height, raster_sum, raster_count)

        # Interpolated gap filling between adjacent pings
        if FILL_GAPS and i > 0:
            prev_ux, prev_uy, prev_head = nav_data[i - 1]
            ping_dist = np.sqrt((ux - prev_ux) ** 2 + (uy - prev_uy) ** 2)

            if 0 < ping_dist <= MAX_FILL_DISTANCE:
                n_fills = max(1, int(ping_dist / pixel_size))
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
                                    min_x, max_y, width, height,
                                    raster_sum, raster_count)

    # Compute average
    raster = np.zeros((height, width), dtype=np.float32)
    mask = raster_count > 0
    raster[mask] = (raster_sum[mask] / raster_count[mask]).astype(np.float32)

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
                        pixel_size, filename, raster_crs=None, transformer=None):
    """
    Merges port and starboard into a single georeferenced mosaic.
    Overlap in the nadir zone is averaged for smooth blending.
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

    raster_sum = np.zeros((height, width), dtype=np.float64)
    raster_count = np.zeros((height, width), dtype=np.uint16)

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
                            min_x, max_y, width, height, raster_sum, raster_count)

            # Interpolated gap fill
            if FILL_GAPS and i > 0:
                prev_ux, prev_uy, prev_head = side_nav[i - 1]
                ping_dist = np.sqrt((ux - prev_ux) ** 2 + (uy - prev_uy) ** 2)

                if 0 < ping_dist <= MAX_FILL_DISTANCE:
                    n_fills = max(1, int(ping_dist / pixel_size))
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
                                        min_x, max_y, width, height,
                                        raster_sum, raster_count)

    # Average and fill gaps
    raster = np.zeros((height, width), dtype=np.float32)
    mask = raster_count > 0
    raster[mask] = (raster_sum[mask] / raster_count[mask]).astype(np.float32)

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


# === Execution ===
if __name__ == "__main__":
    print("=" * 60)
    print("Garmin Sidescan Mosaic Generator")
    print("=" * 60)
    print(f"  Resolution:  {OUTPUT_RESOLUTION * 100:.0f} cm/pixel")
    print(f"  TVG:         {APPLY_TVG} (spreading={TVG_SPREADING_DB} dB, absorption={ABSORPTION_COEFF} dB/m)")
    print(f"  Despeckle:   {APPLY_DESPECKLE} ({DESPECKLE_SIZE}x{DESPECKLE_SIZE} median)")
    print(f"  Texture:     {TEXTURE_WINDOW_SIZE}px window (~{TEXTURE_WINDOW_SIZE * OUTPUT_RESOLUTION:.2f}m)")
    print(f"  Nadir mask:  {NADIR_MASK_BINS} bins ({NADIR_MASK_BINS * OUTPUT_RESOLUTION:.2f}m)")
    print(f"  Gap fill:    {FILL_GAPS} (interpolation, max {MAX_FILL_DISTANCE}m)")
    print(f"  Min speed:   {MIN_SPEED_MS} m/s")
    print(f"  Stretch:     {STRETCH_LOW_PCT}-{STRETCH_HIGH_PCT} percentile")
    print()

    # Ensure metadata exists
    print("Checking metadata files...")
    if not ensure_metadata_exists(RSD_FILE, META_DIR):
        print("\nFailed to generate or find metadata files. Cannot proceed.")
        exit(1)
    print()

    print("Loading metadata...")
    ss_port = pd.read_csv(os.path.join(META_DIR, "B002_ss_port_meta.csv"))
    ss_star = pd.read_csv(os.path.join(META_DIR, "B003_ss_star_meta.csv"))

    # Detect coordinate type and auto-determine UTM CRS
    coord_type = 'projected'
    output_crs = None
    transformer = None
    cols = ss_port.columns

    if 'lon' in cols and 'lat' in cols:
        coord_type = 'geographic'
        # Auto-detect UTM zone from data centroid
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
        # Process both sides
        wf_p, tex_p, nav_p, sign_p = process_side(f, ss_port, "Port", -1)
        wf_s, tex_s, nav_s, sign_s = process_side(f, ss_star, "Starboard", 1)

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
    save_merged_geotiff(
        wf_p, nav_p, wf_s, nav_s,
        OUTPUT_RESOLUTION,
        os.path.join(OUTPUT_DIR, "intensity.tif"),
        raster_crs=output_crs,
        transformer=transformer if coord_type == 'geographic' else None,
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
    print("\nOutputs:")
    print("  port_intensity.tif / star_intensity.tif - Individual sides")
    print("  intensity.tif - Merged backscatter mosaic")
    print("  texture.tif   - Local std deviation texture (high = rough/hard)")
