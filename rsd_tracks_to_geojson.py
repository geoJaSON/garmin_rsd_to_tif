"""
Build a GeoJSON inventory of Garmin RSD survey tracks.

The script scans a folder for .RSD files, reuses or generates PINGVerter metadata,
extracts navigation points, converts them to WGS84 when needed, and writes one
GeoJSON LineString feature per file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# --- Edit these before running ---
INPUT_FOLDER = r""  
OUTPUT_PATH = None  # None -> <INPUT_FOLDER>/rsd_tracks.geojson, or set a full path string
SCAN_SUBFOLDERS = True  # False = only files directly in INPUT_FOLDER
FORCE_REGENERATE_METADATA = False
TRACK_ONLY_MODE = True  # True = skip full sonar metadata generation and extract GPS/nav only.
SKIP_TRACKS_ALREADY_WRITTEN = True  # Resume long runs by skipping RSDs already present in the output file.
MAX_TRACK_POINTS = 2000  # Set to None to keep every point; lower values write much faster.

REQUIRED_META_COLUMNS = {"index", "lon", "lat", "e", "n", "utm_zone"}


def get_output_paths(rsd_file: Path):
    rsd_basename = rsd_file.stem
    output_base_dir = rsd_file.parent / f"garmin_output_{rsd_basename}"
    meta_dir = output_base_dir / "meta"
    return output_base_dir, meta_dir


def ensure_metadata_exists(rsd_file: Path, meta_dir: Path, force_regenerate: bool = False):
    """
    Ensure metadata exists for the requested RSD.

    In track-only mode we skip the full pingverter export and build the
    minimal navigation CSV directly, since only the GPS centerline is needed.
    """
    all_meta = meta_dir / "All-Garmin-Sonar-MetaData.csv"
    port_meta = meta_dir / "B002_ss_port_meta.csv"
    star_meta = meta_dir / "B003_ss_star_meta.csv"

    if TRACK_ONLY_MODE:
        if not force_regenerate and all_meta.exists():
            return True
        print(f"Extracting navigation-only metadata for {rsd_file.name}...")
        return generate_navigation_metadata_fallback(rsd_file, meta_dir)

    if (
        not force_regenerate
        and (all_meta.exists() or (port_meta.exists() and star_meta.exists()))
    ):
        return True

    print(f"Generating metadata for {rsd_file.name}...")
    try:
        from pingverter import gar2pingmapper

        output_base_dir, _ = get_output_paths(rsd_file)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        gar2pingmapper(str(rsd_file), str(output_base_dir))
        return all_meta.exists() or port_meta.exists() or star_meta.exists()
    except KeyError as exc:
        if exc.args == ("sample_cnt",):
            print("  pingverter is missing sample_cnt for this file; generating navigation-only metadata instead...")
            return generate_navigation_metadata_fallback(rsd_file, meta_dir)
        print(f"  ERROR generating metadata: {exc}")
        return False
    except ImportError:
        print("  ERROR: pingverter library not found")
        return False
    except Exception as exc:
        print(f"  ERROR generating metadata: {exc}")
        return False


def generate_navigation_metadata_fallback(rsd_file: Path, meta_dir: Path):
    """
    Build a minimal metadata CSV with navigation only.

    This is enough for rough track extraction when pingverter cannot decode
    the full sonar payload metadata for a file. Sample counts are not needed
    because this script only builds a GPS centerline track.
    """
    try:
        from pingverter.garmin_class import gar
        from pingverter.verter_utils import filterGPS
    except ImportError:
        return False

    meta_dir.mkdir(parents=True, exist_ok=True)
    all_meta = meta_dir / "All-Garmin-Sonar-MetaData.csv"

    try:
        parser = gar(inFile=str(rsd_file), nchunk=0, exportUnknown=False)
        parser._getFileLen()
        parser._parseFileHeader()
        parser.son_struct, parser.son_header_struct, parser.record_body_header_len = parser._getPingHeaderStruct()

        rows = []
        offset = parser.headBytes

        with open(rsd_file, "rb") as handle:
            while offset < parser.file_len:
                row, next_offset = safe_get_ping_header(parser, handle, offset)
                if next_offset <= offset:
                    break
                offset = next_offset
                if row:
                    rows.append(row)

        if not rows:
            return False

        meta_df = pd.DataFrame.from_dict(rows)
        if "scposn_lat" not in meta_df.columns or "scposn_lon" not in meta_df.columns:
            return False

        meta_df["lat"] = meta_df["scposn_lat"].astype("float64") * 360.0 / (1 << 32)
        meta_df["lon"] = meta_df["scposn_lon"].astype("float64") * 360.0 / (1 << 32)
        meta_df["lat"] = np.where(meta_df["lat"] > 180.0, meta_df["lat"] - 360.0, meta_df["lat"])
        meta_df["lon"] = np.where(meta_df["lon"] > 180.0, meta_df["lon"] - 360.0, meta_df["lon"])

        valid_nav = (
            np.isfinite(meta_df["lon"])
            & np.isfinite(meta_df["lat"])
            & meta_df["lon"].between(-180.0, 180.0)
            & meta_df["lat"].between(-90.0, 90.0)
            & ((meta_df["lon"] != 0.0) | (meta_df["lat"] != 0.0))
        )
        meta_df = meta_df.loc[valid_nav].copy()
        if meta_df.empty:
            return False

        meta_df = filterGPS(meta_df)
        meta_df = meta_df.dropna(subset=["lon", "lat"]).copy()
        if meta_df.empty:
            return False

        epsg = get_utm_epsg(meta_df["lon"].iloc[0], meta_df["lat"].iloc[0])
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        easting, northing = transformer.transform(
            meta_df["lon"].to_numpy(dtype=float),
            meta_df["lat"].to_numpy(dtype=float),
        )
        meta_df["e"] = easting
        meta_df["n"] = northing
        meta_df["utm_zone"] = epsg % 100

        keep_cols = [col for col in ["index", "lon", "lat", "e", "n", "utm_zone"] if col in meta_df.columns]
        meta_df = meta_df[keep_cols]
        meta_df.to_csv(all_meta, index=False)
        return all_meta.exists()
    except Exception as exc:
        if all_meta.exists():
            all_meta.unlink(missing_ok=True)
        print(f"  ERROR extracting navigation-only metadata: {exc}")
        return False


def safe_get_ping_header(parser, file_handle, offset: int):
    """
    Read a Garmin ping header and extract only what the track fallback needs.
    """
    ping_body_header = {
        1: [("SP1_bh", "<u1"), ("channel_id", "<u1")],
        10: [("SP0a", "<u1"), ("bottom_depth", "V2")],
        11: [("SP0b", "<u1"), ("bottom_depth", "V3")],
        13: [("SP0d", "<u1"), ("unknown_sp0d", "V5")],
        18: [("SP12", "<u1"), ("drawn_bottom_depth", "V2")],
        19: [("SP13", "<u1"), ("drawn_bottom_depth", "V3")],
        21: [("SP15", "<u1"), ("unknown_sp15", "V5")],
        25: [("SP19", "<u1"), ("first_sample_depth", "<u1")],
        35: [("SP23", "<u1"), ("last_sample_depth", "V3")],
        41: [("SP29", "<u1"), ("gain", "<u1")],
        49: [("SP31", "<u1"), ("sample_status", "<u1")],
        60: [("SP3c", "<u1"), ("sample_cnt", "<u4")],
        65: [("SP41", "<u1"), ("shade_avail", "<u1")],
        76: [("SP4c", "<u1"), ("scposn_lat", "<u4")],
        84: [("SP54", "<u1"), ("scposn_lon", "<u4")],
        92: [("SP5c", "<u1"), ("water_temp", "<f4")],
        97: [("SP61", "<u1"), ("beam", "<u1")],
    }
    beam_info = {
        1: [("SP1_bi", "<u1"), ("port_star_beam_angle", "<u1")],
        9: [("SP9", "<u1"), ("fore_aft_beam_angle", "<u1")],
        17: [("SP11", "<u1"), ("port_star_elem_angle", "<u1")],
        25: [("SP19_bi", "<u1"), ("fore_aft_elem_angle", "<u1")],
        47: [
            ("SP2f", "<u1"),
            ("su2_len", "<u1"),
            ("su2_fcnt", "<u1"),
            ("su2_f0", "<u1"),
            ("port_star_id", "<f4"),
            ("su2_f1", "<u1"),
            ("su2_f1_unkown", "<f4"),
        ],
        55: [
            ("SP37", "<u1"),
            ("su3_len", "<u1"),
            ("su3_fcnt", "<u1"),
            ("su3_f0", "<u1"),
            ("su3_f0_unknown", "<u1"),
            ("su3_f1", "<u1"),
            ("su3_f1_unkown", "<f4"),
            ("su3_f2", "<u1"),
            ("su3_f2_unkown", "<f4"),
            ("su3_f3", "<u1"),
            ("su3_f3_unkown", "<f4"),
            ("su3_f4", "<u1"),
            ("su3_f4_unkown", "<f4"),
            ("su3_f5", "<u1"),
            ("su3_f5_unkown", "<f4"),
            ("su3_f6", "<u1"),
            ("su3_f6_unkown", "<f4"),
        ],
        115: [("SP73", "<u1"), ("interrogation_id", "<u2"), ("son_byte_len", "<u1")],
    }

    file_handle.seek(offset)
    header_buffer = file_handle.read(parser.pingHeaderLen)
    if len(header_buffer) < parser.pingHeaderLen:
        return None, parser.file_len

    header = np.frombuffer(header_buffer, dtype=np.dtype(parser.son_header_struct))
    out_dict = {name: header[name][0].item() for name in header.dtype.fields}

    if out_dict.get("state") != 2:
        return None, offset + parser.pingHeaderLenFirst

    record_body_count = parser._fread_dat(file_handle, 1, "B")[0]
    out_dict["record_body_fcnt"] = record_body_count

    field_count = min(record_body_count, 13)
    has_beam_info = record_body_count > 13

    parsed_fields = 0
    while parsed_fields < field_count:
        field_id = parser._fread_dat(file_handle, 1, "B")[0]
        field_struct = ping_body_header.get(field_id)
        if field_struct is None:
            continue

        out_dict[field_struct[0][0]] = field_id
        dtype = np.dtype(field_struct[1:])
        field_buffer = file_handle.read(dtype.itemsize)
        if len(field_buffer) < dtype.itemsize:
            return None, parser.file_len

        field_values = np.frombuffer(field_buffer, dtype=dtype)
        for name in field_values.dtype.fields:
            out_dict[name] = field_values[name][0].item()
        parsed_fields += 1

    if has_beam_info:
        parser._fread_dat(file_handle, 1, "B")
        parser._fread_dat(file_handle, 1, "B")
        beam_field_count = parser._fread_dat(file_handle, 1, "B")[0]

        parsed_beam_fields = 0
        while parsed_beam_fields < beam_field_count:
            field_id = parser._fread_dat(file_handle, 1, "B")[0]
            field_struct = beam_info.get(field_id)
            if field_struct is None:
                continue

            out_dict[field_struct[0][0]] = field_id
            dtype = np.dtype(field_struct[1:])
            field_buffer = file_handle.read(dtype.itemsize)
            if len(field_buffer) < dtype.itemsize:
                return None, parser.file_len

            field_values = np.frombuffer(field_buffer, dtype=dtype)
            for name in field_values.dtype.fields:
                out_dict[name] = field_values[name][0].item()
            parsed_beam_fields += 1

    out_dict["index"] = offset
    data_size = int(out_dict.get("data_size", 0))
    next_ping = offset + parser.pingHeaderLen + data_size + 12
    return out_dict, next_ping


def get_utm_epsg(lon: float, lat: float):
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return 32600 + zone if lat >= 0 else 32700 + zone


def load_best_metadata(meta_dir: Path):
    """
    Prefer the global metadata file when available; otherwise combine port/star rows.
    """
    all_meta = meta_dir / "All-Garmin-Sonar-MetaData.csv"
    port_meta = meta_dir / "B002_ss_port_meta.csv"
    star_meta = meta_dir / "B003_ss_star_meta.csv"

    if all_meta.exists():
        return read_metadata_csv(all_meta), all_meta.name

    frames = []
    sources = []
    if port_meta.exists():
        frames.append(read_metadata_csv(port_meta))
        sources.append(port_meta.name)
    if star_meta.exists():
        frames.append(read_metadata_csv(star_meta))
        sources.append(star_meta.name)

    if not frames:
        return None, None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged, ",".join(sources)


def read_metadata_csv(csv_path: Path):
    """
    Only load the columns needed to build track lines.
    """
    try:
        return pd.read_csv(csv_path, usecols=lambda col: col in REQUIRED_META_COLUMNS)
    except ValueError:
        return pd.read_csv(csv_path)


@lru_cache(maxsize=60)
def build_transformer_from_zone(zone_value):
    try:
        zone = int(float(zone_value))
    except (TypeError, ValueError):
        return None
    if not (1 <= zone <= 60):
        return None
    return Transformer.from_crs(f"EPSG:{32600 + zone}", "EPSG:4326", always_xy=True)


def thin_track_coordinates(coords, max_points: int | None):
    if max_points is None or max_points < 2 or len(coords) <= max_points:
        return coords

    sample_idx = np.linspace(0, len(coords) - 1, num=max_points, dtype=int)
    sample_idx = np.unique(sample_idx)
    return [coords[idx] for idx in sample_idx]


def extract_track_coordinates(meta_df: pd.DataFrame):
    """
    Return a de-duplicated list of [lon, lat] coordinates in WGS84.
    """
    if meta_df is None or meta_df.empty:
        return []

    sort_col = "index" if "index" in meta_df.columns else None
    if sort_col is not None:
        order_vals = pd.to_numeric(meta_df[sort_col], errors="coerce")
        meta_df = meta_df.assign(_sort_key=order_vals).sort_values("_sort_key", kind="stable")

    has_lon_lat = "lon" in meta_df.columns and "lat" in meta_df.columns
    has_projected = "e" in meta_df.columns and "n" in meta_df.columns
    row_count = len(meta_df)
    lon_vals = np.full(row_count, np.nan, dtype=float)
    lat_vals = np.full(row_count, np.nan, dtype=float)

    if has_lon_lat:
        lon_series = pd.to_numeric(meta_df["lon"], errors="coerce").to_numpy(dtype=float)
        lat_series = pd.to_numeric(meta_df["lat"], errors="coerce").to_numpy(dtype=float)
        valid_lon_lat = np.isfinite(lon_series) & np.isfinite(lat_series)
        lon_vals[valid_lon_lat] = lon_series[valid_lon_lat]
        lat_vals[valid_lon_lat] = lat_series[valid_lon_lat]

    transformer = None
    if has_projected and "utm_zone" in meta_df.columns:
        valid_zones = pd.to_numeric(meta_df["utm_zone"], errors="coerce").dropna()
        if not valid_zones.empty:
            transformer = build_transformer_from_zone(valid_zones.iloc[0])

    if has_projected and transformer is not None:
        easting = pd.to_numeric(meta_df["e"], errors="coerce").to_numpy(dtype=float)
        northing = pd.to_numeric(meta_df["n"], errors="coerce").to_numpy(dtype=float)
        needs_projected = ~(np.isfinite(lon_vals) & np.isfinite(lat_vals))
        valid_projected = needs_projected & np.isfinite(easting) & np.isfinite(northing)
        if np.any(valid_projected):
            proj_lon, proj_lat = transformer.transform(
                easting[valid_projected],
                northing[valid_projected],
            )
            lon_vals[valid_projected] = proj_lon
            lat_vals[valid_projected] = proj_lat

    valid_points = np.isfinite(lon_vals) & np.isfinite(lat_vals)
    if not np.any(valid_points):
        return []

    lon_vals = np.round(lon_vals[valid_points], 8)
    lat_vals = np.round(lat_vals[valid_points], 8)

    keep_mask = np.ones(lon_vals.shape[0], dtype=bool)
    if lon_vals.shape[0] > 1:
        keep_mask[1:] = (lon_vals[1:] != lon_vals[:-1]) | (lat_vals[1:] != lat_vals[:-1])

    coords = np.column_stack((lon_vals[keep_mask], lat_vals[keep_mask])).tolist()

    return thin_track_coordinates(coords, MAX_TRACK_POINTS)


def build_feature(rsd_file: Path, coords, source_meta_name: str):
    return {
        "type": "Feature",
        "properties": {
            "file_name": rsd_file.name,
            "file_path": str(rsd_file),
            "track_points": len(coords),
            "metadata_source": source_meta_name,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }


def normalize_rsd_path(path_value):
    return str(Path(path_value).expanduser().resolve()).lower()


def load_existing_features(output_path: Path):
    if not output_path.exists():
        return [], set()

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Existing output is not valid JSON: {output_path} ({exc})") from exc

    if payload.get("type") != "FeatureCollection":
        raise SystemExit(f"Existing output is not a GeoJSON FeatureCollection: {output_path}")

    seen_paths = set()
    features = []

    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        file_path = properties.get("file_path")
        if not file_path:
            continue
        path_key = normalize_rsd_path(file_path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        features.append(feature)

    return features, seen_paths


def write_feature_collection(output_path: Path, features):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    temp_path.replace(output_path)


def find_rsd_files(input_folder: Path, recursive: bool):
    if recursive:
        iterator = input_folder.rglob("*")
    else:
        iterator = input_folder.iterdir()

    rsd_files = [path for path in iterator if path.is_file() and path.suffix.lower() == ".rsd"]
    return sorted(rsd_files)


def main():
    if not INPUT_FOLDER or not str(INPUT_FOLDER).strip():
        raise SystemExit("Set INPUT_FOLDER at the top of rsd_tracks_to_geojson.py to your RSD folder path.")
    input_folder = Path(INPUT_FOLDER).expanduser().resolve()
    output_path = (
        Path(OUTPUT_PATH).expanduser().resolve()
        if OUTPUT_PATH
        else input_folder / "rsd_tracks.geojson"
    )

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    rsd_files = find_rsd_files(input_folder, recursive=SCAN_SUBFOLDERS)
    if not rsd_files:
        raise SystemExit(f"No .RSD files found under {input_folder}")

    print(f"Found {len(rsd_files)} RSD file(s)")

    features, completed_paths = load_existing_features(output_path)
    if features:
        print(f"Loaded {len(features)} existing track(s) from {output_path}")

    skipped = []
    skipped_existing = 0

    for rsd_file in rsd_files:
        path_key = normalize_rsd_path(rsd_file)
        if SKIP_TRACKS_ALREADY_WRITTEN and path_key in completed_paths:
            print(f"Skipping {rsd_file} (track already written)")
            skipped_existing += 1
            continue

        print(f"Processing {rsd_file}...")
        _, meta_dir = get_output_paths(rsd_file)

        if not ensure_metadata_exists(
            rsd_file,
            meta_dir,
            force_regenerate=FORCE_REGENERATE_METADATA,
        ):
            skipped.append((rsd_file, "metadata unavailable"))
            continue

        meta_df, source_meta_name = load_best_metadata(meta_dir)
        if meta_df is None:
            skipped.append((rsd_file, "metadata CSV not found"))
            continue

        coords = extract_track_coordinates(meta_df)
        if len(coords) < 2:
            skipped.append((rsd_file, "not enough navigation points"))
            continue

        features.append(build_feature(rsd_file, coords, source_meta_name))
        completed_paths.add(path_key)
        write_feature_collection(output_path, features)
        print(f"  Saved track {len(features)} to {output_path}")

    print(f"Wrote {len(features)} track(s) to {output_path}")
    if skipped_existing:
        print(f"Skipped {skipped_existing} file(s) already present in the output")
    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for rsd_file, reason in skipped:
            print(f"  - {rsd_file.name}: {reason}")


if __name__ == "__main__":
    main()
