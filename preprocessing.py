import json
import os
import sys

def expand_path(path):

    if not path:
        return None
    return os.path.expanduser(path)

def load_config(json_path):
    """Load .json"""
    if not os.path.exists(json_path):
        sys.stderr.write(f"Error: File '{json_path}' not found.\n")
        sys.exit(1)
    with open(json_path, 'r') as f:
        return json.load(f)


def map_columns(header_list, samples_cfg):
    """
    Map VCF header columns to sample groups.
    """
    try:
        idx_chrom = header_list.index("CHROM")
        idx_pos = header_list.index("POS")
        idx_ref = header_list.index("REF")
        idx_alt = header_list.index("ALT")
    except ValueError as e:
        raise ValueError(f"Missing standard columns in header: {e}")

    # Create a lookup map
    header_map = {name.split(':')[0]: i for i, name in enumerate(header_list)}

    eu_names_list = set(samples_cfg.get("Europeans", []))
    na_names_list = set(samples_cfg.get("Americans", []))
    af_names_list = set(samples_cfg.get("Africans", []))
    mxl_names_list = set(samples_cfg.get("Mexicans", []))

    # Маппинг 
    col_map_eu = [header_map[name] for name in eu_names_list if name in header_map]
    col_map_na = [header_map[name] for name in na_names_list if name in header_map]
    col_map_af = [header_map[name] for name in af_names_list if name in header_map]
    col_map_mxl = [header_map[name] for name in mxl_names_list if name in header_map]

    col_map_nd = []

    all_names = eu_names_list | na_names_list | af_names_list | mxl_names_list
    for name in all_names - set(header_map.keys()):
        sys.stderr.write(f"!!! Sample {name} not found in VCF.\n")

    start_idx = idx_alt + 1
    for i in range(start_idx, len(header_list)):
        raw_name = header_list[i]
        name = raw_name.split(':')[0]

        if (name not in eu_names_list and
            name not in na_names_list and
            name not in af_names_list and
            name not in mxl_names_list):
            col_map_nd.append(i)
    
    return (idx_chrom, idx_pos, idx_ref, idx_alt, 
            col_map_nd, col_map_eu, col_map_na, col_map_af, col_map_mxl)
