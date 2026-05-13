import csv
import sys


def parse_set_fast(val_str):
    """
    Parses TSV string like '{A,G}' or '{T}' into a Python set.
    Returns None for missing data: '.', '{}', or empty string.
    """
    if not val_str or val_str == '{}' or val_str == '.':
        return None  
    return set(val_str.strip('{}').split(','))

def process_data(tsv_path, bed_path):
    """
    Reads BED (windows) and TSV (genotypes).
    Calculates differences from EU, NA, AF, and ND.
    """
    print(" Loading BED windows...")

    windows_by_chrom = {}
    all_windows_flat = []

    # Load BED
    try:
        with open(bed_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                if parts[0].startswith('#') or parts[0].lower() == 'chrom':
                    continue

                chrom = parts[0]
                if not chrom.startswith('chr'):
                    chrom = 'chr' + chrom

                start = int(parts[1])
                end = int(parts[2])

                window = {
                    's': start,
                    'e': end,
                    'stats': None
                }
                all_windows_flat.append(window)

                if chrom not in windows_by_chrom:
                    windows_by_chrom[chrom] = []
                windows_by_chrom[chrom].append(window)
    except FileNotFoundError:
        raise FileNotFoundError(f"BED file not found: {bed_path}")

    print(f" Loaded {len(all_windows_flat)} windows. Processing TSV...")

    # Process TSV file
    try:
        with open(tsv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')

            # Read Header
            header = next(reader, None)
            if not header:
                raise ValueError("TSV file is empty")

            header_map = {name: i for i, name in enumerate(header)}

            try:
                # Identify chromosome column
                if '#CHROM' in header_map:
                    idx_c = header_map['#CHROM']
                elif 'chr' in header_map:
                    idx_c = header_map['chr']
                else:
                    idx_c = header_map['CHROM']

                idx_p = header_map['POS']
                idx_anc = header_map['Ancestral']

                # NEW COLUMNS
                idx_eu = header_map['EU']
                idx_na = header_map['NA']
                idx_af = header_map['AF']
                idx_nd = header_map['ND']

            except KeyError as e:
                raise ValueError(f"Missing mandatory column in TSV: {e}")

            # Identify Haplotype Columns
            exclude_names = {
                '#CHROM', 'chr', 'CHROM', 'POS', 'REF', 'ALT', 'Ancestral', 
                'EU', 'NA', 'AF', 'ND'  # <--- Added new columns to exclude list
            }

            hap_indices = []
            for i, h in enumerate(header):
                if h not in exclude_names:
                    hap_indices.append((h, i))

            hap_names = [x[0] for x in hap_indices]

            # Initialize stats with 4 counters: [EU, NA, AF, ND]
            for w in all_windows_flat:
                w['stats'] = {name: [0, 0, 0, 0] for name in hap_names}

            # Linear Scan Variables
            current_chrom = None
            current_windows_list = []
            win_idx = 0
            max_win_idx = 0

            for row_num, row in enumerate(reader):
                if row_num % 100000 == 0 and row_num > 0:
                    print(f"   📊 Processed lines: {row_num}...", end='\r')

                try:
                    raw_chrom = row[idx_c]
                    if not raw_chrom.startswith('chr'):
                        chrom = 'chr' + raw_chrom
                    else:
                        chrom = raw_chrom
                    pos = int(row[idx_p]) - 1
                except (ValueError, IndexError):
                    continue

                # Handle Chromosome Switch
                if chrom != current_chrom:
                    current_chrom = chrom
                    current_windows_list = windows_by_chrom.get(chrom, [])
                    win_idx = 0
                    max_win_idx = len(current_windows_list)

                if win_idx >= max_win_idx:
                    continue

                # Advance window index
                curr_win = current_windows_list[win_idx]
                while pos >= curr_win['e']:
                    win_idx += 1
                    if win_idx >= max_win_idx:
                        break
                    curr_win = current_windows_list[win_idx]

                if win_idx >= max_win_idx:
                    continue

                if pos < curr_win['s']:
                    continue

                # CALCULATE STATS
                anc = row[idx_anc]

                # Skip if ancestral is unknown or lowercase
                if not anc or not anc.isupper():
                    continue

                # Parse reference sets for this SNP
                eu_set = parse_set_fast(row[idx_eu])
                na_set = parse_set_fast(row[idx_na])
                af_set = parse_set_fast(row[idx_af])
                nd_set = parse_set_fast(row[idx_nd])

                for hap_name, hap_idx in hap_indices:
                    val = row[hap_idx]

                    if val == '.' or val == anc:
                        continue
                    if val.upper() == anc:
                        continue

                    # Check differences
                    is_diff_eu = (eu_set is not None) and (val not in eu_set)
                    is_diff_na = (na_set is not None) and (val not in na_set)
                    is_diff_af = (af_set is not None) and (val not in af_set)
                    is_diff_nd = (nd_set is not None) and (val not in nd_set)

                    if any([is_diff_eu, is_diff_na, is_diff_af, is_diff_nd]):
                        stats = curr_win['stats'][hap_name]
                        # 0: EU, 1: NA, 2: AF, 3: ND
                        if is_diff_eu: stats[0] += 1
                        if is_diff_na: stats[1] += 1
                        if is_diff_af: stats[2] += 1
                        if is_diff_nd: stats[3] += 1

    except FileNotFoundError:
        raise FileNotFoundError(f" Data file not found: {tsv_path}")

    print("\n [obs.py] Processing done. Aggregating results...")

    final_result = {name: [] for name in hap_names}
    for w in all_windows_flat:
        for name in hap_names:
            final_result[name].append(w['stats'][name])

    return final_result


def get_number_states(result_dict):
    """
    Finds maximum number of differences across all windows and all 4 metrics.
    """
    max_val = -1
    max_info = {}
    labels = ['EU', 'NA', 'AF', 'ND']

    for hap, windows in result_dict.items():
        for i, stats in enumerate(windows):

            for j in range(4):
                if stats[j] > max_val:
                    max_val = stats[j]
                    max_info = {
                        'hap': hap,
                        'win_idx': i,
                        'type': labels[j],
                        'pair': stats
                    }
    return max_val, max_info
