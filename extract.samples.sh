#!/bin/bash

# ./extract.samples.sh json thrds Extract specific samples from 1000 Genomes VCF file

set -e

JSON=$1
nthr=$2

start_time=$(date +%s)
PREF="$(jq -r '.prefix' "${JSON}")"

# Get path to original 1000GP VCF
FILE_1kG=$(jq -r --arg h "$HOME" '.files["1000GP_files"].vcf_initial | sub("^~"; $h)' "${JSON}")

# CHECK 1: Verify input file exists ---
if [[ ! -f "$FILE_1kG" ]]; then
    echo "   CRITICAL ERROR: Input VCF file does not exist!"
    echo "   Path checked: $FILE_1kG"
    exit 1
fi

# Normalize chr name
CHROM_RAW=$(jq -r '.CHROM' "$JSON")
CHROM=${CHROM_RAW#chr}
bed_strict="$(jq -r '.files["1000GP_files"].bed' "${JSON}")"


BASE_VCF_NAME="$(jq -r '.files["1000GP_files"].vcf' "${JSON}")"
OUTPUT_NAME=$(echo "$BASE_VCF_NAME" | sed "s/CHROMOSOME/chr${CHROM}/")
FILTERED_1kG="$PREF/$OUTPUT_NAME"


mkdir -p "$PREF"


# Temporary sample list
SAMPLE_LIST=$(mktemp)
jq -r '.samples.Mexicans[], .samples.Africans[], .samples.Americans[], .samples.Europeans[]' "$JSON" > "$SAMPLE_LIST"
SAMPLE_COUNT=$(wc -l < "$SAMPLE_LIST")

echo " Extracting $SAMPLE_COUNT samples..."
if [[ "$SAMPLE_COUNT" -eq 0 ]]; then
    echo " Error: Sample list is empty. Check JSON configuration."
    rm "$SAMPLE_LIST"
    exit 1
fi


echo "  Running bcftools for chr ..."  $CHROM_RAW
bcftools view --threads $nthr -S "$SAMPLE_LIST" --force-samples --trim-alt-alleles -T ${bed_strict}   ${FILE_1kG} -Ou  | \
  bcftools norm --threads $nthr -m -any -Ou | \
  bcftools view --threads $nthr -v snps -Ou | \
  bcftools norm --threads $nthr -m +any -Ou | \
  bcftools view --threads $nthr -m2 -M4 -Ob -o $FILTERED_1kG


echo " Indexing VCF..."
bcftools index -f --threads "$nthr" "$FILTERED_1kG"

# Temp files
rm "$SAMPLE_LIST"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo " Filtered VCF saved to: $FILTERED_1kG"
echo "  Total time: ${duration} seconds"
