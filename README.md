# DAIseg-m
Highly accurate method for detecting archaic segments in the modern admixed genomes 


![Demography](https://github.com/Genomics-HSE/DAIseg.mexicans/blob/main/Mex.svg)

```bash
echo "  Step 1: restrict_1kG"
python daiseg.py restrict_1kG -json $json -threads 16 

echo "  Step 2: callability"
python daiseg.py callability -json $json -threads 16  

echo "  Step 3: main.prep" 
python daiseg.py main.prep -json $json -threads 16 

```



The simplest command to run DAIseg-m
```bash
nohup python3 daiseg.py run.with.EM  -json $json > daiseg.log 2>&1 &
```
where .json is configuration file. 



<details>
<summary>Example configuration .json: DAIseg-m (GRCh38, chr1)</summary>

```json
{
  "description": "DAIseg.mexicans configuration to run",
  "CHROM": "chr1",
  "output": "MXL.grch38.chr1",
  "prefix": "/path/to/output/MXL.grch38.v2",
  "files": {
    "neand_files": {
      "Vindija33.19": {
        "bed": "/path/to/data/neand/33.19.grch38/bed/chr1_mask.bed.gz",
        "vcf": "/path/to/data/neand/33.19.grch38/chr1.vcf.gz"
      },
      "Altai": {
        "bed": "/path/to/data/neand/altai.grch38/bed/chr1_mask.bed.gz",
        "vcf": "/path/to/data/neand/altai.grch38/chr1.vcf.gz"
      },
      "Chagyrskaya-Phalanx": {
        "bed": "/path/to/data/neand/Chagyrskaya.grch38/bed/chr1_mask.bed.gz",
        "vcf": "/path/to/data/neand/Chagyrskaya.grch38/chr1.vcf.gz"
      }
    },
    "1000GP_files": {
      "bed": "/path/to/data/1000GP/1000GP.grch38/bed/chr1.clean.bed",
      "vcf": "1kG_filtered.chr1.grch38.bcf",
      "vcf_initial": "/path/to/data/1000GP/1000GP.grch38/CCDG_14151_B01_GRM_WGS_2020-08-05_chr1.filtered.shapeit2-duohmm-phased.vcf.gz"
    },
    "ancestral": {
      "fasta": "/path/to/data/Anc.fa/homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_1.fa"
    },
    "reference": {
      "fasta": "/path/to/data/ref.fa/grch38.fasta/GRCh38_full_analysis_set_plus_decoy_hla.fa"
    },
    "chr_lengths": "/path/to/data/ref.fa/grch38.lengths/hg38.chrom.sizes"
  },
  "samples": {
    "Africans": [
      "NA18486", "NA18488", "NA18489", "NA18498", "NA18499", "NA18501",
      "NA18502", "NA18504", "NA18505", "NA18507", "NA18508", "NA18510",
      "NA18511", "NA18516", "NA18517", "NA18519", "NA18520", "NA18522",
      "NA18523", "NA18853", "NA18856", "NA18858", "NA18861", "NA18864",
      "NA18865", "NA18867", "NA18868", "NA18870", "NA18871", "NA18873",
      "NA18874", "NA18876", "NA18877", "NA18878", "NA18879", "NA18881",
      "NA18907", "NA18908", "NA18909", "NA18910", "NA18912", "NA18915",
      "NA18916", "NA18917", "NA18923", "NA18924", "NA18933", "NA18934",
      "NA19092", "NA19093", "NA19095", "NA19096", "NA19098", "NA19099",
      "NA19102", "NA19107", "NA19108", "NA19113", "NA19114", "NA19116",
      "NA19117", "NA19118", "NA19119", "NA19121", "NA19129", "NA19130",
      "NA19131", "NA19137", "NA19138", "NA19141", "NA19143", "NA19144",
      "NA19146", "NA19147", "NA19149", "NA19152", "NA19153", "NA19159",
      "NA19160", "NA19171", "NA19172", "NA19175", "NA19184", "NA19185",
      "NA19189", "NA19190", "NA19197", "NA19198", "NA19200", "NA19201",
      "NA19204", "NA19206", "NA19207", "NA19209", "NA19210", "NA19213",
      "NA19214", "NA19222", "NA19223", "NA19225", "NA19235", "NA19236",
      "NA19238", "NA19239", "NA19247", "NA19248", "NA19256", "NA19257"
    ],
    "Americans": [
      "HG01565", "HG01566", "HG01571", "HG01572", "HG01577", "HG01578",
      "HG01892", "HG01893", "HG01917", "HG01918", "HG01920", "HG01921",
      "HG01923", "HG01924", "HG01926", "HG01927", "HG01932", "HG01933",
      "HG01935", "HG01936", "HG01938", "HG01939", "HG01941", "HG01942",
      "HG01944", "HG01945", "HG01947", "HG01948", "HG01950", "HG01951",
      "HG01953", "HG01954", "HG01961", "HG01965", "HG01967", "HG01968",
      "HG01970", "HG01971", "HG01973", "HG01974", "HG01976", "HG01977",
      "HG01979", "HG01980", "HG01982", "HG01991", "HG01992", "HG01997",
      "HG02002", "HG02003", "HG02006", "HG02008", "HG02089", "HG02090",
      "HG02102", "HG02104", "HG02105", "HG02146", "HG02147", "HG02150",
      "HG02252", "HG02253", "HG02259", "HG02260", "HG02262", "HG02265",
      "HG02266", "HG02271", "HG02272", "HG02274", "HG02275", "HG02277",
      "HG02278", "HG02285", "HG02286", "HG02291", "HG02292", "HG02298",
      "HG02299", "HG02301", "HG02304", "HG02312", "HG02345", "HG02348",
      "HG02425"
    ],
    "Europeans": [
      "HG01500", "HG01501", "HG01503", "HG01504", "HG01506", "HG01507",
      "HG01509", "HG01510", "HG01512", "HG01513", "HG01515", "HG01516",
      "HG01518", "HG01519", "HG01521", "HG01522", "HG01524", "HG01525",
      "HG01527", "HG01528", "HG01530", "HG01531", "HG01536", "HG01537",
      "HG01602", "HG01603", "HG01605", "HG01606", "HG01607", "HG01608",
      "HG01610", "HG01612", "HG01613", "HG01615", "HG01617", "HG01618",
      "HG01619", "HG01620", "HG01623", "HG01624", "HG01625", "HG01626",
      "HG01628", "HG01630", "HG01631", "HG01632", "HG01668", "HG01669",
      "HG01670", "HG01672", "HG01673", "HG01675", "HG01676", "HG01678",
      "HG01679", "HG01680", "HG01682", "HG01684", "HG01685", "HG01686",
      "HG01694", "HG01695", "HG01697", "HG01699", "HG01700", "HG01702",
      "HG01704", "HG01705", "HG01707", "HG01708", "HG01709", "HG01710",
      "HG01746", "HG01747", "HG01756", "HG01757", "HG01761", "HG01762",
      "HG01765", "HG01766", "HG01767", "HG01768", "HG01770", "HG01771",
      "HG01773", "HG01775", "HG01776", "HG01777", "HG01779", "HG01781",
      "HG01783", "HG01784", "HG01785", "HG01786", "HG02219", "HG02220",
      "HG02221", "HG02223", "HG02224", "HG02230", "HG02231", "HG02232",
      "HG02233", "HG02235", "HG02236", "HG02238", "HG02239"
    ],
    "Mexicans": [
      "NA19648", "NA19649", "NA19651", "NA19652", "NA19654", "NA19655",
      "NA19657", "NA19658", "NA19661", "NA19663", "NA19664", "NA19669",
      "NA19670", "NA19676", "NA19678", "NA19679", "NA19681", "NA19682",
      "NA19684", "NA19716", "NA19717", "NA19719", "NA19720", "NA19722",
      "NA19723", "NA19725", "NA19726", "NA19728", "NA19729", "NA19731",
      "NA19732", "NA19734", "NA19735", "NA19740", "NA19741", "NA19746",
      "NA19747", "NA19749", "NA19750", "NA19752", "NA19755", "NA19756",
      "NA19758", "NA19759", "NA19761", "NA19762", "NA19764", "NA19770",
      "NA19771", "NA19773", "NA19774", "NA19776", "NA19777", "NA19779",
      "NA19780", "NA19782", "NA19783", "NA19785", "NA19786", "NA19788",
      "NA19789", "NA19792", "NA19794", "NA19795"
    ],
    "neand": [
      "Vindija33.19",
      "Altai",
      "Chagyrskaya-Phalanx"
    ]
  },
  "parameters_initial": {
    "admixture_nd": 0.02,
    "admixture_modern": [0.5, 0.4, 0.1],
    "introgression_time": 55000,
    "rr": 1e-08,
    "mutation": 1.25e-08,
    "window_length": 1000,
    "generation_time": 29,
    "t_n_c": 550000,
    "t_af_c": 70000,
    "t_introgression_c": 55000,
    "t_ea_c": 41000,
    "t_mexicans_c": 500,
    "t_introgression": 55000,
    "t_mexicans": 500
  },
  "window_callability": {
    "Thousand_genomes": "coverage_1kG.chr1.grch38.bed",
    "Nd_1k_genomes": "coverage_1kG.nd.chr1.grch38.bed"
  },
  "data": "prep.chr1.grch38.tsv",
  "gaps": "/path/to/data/ref.fa/gaps.grch38/gap.txt"
}
```



</details>
