# Regenerating data/

None of the files in this folder are committed to git (see .gitignore).
To rebuild them from scratch:

1. Download `Price-concession-archive-0726.xlsx` from
   https://cpe.org.uk/funding-and-reimbursement/reimbursement/price-concessions/archive/
   ("Price concession archive spreadsheet" link) and place at
   `data/raw/concessions/archive.xlsx`

2. Run the original EPD pull for Nov 2020 - Feb 2025:
   `python src/ingestion/pull_epd_snomed.py --start 202011 --end 202502 --out epd_adhd.csv`

3. Fill the gap (Mar 2025 onward) via OpenPrescribing:
   `python src/ingestion/pull_epd_openprescribing.py --out epd_adhd_recent.csv`

4. Merge:
   `python src/ingestion/merge_epd_files.py`

5. Build the concession panel:
   `python src/features/build_panel.py`