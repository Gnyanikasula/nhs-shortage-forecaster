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

# hadoop/ (not committed — binary, Windows-only, regenerate locally)

Required for PySpark to write files on Windows. Without this, Spark's
Parquet writes fail with `UnsatisfiedLinkError` at the commit step.

1. mkdir hadoop\bin
2. Download winutils.exe and hadoop.dll for Hadoop 3.3.6 (closest trusted
   match to PySpark 4.2.0's bundled Hadoop 3.5.0 -- confirmed compatible
   in practice) from:
   https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe
   https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/hadoop.dll
   into hadoop\bin\
3. Set environment variables (User scope, permanent):
   HADOOP_HOME = <repo path>\hadoop
   Add <repo path>\hadoop\bin to PATH
4. Fully restart your terminal/IDE (registry env var changes need a fresh process)