"""
main.py
Entry point for the CMPS 340 example project.
"""

import argparse
import logging
import sys
import json
import config
from module_tmp import CSVHandler, PickleHandler
import pandas as pd

def setup_logging():
    logging.basicConfig(filename=config.LOG_FILE,
                        level=getattr(logging, config.LOG_LEVEL),
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w',
                        force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Logging initialized.")

def parse_args():
    parser = argparse.ArgumentParser(description="CS340 Example Project - Data Insights Analyzer")
    parser.add_argument('--wnumber', type=str, help="W_Number of a student to inspect (e.g., w07750000)")
    parser.add_argument('--plot-kind', type=str, choices=['bar','line'], default='bar', help="Kind of plot for averages")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    logging.info("Starting Data Insights Analyzer")

    # Pkl to CSV
    df = pd.read_pickle('./Input/data.pkl')
    df.to_csv('./Output/results/pickle_to_csv_view.csv', index=False)

    # Initialize handlers
    csv_handler = CSVHandler(config.CSV_FILE)
    pkl_handler = PickleHandler(config.PKL_FILE)

    # Visualizations
    csv_handler.visualize_pass_fail()
    csv_handler.plot_average_overall(kind=args.plot_kind)
    csv_handler.export_class_stats()

    # Study data processing
    pkl_handler.export_study_averages()

    # If a W_Number is provided, show individual info
    if args.wnumber:
        student = csv_handler.get_student(args.wnumber)
        if not student.empty:
            logging.info("Student info:")
            logging.info(student.to_dict())
            csv_handler.plot_individual(args.wnumber)
            study_vec = pkl_handler.student_study_vector(args.wnumber)
            if study_vec:
                logging.info(f"Study hours (week1..4): {study_vec}")
        else:
            logging.warning(f"No student with W_Number {args.wnumber}")

    # Print summary stats
    stats = csv_handler.class_statistics()
    logging.info("Class statistics summary:")
    logging.info(json.dumps(stats, indent=2))
    logging.info("Program completed successfully.")

if __name__ == '__main__':
    main()
