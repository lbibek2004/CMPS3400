"""
module_tmp.py
Core classes for the CMPS 340 example project.
"""

import os
import math
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

sns.set_style(config.SEABORN_STYLE)

# -------------------------------------------------------------
# Parent: DataHandler - common read / write utilities
# -------------------------------------------------------------
class DataHandler:
    """
    Parent class for reading and writing data.
    Purpose:
        - Provide utilities for reading CSV and Pickle files
        - Basic validation and path handling
    INPUT:
        - None
    OUTPUT:
        - N/A
    """

    def __init__(self):
        self.config = config

    def ensure_output_dirs(self):
        os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.config.PLOTS_PATH, exist_ok=True)
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)

    def read_csv(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logging.error(f"Failed to read CSV at {path}: {e}")
            return pd.DataFrame()

    def read_pickle(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_pickle(path)
            return df
        except Exception as e:
            logging.error(f"Failed to read pickle at {path}: {e}")
            return pd.DataFrame()


# -------------------------------------------------------------
# Child: CSVHandler - handles student CSV
# -------------------------------------------------------------
class CSVHandler(DataHandler):
    """
    Child class that reads student CSV and provides analysis/visuals.
    """

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self.ensure_output_dirs()
        self.df = self.read_csv(csv_path)
        # compute average score column for convenience
        if not self.df.empty:
            self.df['Average'] = self.df[['Exam_1','Exam_2','Exam_3']].mean(axis=1)
            self.df['PassFail'] = self.df['Average'] >= 60.0

    def visualize_pass_fail(self, save: bool = True):
        """Show distribution of pass/fail counts."""
        counts = self.df['PassFail'].value_counts().rename(index={True:'Pass', False:'Fail'})
        fig, ax = plt.subplots(figsize=self.config.FIGSIZE)
        counts.plot(kind='bar', ax=ax)
        ax.set_title('Pass vs Fail Distribution')
        ax.set_ylabel('Number of Students')
        if save:
            out = os.path.join(self.config.PLOTS_PATH, 'pass_fail_distribution.png')
            fig.savefig(out, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Saved pass/fail distribution to {out}")
        else:
            plt.show()

    def plot_average_overall(self, kind: str = 'bar', save: bool = True):
        """Plot average marks for all students as bar or line chart."""
        avg = self.df[['W_Number','Average']].set_index('W_Number').sort_index()
        fig, ax = plt.subplots(figsize=self.config.FIGSIZE)
        if kind == 'line':
            avg.plot(kind='line', ax=ax, legend=False)
            ax.set_ylabel('Average Score')
            ax.set_title('Average Score per Student (line)')
        else:
            avg.plot(kind='bar', ax=ax, legend=False)
            ax.set_ylabel('Average Score')
            ax.set_title('Average Score per Student (bar)')
        if save:
            out = os.path.join(self.config.PLOTS_PATH, f'average_all_{kind}.png')
            fig.savefig(out, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Saved average_all_{kind} to {out}")
        else:
            plt.show()

    def plot_individual(self, w_number: str, save: bool = True):
        """Plot exam scores for an individual student as a bar chart."""
        row = self.df[self.df['W_Number'] == w_number]
        if row.empty:
            logging.warning(f"No student found with W_Number={w_number}")
            return None
        exams = row[['Exam_1','Exam_2','Exam_3']].iloc[0]
        fig, ax = plt.subplots(figsize=self.config.FIGSIZE)
        exams.plot(kind='bar', ax=ax)
        ax.set_ylim(0, 100)
        ax.set_title(f'Exam Scores for {row.iloc[0]["First_Name"]} {row.iloc[0]["Last_Name"]} ({w_number})')
        if save:
            out = os.path.join(self.config.PLOTS_PATH, f'{w_number}_exams.png')
            fig.savefig(out, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Saved individual exam plot for {w_number} to {out}")
        else:
            plt.show()
        return exams.to_dict()

    def get_student(self, w_number: str) -> pd.Series:
        """Return row for student with W_Number."""
        row = self.df[self.df['W_Number'] == w_number]
        if row.empty:
            logging.warning(f"No student found with W_Number={w_number}")
            return pd.Series()
        return row.iloc[0]

    def class_statistics(self) -> Dict[str, Any]:
        """Return mean, median, std for Average and each exam."""
        stats = {}
        for col in ['Exam_1','Exam_2','Exam_3','Average']:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std())
            }
        # top performers
        stats['top_5'] = self.df.sort_values('Average', ascending=False).head(5)[['W_Number','First_Name','Last_Name','Average']].to_dict(orient='records')
        return stats

    def export_class_stats(self, filename: str = 'class_statistics.csv'):
        stats = self.class_statistics()
        # create a compact table for export
        rows = []
        for col in ['Exam_1','Exam_2','Exam_3','Average']:
            rows.append({'Metric': col, 'Mean': stats[col]['mean'], 'Median': stats[col]['median'], 'Std': stats[col]['std']})
        out_path = os.path.join(self.config.RESULTS_PATH, filename)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logging.info(f"Exported class stats to {out_path}")
        return out_path


# -------------------------------------------------------------
# Parent: AnalysisOps - math and vector utilities
# -------------------------------------------------------------
class AnalysisOps:
    """Parent class for analysis operations."""


    def mean_std(self, arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        return {'mean': float(np.mean(a)), 'std': float(np.std(a)), 'median': float(np.median(a))}


# -------------------------------------------------------------
# Child: PickleHandler - handles study-hour pickle
# -------------------------------------------------------------
class PickleHandler(AnalysisOps, DataHandler):
    """Child class to read pickle data about study hours."""

    def __init__(self, pkl_path: str):
        DataHandler.__init__(self)
        self.pkl_path = pkl_path
        self.df = self.read_pickle(pkl_path)

    def average_study_hours(self) -> pd.Series:
        """Return average study hours per student over the 4 weeks."""
        if self.df.empty:
            return pd.Series()
        self.df['Avg_Study_Hours'] = self.df[['Week1','Week2','Week3','Week4']].mean(axis=1)
        return self.df.set_index('W_Number')['Avg_Study_Hours']

    def student_study_vector(self, w_number: str) -> List[float]:
        row = self.df[self.df['W_Number'] == w_number]
        if row.empty:
            logging.warning(f"No study data for {w_number}")
            return []
        return row[['Week1','Week2','Week3','Week4']].iloc[0].tolist()

    def export_study_averages(self, filename: str = 'study_averages.csv'):
        avgs = self.average_study_hours()
        out_path = os.path.join(self.config.RESULTS_PATH, filename)
        avgs.reset_index().to_csv(out_path, index=False)
        logging.info(f"Exported study averages to {out_path}")
        return out_path
