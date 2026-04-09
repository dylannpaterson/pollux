import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_lc(db_path, target_uuid, output_path):
    conn = sqlite3.connect(db_path)
    
    query = f"""
        SELECT i.obs_time, o.flux_raw, o.mag_raw, o.err_m, i.filter
        FROM observations o
        JOIN images i ON o.image_id = i.image_id
        WHERE o.target_uuid = '{target_uuid}' AND o.is_detection = 1
        ORDER BY i.obs_time
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No detections found for target {target_uuid}")
        return

    plt.figure(figsize=(10, 6))
    
    # Mathematical conversion from natural log error to flux error:
    # Since err_m is sigma_lnf, and d(ln f)/df = 1/f,
    # then sigma_lnf = sigma_f / f  =>  sigma_f = f * sigma_lnf
    for filt in df['filter'].unique():
        sub = df[df['filter'] == filt]
        flux = sub['flux_raw']
        err_m = sub['err_m']
        err_f = flux * err_m
        
        plt.errorbar(sub['obs_time'], flux, yerr=err_f, 
                     fmt='o-', label=filt, capsize=3)
    
    plt.xlabel('Time (MJD)')
    plt.ylabel('Raw Flux (ADU)')
    plt.title(f'Light Curve for Target {target_uuid}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/prototype/microlensing_final.db")
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--out", default="db_lightcurve.png")
    args = parser.parse_args()
    
    plot_lc(args.db, args.uuid, args.out)
