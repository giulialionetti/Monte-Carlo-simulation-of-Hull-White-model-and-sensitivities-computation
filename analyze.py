#!/usr/bin/env python3
"""
Post-processing and visualization for Hull-White simulation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_P_and_f():
    """Plot zero-coupon bond prices and forward rates."""
    P_df = pd.read_csv('data/P_curve.csv')
    f_df = pd.read_csv('data/f_curve.csv')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot P(0,T)
    print(P_df.head())
    ax1.plot(P_df['T'], P_df['P(0 T)'], 'b-', linewidth=2)
    ax1.set_xlabel('Maturity T (years)')
    ax1.set_ylabel('P(0,T)')
    ax1.set_title('Zero-Coupon Bond Prices')
    ax1.grid(True, alpha=0.3)
    
    # Plot f(0,T)
    ax2.plot(f_df['T'], f_df['f(0 T)'] * 100, 'r-', linewidth=2)
    ax2.set_xlabel('Maturity T (years)')
    ax2.set_ylabel('f(0,T) (%)')
    ax2.set_title('Forward Rates')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/curves.png', dpi=300)
    print("Saved plots/curves.png")

def plot_theta_recovery():
    """Plot theta recovery comparison."""
    df = pd.read_csv('data/theta_comparison.csv')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['T'], df['theta_original'], 'b-', linewidth=2, label='Original θ(t)')
    plt.plot(df['T'], df['theta_recovered'], 'r--', linewidth=2, label='Recovered θ(t)')
    plt.xlabel('Time t (years)')
    plt.ylabel('theta(t)')
    plt.title('Theta Function Recovery from Forward Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/theta_recovery.png', dpi=300)
    print("Saved plots/theta_recovery.png")

def print_summary():
    """Print summary statistics from JSON files."""
    with open('data/q1_results.json') as f:
        q1 = json.load(f)
    
    with open('data/q2a_results.json') as f:
        q2a = json.load(f)
    
    with open('data/q2b_results.json') as f:
        q2b = json.load(f)
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    print(f"\nQ1: Zero-Coupon Bond Pricing")
    print(f"  P(0,0) = {q1['validation']['P_0_0']:.8f}")
    print(f"  P(0,10) = {q1['validation']['P_0_10']:.8f}")
    print(f"  Throughput: {q1['performance']['throughput_Mpaths_per_sec']:.2f} M paths/sec")
    
    print(f"\nQ2a: Theta Recovery")
    print(f"  Max error: {q2a['error_metrics']['max_error']:.2e}")
    print(f"  Status: {'SUCCESS' if q2a['error_metrics']['success'] else 'FAILED'}")
    
    print(f"\nQ2b: ZBC Option")
    print(f"  ZBC(5,10,e^-0.1) = {q2b['results']['ZBC_control_variate']:.8f}")
    print(f"  Control deviation: {q2b['results']['control_deviation']:.2e}")
    print(f"  Throughput: {q2b['performance']['throughput_Mpaths_per_sec']:.2f} M paths/sec")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)
    
    plot_P_and_f()
    # TODO 
    plot_theta_recovery()
    print_summary()
    
    print("\nAnalysis complete. Check plots/ and data/summary.txt")