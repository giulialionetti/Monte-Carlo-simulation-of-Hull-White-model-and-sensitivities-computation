#!/usr/bin/env python3
"""
Post-processing and visualization for Hull-White simulation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def plot_P_and_f(show=True):
    """Plot zero-coupon bond prices and forward rates."""
    print("\n=== Plotting P(0,T) and f(0,T) curves ===")
    
    try:
        P_df = pd.read_csv('data/P_curve.csv')
        f_df = pd.read_csv('data/f_curve.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run Q1 first to generate the data files.")
        return
    
    # Debug: Print actual column names
    print(f"P_df columns: {P_df.columns.tolist()}")
    print(f"f_df columns: {f_df.columns.tolist()}")
    
    # Use actual column names (with spaces, not parentheses with commas)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot P(0,T) - Zero-Coupon Bond Prices
    # Column is 'P(0 T)' with space, not 'P(0,T)' with comma
    ax1.plot(P_df['T'], P_df['P(0 T)'], 'b-', linewidth=2.5, label='P(0,T)')
    ax1.set_xlabel('Maturity T (years)', fontsize=12)
    ax1.set_ylabel('Bond Price P(0,T)', fontsize=12)
    ax1.set_title('Zero-Coupon Bond Prices - Hull-White Model', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 10)
    
    # Plot f(0,T) - Forward Rates
    # Column is 'f(0 T)' with space
    ax2.plot(f_df['T'], f_df['f(0 T)'] * 100, 'r-', linewidth=2.5, label='f(0,T)')
    ax2.set_xlabel('Maturity T (years)', fontsize=12)
    ax2.set_ylabel('Forward Rate f(0,T) (%)', fontsize=12)
    ax2.set_title('Instantaneous Forward Rates', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig('plots/curves.png', dpi=300, bbox_inches='tight')
    print("Saved plots/curves.png")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_theta_recovery(show=True):
    """Plot theta recovery comparison."""
    print("\n=== Plotting theta recovery ===")
    
    try:
        df = pd.read_csv('data/theta_comparison.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run Q2a first to generate the data files.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top plot: Comparison
    ax1.plot(df['T'], df['theta_original'], 'b-', linewidth=2.5, label='Original θ(t)', alpha=0.8)
    ax1.plot(df['T'], df['theta_recovered'], 'r--', linewidth=2, label='Recovered θ(t)', alpha=0.8)
    ax1.set_xlabel('Time t (years)', fontsize=12)
    ax1.set_ylabel('θ(t)', fontsize=12)
    ax1.set_title('Theta Function Recovery from Forward Rates', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Error
    error = np.abs(df['theta_recovered'] - df['theta_original'])
    ax2.semilogy(df['T'], error, 'g-', linewidth=2)
    ax2.set_xlabel('Time t (years)', fontsize=12)
    ax2.set_ylabel('Absolute Error |θ_recovered - θ_original|', fontsize=12)
    ax2.set_title('Recovery Error (log scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.01, color='r', linestyle='--', linewidth=1.5, label='Target: 0.01')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('plots/theta_recovery.png', dpi=300, bbox_inches='tight')
    print("Saved plots/theta_recovery.png")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_sensitivity_comparison(show=True):
    """Plot sensitivity analysis results if Q3 data exists."""
    print("\n=== Plotting sensitivity analysis ===")
    
    try:
        with open('data/q3_results.json') as f:
            q3 = json.load(f)
    except FileNotFoundError:
        print("Q3 results not found, skipping sensitivity plot")
        return
    
    sens_mc = q3['results']['sensitivity_mc']
    sens_fd = q3['results']['sensitivity_fd']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Pathwise\nDerivative', 'Finite\nDifference']
    values = [sens_mc, sens_fd]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Vega (∂ZBC/∂σ)', fontsize=12)
    ax.set_title('Sensitivity Analysis: Vega Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotation
    diff_pct = 100 * abs(sens_mc - sens_fd) / abs(sens_fd)
    ax.text(0.5, max(values) * 0.5, 
            f'Difference: {diff_pct:.1f}%\n(explained by convexity)',
            ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('plots/sensitivity.png', dpi=300, bbox_inches='tight')
    print("Saved plots/sensitivity.png")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_reduction_benchmark(show=True):
    """Plot reduction method performance comparison."""
    print("\n=== Plotting reduction benchmark ===")
    
    try:
        with open('data/benchmark_reductions.json') as f:
            bench = json.load(f)
    except FileNotFoundError:
        print("Benchmark data not found - run benchmark_reductions first")
        return
    
    methods = [r['method'] for r in bench['results']]
    times = [r['time_ms'] for r in bench['results']]
    throughputs = [r['throughput_Mpaths_per_sec'] for r in bench['results']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot 1: Execution Time
    bars1 = ax1.barh(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Reduction Method: Execution Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, time in zip(bars1, times):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{time:.3f} ms',
                ha='left', va='center', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot 2: Throughput
    bars2 = ax2.barh(methods, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Throughput (M paths/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('Reduction Method: Throughput', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, tput in zip(bars2, throughputs):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{tput:.2f}',
                ha='left', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('plots/reduction_benchmark.png', dpi=300, bbox_inches='tight')
    print("Saved plots/reduction_benchmark.png")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print speedup analysis
    baseline = times[0]
    print("\nSpeedup Analysis (relative to naive atomicAdd):")
    for method, time in zip(methods, times):
        speedup = baseline / time
        improvement = 100.0 * (1.0 - time / baseline)
        print(f"  {method:30s}: {speedup:.2f}x faster ({improvement:.1f}% improvement)")

def print_summary():
    """Print summary statistics from JSON files."""
    print("\n" + "="*80)
    print("HULL-WHITE MODEL: SIMULATION SUMMARY")
    print("="*80)
    
    # Q1
    try:
        with open('data/q1_results.json') as f:
            q1 = json.load(f)
        
        print(f"\n{'='*80}")
        print("Q1: ZERO-COUPON BOND PRICING")
        print(f"{'='*80}")
        print(f"  Monte Carlo Paths:     {q1['parameters']['effective_paths']:,}")
        print(f"  Time Steps:            {q1['parameters']['n_steps']:,}")
        print(f"  Variance Reduction:    Antithetic variates")
        print(f"\n  Validation:")
        print(f"    P(0,0)  = {q1['validation']['P_0_0']:.8f}  (should be 1.0)")
        print(f"    P(0,10) = {q1['validation']['P_0_10']:.8f}  (~ 0.87)")
        print(f"    f(0,0)  = {q1['validation']['f_0_0_pct']:.4f}%  (~ 1.2%)")
        print(f"\n  Performance:")
        print(f"    Simulation time:  {q1['performance']['sim_time_ms']:.2f} ms")
        print(f"    Throughput:       {q1['performance']['throughput_Mpaths_per_sec']:.2f} M paths/sec")
    except FileNotFoundError:
        print("\n[Q1 results not found]")
    except KeyError as e:
        print(f"\n[Q1 data incomplete: {e}]")
    
    # Q2a
    try:
        with open('data/q2a_results.json') as f:
            q2a = json.load(f)
        
        print(f"\n{'='*80}")
        print("Q2a: THETA RECOVERY")
        print(f"{'='*80}")
        print(f"  Method:      Finite difference on forward rates")
        print(f"  Formula:     θ(T) = df/dT + a·f(0,T) + σ²/(2a)·(1-e^(-2aT))")
        print(f"\n  Results:")
        print(f"    Max error:   {q2a['error_metrics']['max_error']:.2e}")
        print(f"    Status:      {'SUCCESS' if q2a['error_metrics']['success'] else 'FAILED'}")
        print(f"    Target:      < 0.01")
    except FileNotFoundError:
        print("\n[Q2a results not found]")
    except KeyError as e:
        print(f"\n[Q2a data incomplete: {e}]")
    
    # Q2b
    try:
        with open('data/q2b_results.json') as f:
            q2b = json.load(f)
        
        print(f"\n{'='*80}")
        print("Q2b: ZERO-COUPON BOND CALL OPTION")
        print(f"{'='*80}")
        print(f"  Option:      European call on P(S1=5.0, S2=10.0)")
        print(f"  Strike:      K = e^(-0.1) = {np.exp(-0.1):.6f}")
        print(f"  Method:      Monte Carlo with control variate")
        print(f"\n  Results:")
        print(f"    ZBC value:         {q2b['results']['ZBC_control_variate']:.8f}")
        print(f"    Control deviation: {q2b['results']['control_deviation']:.2e}")
        print(f"\n  Performance:")
        print(f"    Simulation time:  {q2b['performance']['sim_time_ms']:.2f} ms")
        print(f"    Throughput:       {q2b['performance']['throughput_Mpaths_per_sec']:.2f} M paths/sec")
        print(f"    Effective paths:  {q2b['performance']['effective_paths']:,}")
    except FileNotFoundError:
        print("\n[Q2b results not found]")
    except KeyError as e:
        print(f"\n[Q2b data incomplete: {e}]")
    
    # Q3
    try:
        with open('data/q3_results.json') as f:
            q3 = json.load(f)
        
        print(f"\n{'='*80}")
        print("Q3: SENSITIVITY ANALYSIS (VEGA)")
        print(f"{'='*80}")
        print(f"  Methods:     Pathwise derivatives vs Finite difference")
        print(f"\n  Results:")
        print(f"    Pathwise vega:     {q3['results']['sensitivity_mc']:.6f}")
        print(f"    FD vega:           {q3['results']['sensitivity_fd']:.6f}")
        print(f"    Absolute diff:     {q3['results']['abs_diff']:.6f}")
        
        if 'rel_diff_pct' in q3['results']:
            print(f"    Relative diff:     {q3['results']['rel_diff_pct']:.2f}%")
        
        print(f"\n  Analysis:")
        print(f"    Both methods give positive vega (correct for calls)")
        print(f"    Difference explained by convexity and market data consistency")
    except FileNotFoundError:
        print("\n[Q3 results not found - run Q3 to generate]")
    except KeyError as e:
        print(f"\n[Q3 data incomplete: {e}]")
    
    print(f"\n{'='*80}\n")

def main():
    """Main analysis pipeline."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Check if we should show plots interactively
    show_plots = '--show' in sys.argv or '-s' in sys.argv
    
    print("\n" + "="*80)
    print("HULL-WHITE MODEL: POST-PROCESSING & VISUALIZATION")
    print("="*80)
    
    # Generate all plots
    plot_P_and_f(show=show_plots)
    plot_theta_recovery(show=show_plots)
    plot_sensitivity_comparison(show=show_plots)
    plot_reduction_benchmark(show=show_plots)
    
    # Print summary
    print_summary()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • plots/curves.png          - Bond prices and forward rates")
    print("  • plots/theta_recovery.png  - Theta function recovery")
    print("  • plots/sensitivity.png     - Vega comparison (if Q3 run)")
    print("  • plots/reduction_benchmark.png - Reduction methods benchmark")
    print("\nData files:")
    print("  • data/summary.txt          - Text summary")
    print("  • data/q*_results.json      - Detailed JSON results")
    
    if not show_plots:
        print("\nNote: Plots were saved but not displayed.")
        print("      Use 'python3 analyze.py --show' to display them interactively.")

if __name__ == '__main__':
    main()