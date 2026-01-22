import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

def plot_local_global(heart_data, wine_2N1L, wine_3N1L,n, n_layers):
    """
    Reproduces plots from the paper depicting the difference in test accuracy 
    when using global vs local depolarising noise models for three datasets
    """
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(30,10), sharex=True)

    #Wine Dataset (2 qubits, 1 layer)
    data = wine_2N1L
    p_local = np.array([d[0] for d in data])
    p_global = 1 - (1-p_local)**(n*n_layers) #formula used to equate survival probability of both models

    local_acc = np.array([d[1] for d in data])
    local_std = np.array([d[2] for d in data])
    global_acc = np.array([d[3] for d in data])
    global_std = np.array([d[4] for d in data])

    diff = global_acc - local_acc
    diff_std = np.sqrt(local_std**2 + global_std**2) #error(a-b) = sqrt(a^2 + b^2)

    p_glob_idx = [0,1,2,4,6,8,10,11] #display selective values for p_global for visual purposes

    ax1.axhline(y=0, color = "gray", linestyle="--", linewidth=1.5, alpha=0.3)
    ax1.fill_between(p_local, diff-diff_std, diff+diff_std, alpha=0.1, color = 'blue')
    ax1.plot(p_local, diff, 'o-', color = "blue", linewidth=2, markersize=8, label="Wine Dataset (N=2,L=1)")
    ax1.set_ylabel("Accuracy Difference", fontsize=22)
    ax1.grid(True)
    ax1.legend(fontsize=16)
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(p_local[p_glob_idx])
    ax1_top.set_xticklabels([f"{p_global[i]:.2f}" for i in p_glob_idx])

    #Heart Disease Dataset
    data = heart_data
    p_local = np.array([d[0] for d in data])

    p_global = 1 - (1-p_local)**(n*n_layers) #to match survival prob
    local_acc = np.array([d[1] for d in data])
    local_std = np.array([d[2] for d in data])
    global_acc = np.array([d[3] for d in data])
    global_std = np.array([d[4] for d in data])

    diff = global_acc - local_acc
    diff_std = np.sqrt(local_std**2 + global_std**2) #error(a-b) = sqrt(a^2 + b^2)
    p_glob_idx = [0,2,3,4,5,6,7,9]

    ax2.axhline(y=0, color = "gray", linestyle="--", linewidth=1.5, alpha=0.3)
    ax2.fill_between(p_local, diff-diff_std, diff+diff_std, alpha=0.1, color = 'blue')
    ax2.plot(p_local, diff, 'o-', color = "blue", linewidth=2, markersize=8, label="Heart Disease Dataset")
    ax2.set_xlabel('Local Depolarising Noise Probability', fontsize=22)
    ax2.grid(True)
    ax2_top = ax2.twiny() #dual axis
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(p_local[p_glob_idx])
    ax2_top.set_xticklabels([f"{p_global[i]:.2f}" for i in p_glob_idx])
    ax2_top.set_xlabel("Equivalent Global Depolarising Noise", fontsize=22)
    ax2.legend(fontsize=16)

    #Wine Dataset (3 qubits, 1 layer)
    data = wine_3N1L
    p_local = np.array([d[0] for d in data])

    n=3
    n_layers = 1
    p_global = 1 - (1-p_local)**(n*n_layers) #to match survival prob
    local_acc = np.array([d[1] for d in data])
    local_std = np.array([d[2] for d in data])
    global_acc = np.array([d[3] for d in data])
    global_std = np.array([d[4] for d in data])

    diff = global_acc - local_acc
    diff_std = np.sqrt(local_std**2 + global_std**2)
    p_glob_idx = [0,1,2,4,6,8, 10, 11] #display selective values for p_global for visual purposes

    ax3.axhline(y=0, color = "gray", linestyle="--", linewidth=1.5, alpha=0.3)
    ax3.fill_between(p_local, diff-diff_std, diff+diff_std, alpha=0.1, color = 'blue')
    ax3.plot(p_local, diff, 'o-', color = "blue", linewidth=2, markersize=8, label="Wine Dataset (N=3,L=1)")
    ax3.grid(True)
    ax3.legend(fontsize=16, loc='upper left')
    ax3_top = ax3.twiny()
    ax3_top.set_xlim(ax3.get_xlim())

    ax3_top.set_xticks(p_local[p_glob_idx])
    ax3_top.set_xticklabels([f"{p_global[i]:.2f}" for i in p_glob_idx])

    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    ax3.tick_params(labelsize=18)

    ax1_top.tick_params(labelsize=18)
    ax2_top.tick_params(labelsize=18)
    ax3_top.tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()


def plot_upper_bound(p_local_list,heart_margin, heart_upper, gaus_margin, gaus_upper, bc_margin, bc_upper, wineL1_margin, wineL1_upper, wineL2_margin, wineL2_upper):
    """
    Reproduces plots from paper depicting the theoretical margin values
    computed using the upper bound and the actual noisy margin values 
    for visual comparison for various datasets
    """
    fig, axes = plt.subplots(1,5, figsize=(15,4), sharex=True)
    
    axes[0].plot(p_local_list, heart_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[0].plot(p_local_list, heart_upper,'o--', linewidth=2, markersize=6, label = "Upper Bound", color = "red")
    axes[0].set_ylabel("Margin", fontsize=12)
    axes[0].grid(True, linestyle=":", linewidth = 0.5)
    axes[0].tick_params(labelsize=10)

    axes[1].plot(p_local_list, gaus_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[1].plot(p_local_list, gaus_upper,'o--', linewidth=2, markersize=6, label = "Upper Bound", color = "red")
    axes[1].grid(True, linestyle=":", linewidth = 0.5)
    axes[1].tick_params(labelsize=10)

    axes[2].plot(p_local_list, bc_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[2].plot(p_local_list, bc_upper,'o--', linewidth=2, markersize=6, label = "Upper Bound", color = "red")
    axes[2].grid(True, linestyle=":", linewidth = 0.5)
    axes[2].tick_params(labelsize=10)

    axes[3].plot(p_local_list, wineL1_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[3].plot(p_local_list, wineL1_upper,'o--', linewidth=2, markersize=6, label = "Upper Bound", color = "red")
    axes[3].grid(True, linestyle=":", linewidth = 0.5)
    axes[3].tick_params(labelsize=10)

    axes[4].plot(p_local_list, wineL2_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[4].plot(p_local_list, wineL2_upper,'o--', linewidth=2, markersize=6, label = "Upper Bound", color = "red")
    axes[4].grid(True, linestyle=":", linewidth = 0.5)
    axes[4].tick_params(labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor = (0.99, 0.1),
            ncol = 1, fontsize = 11, frameon=True)
    fig.text(0.5, 0.02, 'Local Depolarising Noise', ha='center', fontsize = 13)

    axes[0].text(0.5, 0.95, 'Heart Disease', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.65, 0.95, 'Gaussian', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.5, 0.95, 'Breast Cancer', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.4, 0.95, 'Wine (N=2,L=1)', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[4].text(0.4, 0.95, 'Wine (N=2, L=2)', transform = axes[4].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    plt.show()


def plot_lower_bound(p_local_list,heart_margin, heart_lower, gaus_margin, gaus_lower, bc_margin, bc_lower, wineL1_margin, wineL1_lower, wineL2_margin, wineL2_lower):
    """
    Reproduces plots from paper depicting the theoretical margin values
    computed using the lower bound and the actual noisy margin values 
    for visual comparison for various datasets
    """
    fig, axes = plt.subplots(1,5, figsize=(15,4), sharex=True)
    print(axes.shape)
    axes[0].plot(p_local_list, heart_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[0].plot(p_local_list, heart_lower,'o--', linewidth=2, markersize=6, label = "Lower Bound", color = "black")
    axes[0].set_ylabel("Margin", fontsize=12)
    axes[0].grid(True, linestyle=":", linewidth = 0.5)
    axes[0].tick_params(labelsize=10)

    axes[1].plot(p_local_list, gaus_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[1].plot(p_local_list, gaus_lower,'o--', linewidth=2, markersize=6, label = "Lower Bound", color = "black")
    axes[1].grid(True, linestyle=":", linewidth = 0.5)
    axes[1].tick_params(labelsize=10)

    axes[2].plot(p_local_list, bc_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[2].plot(p_local_list, bc_lower,'o--', linewidth=2, markersize=6, label = "Lower Bound", color = "black")
    axes[2].grid(True, linestyle=":", linewidth = 0.5)
    axes[2].tick_params(labelsize=10)

    axes[3].plot(p_local_list, wineL1_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[3].plot(p_local_list, wineL1_lower,'o--', linewidth=2, markersize=6, label = "Lower Bound", color = "black")
    axes[3].grid(True, linestyle=":", linewidth = 0.5)
    axes[3].tick_params(labelsize=10)
    
    axes[4].plot(p_local_list, wineL2_margin, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[4].plot(p_local_list, wineL2_lower,'o--', linewidth=2, markersize=6, label = "Lower Bound", color = "black")
    axes[4].grid(True, linestyle=":", linewidth = 0.5)
    axes[4].tick_params(labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor = (0.99, 0.1),
            ncol = 1, fontsize = 11, frameon=True)
    fig.text(0.5, 0.02, 'Local Depolarising Noise', ha='center', fontsize = 13)

    axes[0].text(0.5, 0.95, 'Heart Disease', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.65, 0.95, 'Gaussian', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.5, 0.95, 'Breast Cancer', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.4, 0.95, 'Wine (N=2,L=1)', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[4].text(0.4, 0.95, 'Wine (N=2, L=2)', transform = axes[4].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def plot_boxplots(htru2_df, wine_df, heart_df, gaus_df):
    """
    Reproduces plots from paper comparing boxplots highlighting
    the median (per-sample) geometric margin obtained for 
    three corruption levels for four datasets
    """
    fig,axes = plt.subplots(1,4, figsize=(14,6), sharex=True)

    #HTRU2 Dataset
    sns.boxplot(x = "corruption_levels", y ="geometric_margin", data = htru2_df, ax = axes[0], palette= "viridis", showfliers=False )
    axes[0].set_xlabel("", fontsize = 14)
    axes[0].set_ylabel("Median Geometric Margin", fontsize = 14)
    axes[0].set_title("(a)", loc="left")
    axes[0].grid(True, linewidth=0.5, linestyle = ":")

    #Wine Dataset
    sns.boxplot(x = "corruption_levels", y ="geometric_margin", data = wine_df, ax = axes[1], palette= "viridis", showfliers=False )
    axes[1].set_xlabel("", fontsize = 14, loc="right")
    axes[1].set_ylabel("", fontsize = 14)
    axes[1].set_title("(b)", loc="left")
    axes[1].grid(True, linewidth=0.5, linestyle = ":")

    #Heart Disease Dataset
    sns.boxplot(x = "corruption_levels", y ="geometric_margin", data = heart_df, ax = axes[2], palette= "viridis", showfliers=False )
    axes[2].set_xlabel("", fontsize = 14)
    axes[2].set_ylabel("", fontsize = 14)
    axes[2].set_title("(c)", loc="left")
    axes[2].grid(True, linewidth=0.5, linestyle = ":")

    #Gaussian Dataset
    sns.boxplot(x = "corruption_levels", y ="geometric_margin", data = gaus_df, ax = axes[3], palette= "viridis", showfliers=False )
    axes[3].set_xlabel("", fontsize = 14)
    axes[3].set_ylabel("", fontsize = 14)
    axes[3].set_title("(d)", loc="left")
    axes[3].grid(True, linewidth=0.5, linestyle = ":")

    axes[0].text(0.76, 0.986, 'HTRU2', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.65, 0.986, 'Wine Quality', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.8, 0.983, 'Heart', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.7, 0.983, 'Gaussian', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    fig.text(0.53, 0.005, 'Corruption Fraction', ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.show()

def dual_plot(htru2_acc, htru2_std, htru2_geom_margin, wine_acc, wine_std, wine_geom_margin, heart_acc, heart_std, heart_geom_margin, gaus_acc, gaus_std, gaus_geom_margin):
    """
    Reproduces plots from paper highlighting
    the decay in test accuracy and median geometric margin
    with increasing corruption levels for four datasets
    """
    fig,axes = plt.subplots(1,4, figsize=(14,6))
    corr_levels = [0,0.1,0.25,0.5,0.6,0.75,1.0]

    #Combined legend with twin axes
    ax2 = axes[0].twinx()

    line1 = axes[0].errorbar(corr_levels, htru2_acc, yerr=htru2_std, marker = 'o',color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, htru2_geom_margin, 'ro--', label="Margin")

    axes[0].set_ylabel("Test Accuracy", fontsize = 14)
    axes[0].set_title("(a)", loc="left")

    #Combine legends
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    #Combined legend with twin axes
    ax2 = axes[1].twinx()

    line1 = axes[1].errorbar(corr_levels, wine_acc, yerr=wine_std, marker = 'o',color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, wine_geom_margin, 'ro--', label="Geometric Margin")

    axes[1].set_title("(b)", loc="left")
    #Combine legends
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    #Combined legend with twin axes
    ax2 = axes[2].twinx()

    line1 = axes[2].errorbar(corr_levels, heart_acc, yerr=heart_std, marker = 'o',color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, heart_geom_margin, 'ro--', label="Geometric Margin")

    axes[2].set_title("(c)", loc="left")

    #Combine legends
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    #Combined legend with twin axes
    ax2 = axes[3].twinx()

    line1 = axes[3].errorbar(corr_levels, gaus_acc, yerr=gaus_std, marker = 'o',color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, gaus_geom_margin, 'ro--', label="Margin")

    ax2.set_ylabel("Median Geometric Margin", fontsize = 14)
    axes[3].set_title("(d)", loc="left")

    #Combine legends
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()


    fig.text(0.5, 0.001, 'Corruption Fraction', ha='center', fontsize = 14)

    axes[0].text(0.08, 0.05, 'HTRU2', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.08, 0.05, 'Wine Quality', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.08, 0.05, 'Heart Disease', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.08, 0.05, 'Gaussian', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    axes[3].legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

def acc_margin_plot(htru2_acc, htru2_geom_margin, wine_acc, wine_geom_margin, heart_acc, heart_geom_margin, gaus_acc, gaus_geom_margin):
    """
    Reproduces plots from paper highlighting
    the linear correlation between test accuracy and 
    median geometric margins using four datasets
    """
    fig,axes = plt.subplots(1,4, figsize=(14,6))
    axes[0].scatter(htru2_geom_margin, htru2_acc ,color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="HTRU2 Dataset")

    r,p = pearsonr(htru2_geom_margin, htru2_acc) #calculate the Pearson correlation coefficient

    #trend line
    z = np.polyfit(htru2_geom_margin, htru2_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)

    axes[0].plot(htru2_geom_margin, trend_line(htru2_geom_margin), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    axes[0].text(0.65, 0.06, f'r={r:.4f}', transform = axes[0].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel("Test Accuracy", fontsize = 14)
    axes[0].set_title("(a)", loc="left")
    axes[0].legend()


    axes[1].scatter(wine_geom_margin, wine_acc, color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="Wine Dataset")

    r,p = pearsonr(wine_geom_margin, wine_acc)

    #trend line
    z = np.polyfit(wine_geom_margin, wine_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)
    axes[1].plot(wine_geom_margin, trend_line(wine_geom_margin), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    axes[1].text(0.65, 0.06, f'r={r:.4f}', transform = axes[1].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))

    axes[1].set_title("(b)", loc="left")
    axes[1].legend()

    axes[2].scatter(heart_geom_margin, heart_acc, color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="Heart Disease Dataset")

    r,p = pearsonr(heart_geom_margin, heart_acc)

    #trend line
    z = np.polyfit(heart_geom_margin, heart_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)
    axes[2].plot(heart_geom_margin, trend_line(heart_geom_margin), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    axes[2].text(0.65, 0.06, f'r={r:.4f}', transform = axes[2].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))
    axes[2].set_title("(c)", loc="left")
    axes[2].legend()


    axes[3].scatter(gaus_geom_margin, gaus_acc,color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="Gaussian Dataset")

    r,p = pearsonr(gaus_geom_margin, gaus_acc)

    #trend line
    z = np.polyfit(gaus_geom_margin, gaus_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)
    axes[3].plot(gaus_geom_margin, trend_line(gaus_geom_margin), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    axes[3].text(0.65, 0.06, f'r={r:.4f}', transform = axes[3].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))

    fig.text(0.5, 0.001, 'Median Geometric Margin', ha='center', fontsize = 13)

    axes[3].set_title("(d)", loc="left")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def ibm_plots(p_local_list_upper, noisy_margin_upper, upper_bound_arr, p_local_list_lower, noisy_margin_lower, lower_bound_arr, ibm_margin):
    """
    Reproduces plots from paper depicting the upper and lower bound values
    with the simulated noisy margin for increasing noise values for comparison
    with the noisy margin value obtained using real quantum hardware
    """
    fig, axes = plt.subplots(1,2, figsize=(15,4), sharex=False)
    print(axes.shape)
    axes[0].plot(p_local_list_upper, noisy_margin_upper, 's-', linewidth=2, markersize=6, color = "blue")
    axes[0].plot(p_local_list_upper, upper_bound_arr,'o--', linewidth=2, markersize=6, label = "Upper Bound", color = "red")
    axes[0].axhline(y=ibm_margin,color = 'green', linestyle='--', linewidth = 2)
    axes[0].set_ylabel("Margin", fontsize=12)
    axes[0].grid(True, linestyle=":", linewidth = 0.5)
    axes[0].tick_params(labelsize=10)


    axes[1].plot(p_local_list_lower, noisy_margin_lower, 's-', linewidth=2, markersize=6, label = 'Empirical Margin', color = "blue")
    axes[1].plot(p_local_list_lower, lower_bound_arr,'o--', linewidth=2, markersize=6, label = "Lower Bound", color = "black")
    axes[1].axhline(y=ibm_margin,color = 'green', linestyle='--', linewidth = 2, label = 'ibm_strasbourg')
    axes[1].grid(True, linestyle=":", linewidth = 0.5)
    axes[1].tick_params(labelsize=10)

    axes[1].text(0.73, 0.95, 'Breast Cancer (Subset)', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    fig.legend(handles1+handles2, labels1+labels2, bbox_to_anchor = (0.49, 0.96),
            ncol = 1, fontsize = 11, frameon
            =True)

    fig.text(0.5, 0.02, 'Local Depolarising Noise', ha='center', fontsize = 13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()