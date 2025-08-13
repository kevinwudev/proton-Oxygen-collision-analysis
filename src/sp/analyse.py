from __future__ import annotations
from chromo.kinematics import CenterOfMass 
from chromo.constants import TeV
from chromo.models import (DpmjetIII191, DpmjetIII193, DpmjetIII307, EposLHC, EposLHCR, QGSJetIII, Sibyll23e) 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import namedtuple
from particle import Particle
from particle import literals as lp
import polars as pl
import numpy as np
from pathlib import Path
from collections import defaultdict

from sp import root_base

def adjust(df):
    df_update = df.filter(~(pl.col("eta").is_infinite() & (pl.col("eta") < 0)))
    df_update = df_update.with_columns(
        pl.when(pl.col("n_wounded") == 0)
        .then(1)
        .otherwise(pl.col("n_wounded"))
        .alias("n_wounded")
    )

    info = f"Remove # of data {len(df) - len(df_update)} rows, {len(df_update)} rows remain."

    return df_update, info


def plot_with_ratio(gs, 
                    fig, 
                    df_dict : dict[str, pl.DataFrame], 
                    col_name  : str = 'eta',
                    pid       : int = None,
                    title     : str = "", 
                    x_label   : str = r"$\eta$", 
                    y_label   : str = r"$dN/d\eta$",
                    main_ylim : tuple[float, float] = None, 
                    ratio_ylim: tuple[float, float] = (0.5, 2.0),
                    bins      : int = 50,
                    range     : tuple[float, float] = None,
                    gevt      : int = 10,
                    islog     : bool = False,
                    ignore_wo : bool = False,
                    ):
    
    # 5 : 1
    ax_main = fig.add_subplot(gs[0])   # 5/6
    ax_ratio = fig.add_subplot(gs[1])  # 1/6
    
    # set "EPOS-LHC-R" as the reference model
    ref_model_name : str          = "EPOS-LHC-R"
    ref_model      : pl.DataFrame = df_dict[ref_model_name].filter(pl.col('pid') == pid) if pid else df_dict[ref_model_name]
    ref_model , _                 = adjust(ref_model)
    ref_model_data : np.array     = ref_model[col_name].to_numpy()
    print(f"ref model: {ref_model_name}, data size {len(ref_model_data)}")

    # If range is not provided, calculate it as mean +/- 3*std of the reference data
    if range is None:
        mean = np.mean(ref_model_data)
        std = np.std(ref_model_data)
        range = (mean - 3 * std, mean + 3 * std)
        print(f"Range not provided. Calculated range based on reference model: {range}")

    # set up the weights of reference data (EPOS) for histogram plot.
    if ignore_wo:
        weights = np.ones(len(ref_model)) / gevt
    else:
        weights = 1 / (ref_model['n_wounded'] * gevt) if 'n_wounded' in ref_model.columns else np.ones(len(ref_model)) / gevt

    # set up log parameter for histogram
    if islog:
        bin_edges = np.logspace(-5, 1, 100)
        ref_counts, ref_bin_edges = np.histogram(ref_model_data, bins=bin_edges, range=range, weights=weights)
        bin_centers = 0.5 * (ref_bin_edges[:-1] + ref_bin_edges[1:]) # 計算 bin 中心位置
        bin_width = np.diff(ref_bin_edges) # 計算 bin 寬度

    else:
        ref_counts, ref_bin_edges = np.histogram(ref_model_data, bins=bins, range=range, weights=weights)
        bin_width = ref_bin_edges[1] - ref_bin_edges[0]
    
    ref_counts_density = ref_counts / bin_width

    
    for i, (model, df) in enumerate(df_dict.items()):
        df, remove_info = adjust(df)
        df = df.filter(pl.col('pid') == pid) if pid else df
        print(f' {model=} require {pid=} ') if pid else None
        
        data = df[col_name]

        # set up the weights of current data for histogram plot.
        if ignore_wo:
            weights = np.ones(len(df)) / gevt
        else:
            weights = 1 / (df['n_wounded'] * gevt) if 'n_wounded' in df.columns else np.ones(len(df)) / gevt

        print(f"{model=} | {df.shape=}", 
              f"remove eta -inf, set n_wounded >=0", 
              f"{remove_info}" if remove_info else ""
              )
        
        # main - histogram
        if islog:
            log_bin_edges = np.logspace(-5, 1, 100)
            current_counts, bin_edges = np.histogram(data, bins=log_bin_edges, range=range, weights=weights)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # 計算 bin 中心位置
            bin_width = np.diff(bin_edges) # 計算 bin 寬度
            
        else:
            current_counts, bin_edges = np.histogram(data, bins=bins, range=range, weights=weights)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # 計算 bin 中心位置
            bin_width = bin_edges[1] - bin_edges[0] # 計算 bin 寬度

        current_counts_density = current_counts / bin_width

        # main plot
        ax_main.stairs(current_counts_density, bin_edges, 
                       label=model, alpha=0.7, linewidth=1.5)

        # ratio - avoid divide by zero
        ratio = np.divide(current_counts_density, ref_counts_density, 
                         out=np.ones_like(current_counts_density, dtype=float), 
                         where=ref_counts_density!=0)
        
        # subplot - plot
        # ax_ratio.plot(bin_centers, ratio, drawstyle='steps-mid', linewidth=1.2)
        ax_ratio.stairs(ratio, bin_edges, linewidth=1.2)
    
    # main plot setting
    ax_main.set_title(title, fontsize=11, pad=15, weight='bold')
    ax_main.set_ylabel(y_label, fontsize=10)
    ax_main.set_xscale('log') if islog else None
    ax_main.set_yscale('log') if islog else None
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.tick_params(axis='y', labelsize=9)
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax_main.grid(True, linestyle=":", alpha=0.4)
    ax_main.set_xlim(range[0], range[1])
    
    # main plot y-label range
    if main_ylim is not None:
        ax_main.set_ylim(main_ylim)
    else:
        ax_main.margins(y=0.1)
    
    # subplot setting
    ax_ratio.set_ylabel(f"Model/{ref_model_name}", fontsize=9, ha='center')
    ax_ratio.set_xlabel(x_label, fontsize=10)
    ax_ratio.set_xlim(range[0], range[1])
    ax_ratio.set_ylim(ratio_ylim)
    ax_ratio.set_xscale('log') if islog else None
    ax_ratio.tick_params(labelsize=9)
    ax_ratio.grid(True, linestyle=":", alpha=0.4)
    ax_ratio.axhline(y=1, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # share x-axis
    ax_main.sharex(ax_ratio)
    
    return ax_main, ax_ratio

def plot_all_plot(plot_specs : list[PlotSpec], 
         gevt : int,
         output_path_name : str,
         title : str,
         ):

    
    ncols = 2
    nrows = len(plot_specs) // ncols
    
    fig = plt.figure(figsize=(5 * ncols, 4.6 * nrows), dpi=200)
    fig.suptitle(f"{title}, generated in {gevt} events",
                fontsize=16, fontweight='bold')
    
    print("🎨 Start plotting figure...")
    
    # set up GridSpec
    for idx, spec in enumerate(plot_specs):
        row = idx // ncols 
        col = idx % ncols  
        
        # set up the position in the plot grid
        start_row = row * 6
        end_row = start_row + 6
        
        # 5 : 1
        gs_subplot = gridspec.GridSpec(2, 1, 
                                     figure=fig,
                                     left=col/ncols + 0.04,     
                                     right=(col+1)/ncols - 0.02,
                                     bottom=1 - end_row/(nrows*6) + 0.06,  
                                     top=1 - start_row/(nrows*6) - 0.06,   
                                     height_ratios=[5, 1],
                                     hspace=0.1)  
        
        # plot the figure
        try:
            plot_with_ratio(gs_subplot, fig, spec.df_dict, 
                              col_name=spec.col_name, title=spec.title, pid = spec.pid,
                              main_ylim=spec.main_ylim, ratio_ylim=spec.ratio_ylim,
                              range=spec.range, x_label=spec.x_label, y_label=spec.y_label,
                              gevt = gevt, islog = spec.islog, ignore_wo = spec.ignore_wo)
        except Exception as e:
            print(f"⚠️  error when plotting subfigure {idx}: {e}")
            continue
    
    # save
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.95, 
                       hspace=0.3, wspace=0.25)  
    
    try:
        output_path = Path(root_base) / "figure" / f"{output_path_name}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ save to: {root_base}/figure/{output_path_name}.pdf")
    except Exception as e:
        print(f"❌ error when saving figure: {e}")

def load_parquet_by_prefix_and_model(folder: str = "pq") -> dict:

        result = defaultdict(dict)

        for file in Path(folder).glob("*.parquet"):
            stem_parts = file.stem.split("_")
            if len(stem_parts) < 2:
                print(f"ignore：{file.name}")
                continue
            
            prefix1 = stem_parts[0]
            module = stem_parts[1]

            try:
                df = pl.read_parquet(file)
                result[prefix1][module] = df
            except Exception as e:
                print(f"Error when attempting to reading :{file.name} | {e}")

        return dict(result)
