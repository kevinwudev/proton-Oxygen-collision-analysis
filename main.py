from chromo.kinematics import CenterOfMass
from chromo.models import (DpmjetIII191, DpmjetIII193, DpmjetIII307, EposLHC, EposLHCR, QGSJetIII, Sibyll23e)
from chromo.constants import TeV, MeV
import argparse
from collections import namedtuple
from particle import literals as lp
import polars as pl

from sp.gen import run_model
from sp.analyse import load_parquet_by_prefix_and_model, main

kins = [
        [CenterOfMass(5 * TeV, "p", "p"), 'pp'],
        [CenterOfMass(5 * TeV, "p", (16, 8)), 'pO'],
        [CenterOfMass(5 * TeV, (16, 8), "p"), 'Op'],
        [CenterOfMass(5 * TeV, (16, 8), (16, 8)), 'OO'],
    ]

models = [
    EposLHCR,  
    QGSJetIII, 
    # Sibyll23e    # Sibyll23e is currently not used in the main.py due to wounded candidates issue.
    DpmjetIII193
    ]

gevt = 10

variables = ["pid", "eta", "charge", "n_wounded", "xf", "xlab"]

def parse():

    parser = argparse.ArgumentParser(description="main controller")
    parser.add_argument("-g", "--gen", help="generate data")
    parser.add_argument("-a", "--analyse", type = str, help="variable to analyse")

    args = parser.parse_args()
    return args

def gen(label):
    print("🚀 Start simulation...")
    for model in models:
        run_model(kin_list=kins[label], 
                  model=model, 
                  gevt=gevt, 
                  variables=variables
                  )

def analyse(var):
    # 繪圖規格定義 - 新增 y 軸範圍設定
    PlotSpec = namedtuple("PlotSpec", ["df_dict", "title", "pid", "col_name", "main_ylim", "ratio_ylim", "range",
                                       "x_label", "y_label"])
    col_name = var
    main_ylim = None
    ratio_ylim = (0.0, 1.5)
    if var == "eta":
        range = (-10, 10)
    elif var == "xf":
        range = (-0.02, 0.02)
    elif var == "xlab":
        range = (-0.001, 0.005)
    else:
        range = None

    x_label = "var"
    y_label = "Candidates"
    
    # 定義每個圖的 y 軸範圍（可根據需要調整）
    df_dict : dict[str, dict[str, pl.DataFrame]] = load_parquet_by_prefix_and_model("pq")

    # print(df_dict)
    plot_specs = [
        # Charged
        PlotSpec(df_dict['pp'], r"charged : pp", None, col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        PlotSpec(df_dict['pO'], r"charged : pO", None, col_name, main_ylim, ratio_ylim, range, x_label, y_label),

        # π⁰
        PlotSpec(df_dict['pp'], r"$\pi^0$ : pp", int(lp.pi_0.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        PlotSpec(df_dict['pO'], r"$\pi^0$ : pO", int(lp.pi_0.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        
        # π⁺
        PlotSpec(df_dict['pp'], r"$\pi^+$ : pp", int(lp.pi_plus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        PlotSpec(df_dict['pO'], r"$\pi^+$ : pO", int(lp.pi_plus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        
        # π⁻
        PlotSpec(df_dict['pp'], r"$\pi^-$ : pp", int(lp.pi_minus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        PlotSpec(df_dict['pO'], r"$\pi^-$ : pO", int(lp.pi_minus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        
        # K⁺
        PlotSpec(df_dict['pp'], r"$K^+$ : pp", int(lp.K_plus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        PlotSpec(df_dict['pO'], r"$K^+$ : pO", int(lp.K_plus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        
        # K⁻
        PlotSpec(df_dict['pp'], r"$K^-$ : pp", int(lp.K_minus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
        PlotSpec(df_dict['pO'], r"$K^-$ : pO", int(lp.K_minus.pdgid), col_name, main_ylim, ratio_ylim, range, x_label, y_label),
    ]

    main(plot_specs, gevt = gevt, output_path_name = str(var))


if __name__ == "__main__":
    args = parse()
    
    if args.gen == "pp":
        gen(0)
    if args.gen == "pO":
        gen(1)
    if args.gen == "Op":
        gen(2)
    if args.gen == "OO":
        gen(3)

    if args.analyse:
        analyse(args.analyse)

