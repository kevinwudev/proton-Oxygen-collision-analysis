from sp.gen import run_model
from chromo.kinematics import CenterOfMass
from chromo.models import (DpmjetIII191, DpmjetIII193, DpmjetIII307, EposLHC, EposLHCR, QGSJetIII, Sibyll23e)
from chromo.constants import TeV, MeV


if __name__ == "__main__":

    kins = [
        [CenterOfMass(5 * TeV, "p", (16, 8)), 'pO'],
        [CenterOfMass(5 * TeV, "p", "p"), 'pp'],
    ]

    models = [
        EposLHCR,  # 30 min for 50K events , 20846 candidates of 100 events
        QGSJetIII, # 71 min for 40K events , 23072 candidates of 100 events
        Sibyll23e, # 0.49 min for 40K events , 21235 candidates of 100 events
        DpmjetIII193 # 1.37 min for 40K events ,  21868 candidates of 100 events
        ]
    
    gevt = 100

    print("🚀 開始執行物理模型模擬...")
    for model in models:
        run_model(kin_list=kins[0], model=model, gevt=gevt)

