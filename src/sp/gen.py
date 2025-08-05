import chromo
import numpy as np
from chromo.kinematics import CenterOfMass
from chromo.models import EposLHCR
from chromo.constants import TeV, MeV
from pathlib import Path

from sp import root_base

def run_model(kin_list   : list [CenterOfMass, str] = [CenterOfMass(5 * TeV, "p", (16, 8)), "pO"],
              model : chromo.models = EposLHCR,
              gevt  : int = 100,
              ):
    import polars as pl
    import numpy as np
    import time

    # Single-process batch optimized version
    
    # Initialize generator once
    kin = kin_list[0]
    gen = model(kin)
    gen.set_stable(111) 
    
    total_events = gevt
    batch_size = total_events / 40  # Larger batch size for better performance
    
    # Pre-allocate result containers
    all_pid_chunks = []
    all_charged_chunks = []
    all_eta_chunks = []
    all_mass_chunks = []
    all_n_wounded_chunks = []

    events_processed = 0
    current_batch_pids = []
    current_batch_charged = []
    current_batch_eta = []
    current_batch_mass = []
    current_batch_n_wounded = []
    
    print(f"Start processing {total_events:,} events...")
    
    start_time = time.time()
    for i, event in enumerate(gen(total_events)):
        # Process each event
        f = event.final_state()

        n_candidates = f.pid.size
        n_wounded = f.n_wounded[1]
        wounded_array = np.full(n_candidates, n_wounded)

        current_batch_pids.extend(f.pid)
        current_batch_eta.extend(f.eta)
        current_batch_charged.extend(f.charge)
        current_batch_mass.extend(f.m)
        current_batch_n_wounded.extend(wounded_array)
        events_processed += 1
        
        # Process current batch
        if events_processed % batch_size == 0:
            # Convert to numpy arrays for performance
            pid_chunk = np.array(current_batch_pids, dtype=np.int32)
            eta_chunk = np.array(current_batch_eta, dtype=np.float32)
            charged_chunk = np.array(current_batch_charged, dtype=np.int32)
            mass_chunk = np.array(current_batch_mass, dtype=np.float32)
            n_wounded_chunk  = np.array(current_batch_n_wounded, dtype=np.int32)

            all_pid_chunks.append(pid_chunk)
            all_eta_chunks.append(eta_chunk)
            all_charged_chunks.append(charged_chunk)    
            all_mass_chunks.append(mass_chunk)
            all_n_wounded_chunks.append(n_wounded_chunk)

            current_batch_pids = []  # Clear current batch
            current_batch_eta = []
            current_batch_charged = []
            current_batch_mass = []
            current_batch_n_wounded = []
            
            # Progress report
            elapsed = time.time() - start_time
            rate = events_processed / elapsed
            eta = (total_events - events_processed) / rate
            print(f"Processed: {events_processed:,}/{total_events:,} "
                f"({events_processed/total_events*100:.1f}%) "
                f"Rate: {rate:.0f} events/s, ETA: {eta/60:.1f}min")
    
    # Handle remaining events
    if current_batch_pids:
        pid_chunk = np.array(current_batch_pids, dtype=np.int32)
        all_pid_chunks.append(pid_chunk)
    
    if current_batch_eta:
        eta_chunk = np.array(current_batch_eta, dtype=np.float32)
        all_eta_chunks.append(eta_chunk)
    
    if current_batch_charged:
        charged_chunk = np.array(current_batch_charged, dtype=np.int32)
        all_charged_chunks.append(charged_chunk)
    
    if current_batch_mass:
        mass_chunk = np.array(current_batch_mass, dtype=np.float32)
        all_mass_chunks.append(mass_chunk)

    if current_batch_n_wounded:
        n_wounded_chunk = np.array(current_batch_n_wounded, dtype=np.int32)
        all_n_wounded_chunks.append(n_wounded_chunk)

    # Efficiently concatenate all chunks
    print("Merging results...")
    all_pids = np.concatenate(all_pid_chunks) if all_pid_chunks else np.array([], dtype=np.int32)
    all_eta = np.concatenate(all_eta_chunks) if all_eta_chunks else np.array([], dtype=np.float32)
    all_charged = np.concatenate(all_charged_chunks) if all_charged_chunks else np.array([], dtype=np.int32)
    all_mass = np.concatenate(all_mass_chunks) if all_mass_chunks else np.array([], dtype=np.float32)
    all_n_wounded = np.concatenate(all_n_wounded_chunks) if all_n_wounded_chunks else np.array([], dtype=np.int32)
    
    # Create Polars DataFrame
    df = pl.DataFrame({"pid": all_pids, "eta": all_eta, "charged": all_charged, "mass": all_mass, "n_wounded": all_n_wounded})
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print(f"Average rate: {total_events/total_time:.0f} events/s")
    print(f"Total particles generated: {len(df):,}")
    print(df.shape)
    print(df)

    try:
        # 使用 pathlib 構建路徑並確保目錄存在
        output_path = Path(root_base) / "pq" / f"{kin_list[1]}_{gen.label}_{gevt}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 寫入 parquet 檔案
        df.write_parquet(output_path)
        print(f'File saved to: {output_path}')
        
    except Exception as e:
        print(f"Error writing parquet file: {e}")
