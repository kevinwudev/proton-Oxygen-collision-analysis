import chromo
import numpy as np
from chromo.kinematics import CenterOfMass
from chromo.models import EposLHCR
from chromo.constants import TeV
from pathlib import Path

from sp import root_base

def run_model(kin_list   : list [CenterOfMass, str] = [CenterOfMass(9.9 * TeV, "p", (16, 8)), "pO"],
              model : chromo.models = EposLHCR,
              gevt  : int = 100,
              variables: list[str] = ["pid", "eta", "charge", "n_wounded", "xf", "xlab"]
              ):
    import polars as pl
    import numpy as np
    import time

    """單進程批次優化版本 - 支援動態變量選擇"""
    
    # 初始化一次生成器
    kin = kin_list[0]
    gen = model(kin)
    gen.set_stable(111) 
    
    total_events = gevt
    batch_size = total_events / 40 # 較大的批次提高效率
    
    # 動態初始化變量容器
    all_chunks = {var: [] for var in variables}
    current_batch = {var: [] for var in variables}
    
    # 數據類型映射 - 根據常見屬性推斷
    dtype_mapping = {
        'pid': np.int32,
        'charge': np.int32, 
        'eta': np.float32,
        'm': np.float32,
        'n_wounded': np.int32
    }
    
    def get_dtype(var_name):
        """自動推斷數據類型"""
        if var_name in dtype_mapping:
            return dtype_mapping[var_name]
        # 默認推斷邏輯
        if var_name.endswith('_id') or var_name == 'pid' or 'charge' in var_name:
            return np.int32
        return np.float32  # 物理量通常是浮點數
    
    events_processed = 0
    
    print(f"開始處理 {total_events:,} 個事件...")
    print(f"收集變量: {variables}")
    
    start_time = time.time()
    for i, event in enumerate(gen(total_events)):
        # 處理事件
        f = event.final_state()
        n_candidates = f.pid.size
        
        # 動態收集指定變量
        for var in variables:
            if var == "n_wounded":
                # 特殊處理 n_wounded
                n_wounded = f.n_wounded[1]
                wounded_array = np.full(n_candidates, n_wounded)
                current_batch[var].extend(wounded_array)
            else:
                # 一般屬性直接獲取
                attr_value = getattr(f, var)
                current_batch[var].extend(attr_value)
        
        events_processed += 1
        
        # 批次處理完成
        if events_processed % batch_size == 0:
            # 將當前批次轉換為 numpy array 並加入總容器
            for var in variables:
                dtype = get_dtype(var)
                chunk = np.array(current_batch[var], dtype=dtype)
                all_chunks[var].append(chunk)
                current_batch[var] = []  # 清空當前批次
            
            # 進度報告
            elapsed = time.time() - start_time
            rate = events_processed / elapsed
            eta = (total_events - events_processed) / rate
            print(f"已處理: {events_processed:,}/{total_events:,} "
                f"({events_processed/total_events*100:.1f}%) "
                f"速率: {rate:.0f} events/s, ETA: {eta/60:.1f}min")
    
    # 處理剩餘事件
    for var in variables:
        if current_batch[var]:
            dtype = get_dtype(var)
            chunk = np.array(current_batch[var], dtype=dtype)
            all_chunks[var].append(chunk)
    
    # 高效合併所有chunks
    print("合併結果...")
    final_data = {}
    for var in variables:
        if all_chunks[var]:
            final_data[var] = np.concatenate(all_chunks[var])
        else:
            dtype = get_dtype(var)
            final_data[var] = np.array([], dtype=dtype)
    
    # 創建 DataFrame
    df = pl.DataFrame(final_data)
    
    total_time = time.time() - start_time
    print(f"總處理時間: {total_time/60:.2f} 分鐘")
    print(f"平均速率: {total_events/total_time:.0f} events/s")
    print(f"生成粒子數: {len(df):,}")
    print(df.shape)
    print(df)

    try:
        output_path = Path(root_base) / "pq" / f"{kin_list[1]}_{gen.label}_{gevt}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(output_path)
        print(f'File saved to: {output_path}')
        
    except Exception as e:
        print(f"Error writing parquet file: {e}")
