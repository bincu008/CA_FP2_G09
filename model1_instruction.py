import os

# --- Configurations ---
GROUP_ID = "G09"
MODEL_ID = 1
CORES_ASSIGNED = [6, 7]
# Extracted directly from workload_config.json for Model 1
ARCH = [297, 2975, 793, 372, 13]

# --- Global Tensor Tracking ---
INPUT_OFFSET = 440  # Model 0 took elements 0-439
OUTPUT_OFFSET = 9   # Model 0 took outputs 0-8

def get_xy(core_id):
    return core_id % 8, core_id // 8

def split_dim(dim, num_cores):
    base = dim // num_cores
    rem = dim % num_cores
    return [base + 1 if i < rem else base for i in range(num_cores)]

def generate_model1_instructions():
    files = {c: open(f"core_{c}_{GROUP_ID}.csv", "w") for c in CORES_ASSIGNED}
    
    def w(core, text):
        files[core].write(text + "\n")

    # ==========================================
    # LAYER 0: COLUMN SPLIT
    # ==========================================
    out_chunks = split_dim(ARCH[1], len(CORES_ASSIGNED))
    
    for i, core in enumerate(CORES_ASSIGNED):
        my_chunk = out_chunks[i]
        w(core, f"# === Layer 0: Column Split ===")
        # FIXED: Reading from INPUT_OFFSET instead of 0
        w(core, f"DRAM,READ,TOKEN_I,0,0,{INPUT_OFFSET},{ARCH[0]}")
        w(core, f"ARRAY,IS,model.0.weight,{MODEL_ID},{i},{ARCH[0]},{my_chunk}")
        
        w(core, f"DMA,1,0,2,0,{my_chunk}")
        w(core, f"DRAM,READ,MODEL,model.0.bias,3,0,{MODEL_ID},{i},{my_chunk}")
        w(core, f"VECTOR,4,{my_chunk}")
        w(core, f"DMA,4,0,2,0,{my_chunk}")
        w(core, f"VECTOR,3,{my_chunk}")
        w(core, f"DMA,4,0,0,0,{my_chunk}")

    # ==========================================
    # LAYERS 1 to 2: ROW SPLIT (Reduce-Scatter)
    # ==========================================
    in_chunks = out_chunks
    for L in range(1, 3):
        out_len = ARCH[L+1]
        out_chunks = split_dim(out_len, len(CORES_ASSIGNED))
        offsets = [sum(out_chunks[:j]) for j in range(len(CORES_ASSIGNED))]
        
        for i, core in enumerate(CORES_ASSIGNED):
            my_chunk = out_chunks[i]
            my_offset = offsets[i]
            
            w(core, f"\n# === Layer {L}: Row Split + Reduce Scatter ===")
            w(core, f"ARRAY,IS,model.{L*2}.weight,{MODEL_ID},{i},{in_chunks[i]},{out_len}")
            w(core, f"DMA,1,{my_offset},2,0,{my_chunk}")
            
            for j in range(len(CORES_ASSIGNED)):
                if j != i:
                    target_core = CORES_ASSIGNED[j]
                    tx, ty = get_xy(target_core)
                    w(core, f"NoC,1,{offsets[j]},{tx},{ty},3,0,{out_chunks[j]}")
            
            for _ in range(len(CORES_ASSIGNED) - 1):
                w(core, f"NoC,IN")
                w(core, f"VECTOR,4,{my_chunk}")
                w(core, f"DMA,4,0,2,0,{my_chunk}")
                
            w(core, f"DRAM,READ,MODEL,model.{L*2}.bias,3,0,{MODEL_ID},{i},{my_chunk}")
            w(core, f"VECTOR,4,{my_chunk}")
            w(core, f"DMA,4,0,2,0,{my_chunk}")
            w(core, f"VECTOR,3,{my_chunk}")
            w(core, f"DMA,4,0,0,0,{my_chunk}")
            
        in_chunks = out_chunks

    # ==========================================
    # LAYER 3: ROW SPLIT -> GATHER -> SOFTMAX
    # ==========================================
    L = 3
    out_len = ARCH[L+1] # 13 Classes
    out_chunks = split_dim(out_len, len(CORES_ASSIGNED))
    offsets = [sum(out_chunks[:j]) for j in range(len(CORES_ASSIGNED))]

    for i, core in enumerate(CORES_ASSIGNED):
        my_chunk = out_chunks[i]
        my_offset = offsets[i]
        
        w(core, f"\n# === Layer 3: Row Split + Reduce Scatter ===")
        w(core, f"ARRAY,IS,model.{L*2}.weight,{MODEL_ID},{i},{in_chunks[i]},{out_len}")
        w(core, f"DMA,1,{my_offset},2,0,{my_chunk}")
        
        for j in range(len(CORES_ASSIGNED)):
            if j != i:
                target_core = CORES_ASSIGNED[j]
                tx, ty = get_xy(target_core)
                w(core, f"NoC,1,{offsets[j]},{tx},{ty},3,0,{out_chunks[j]}")
        
        for _ in range(len(CORES_ASSIGNED) - 1):
            w(core, f"NoC,IN")
            w(core, f"VECTOR,4,{my_chunk}")
            w(core, f"DMA,4,0,2,0,{my_chunk}")
            
        w(core, f"DRAM,READ,MODEL,model.{L*2}.bias,3,0,{MODEL_ID},{i},{my_chunk}")
        w(core, f"VECTOR,4,{my_chunk}") 
        
        w(core, f"\n# === Gather to Core {CORES_ASSIGNED[0]} for Softmax ===")
        if core != CORES_ASSIGNED[0]:
            tx, ty = get_xy(CORES_ASSIGNED[0])
            w(core, f"NoC,4,0,{tx},{ty},2,{my_offset},{my_chunk}")
        else:
            w(core, f"DMA,4,0,2,{my_offset},{my_chunk}")
            for _ in range(len(CORES_ASSIGNED) - 1):
                w(core, f"NoC,IN")
                
            w(core, f"VECTOR,2,{out_len}") 
            w(core, f"DRAM,WRITE,TOKEN_O,4,0,{OUTPUT_OFFSET},{out_len}")

    for f in files.values():
        f.close()
        
    print(f"Generated instructions for Cores {CORES_ASSIGNED[0]}-{CORES_ASSIGNED[-1]} (Model {MODEL_ID}).")

if __name__ == "__main__":
    generate_model1_instructions()