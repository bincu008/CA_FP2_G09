import os

# --- Configurations ---
GROUP_ID = "G09"
MODEL_ID = 5
CORES_ASSIGNED = list(range(33, 39))  # Cores 33 to 38 (6 cores)

# Extracted directly from workload_config.json for Model 5
ARCH = [66, 1454, 3586, 699, 4058, 10]

# --- Global Tensor Tracking ---
INPUT_OFFSET = 1615  # (440 + 297 + 93 + 350 + 435)
OUTPUT_OFFSET = 50   # (9 + 13 + 10 + 11 + 7)

def get_xy(core_id):
    """Convert linear core ID to NoC Mesh coordinates."""
    return core_id % 8, core_id // 8

def split_dim(dim, num_cores):
    """Partition a dimension into contiguous chunks, distributing remainder."""
    base = dim // num_cores
    rem = dim % num_cores
    return [base + 1 if i < rem else base for i in range(num_cores)]

def generate_model5_instructions():
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
        w(core, f"DRAM,READ,TOKEN_I,0,0,{INPUT_OFFSET},{ARCH[0]}")
        w(core, f"ARRAY,IS,model.0.weight,{MODEL_ID},{i},{ARCH[0]},{my_chunk}")
        
        w(core, f"DMA,1,0,2,0,{my_chunk}")
        w(core, f"DRAM,READ,MODEL,model.0.bias,3,0,{MODEL_ID},{i},{my_chunk}")
        w(core, f"VECTOR,4,{my_chunk}") 
        w(core, f"DMA,4,0,2,0,{my_chunk}")
        w(core, f"VECTOR,3,{my_chunk}") 
        w(core, f"DMA,4,0,0,0,{my_chunk}") 

    # ==========================================
    # LAYERS 1 to 3: ROW SPLIT (Reduce-Scatter)
    # ==========================================
    in_chunks = out_chunks
    for L in range(1, 4):
        out_len = ARCH[L+1]
        out_chunks = split_dim(out_len, len(CORES_ASSIGNED))
        offsets = [sum(out_chunks[:j]) for j in range(len(CORES_ASSIGNED))]
        
        for i, core in enumerate(CORES_ASSIGNED):
            my_chunk = out_chunks[i]
            my_offset = offsets[i]
            
            w(core, f"\n# === Layer {L}: Row Split + Reduce Scatter ===")
            w(core, f"ARRAY,IS,model.{L*2}.weight,{MODEL_ID},{i},{in_chunks[i]},{out_len}")
            
            if my_chunk > 0:
                w(core, f"DMA,1,{my_offset},2,0,{my_chunk}")
            
            # SCATTER: Only send if the target chunk length is > 0
            for j in range(len(CORES_ASSIGNED)):
                if j != i and out_chunks[j] > 0:
                    target_core = CORES_ASSIGNED[j]
                    tx, ty = get_xy(target_core)
                    w(core, f"NoC,1,{offsets[j]},{tx},{ty},3,0,{out_chunks[j]}")
            
            # REDUCE & BIAS: Only if this core has a valid chunk
            if my_chunk > 0:
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
    # LAYER 4: ROW SPLIT -> GATHER -> SOFTMAX
    # ==========================================
    L = 4
    out_len = ARCH[L+1] # 10 Classes
    out_chunks = split_dim(out_len, len(CORES_ASSIGNED))
    offsets = [sum(out_chunks[:j]) for j in range(len(CORES_ASSIGNED))]

    for i, core in enumerate(CORES_ASSIGNED):
        my_chunk = out_chunks[i]
        my_offset = offsets[i]
        
        w(core, f"\n# === Layer {L}: Row Split + Reduce Scatter ===")
        w(core, f"ARRAY,IS,model.{L*2}.weight,{MODEL_ID},{i},{in_chunks[i]},{out_len}")
        
        if my_chunk > 0:
            w(core, f"DMA,1,{my_offset},2,0,{my_chunk}")
        
        # SCATTER
        for j in range(len(CORES_ASSIGNED)):
            if j != i and out_chunks[j] > 0:
                target_core = CORES_ASSIGNED[j]
                tx, ty = get_xy(target_core)
                w(core, f"NoC,1,{offsets[j]},{tx},{ty},3,0,{out_chunks[j]}")
        
        # REDUCE & BIAS
        if my_chunk > 0:
            for _ in range(len(CORES_ASSIGNED) - 1):
                w(core, f"NoC,IN")
                w(core, f"VECTOR,4,{my_chunk}")
                w(core, f"DMA,4,0,2,0,{my_chunk}")
                
            w(core, f"DRAM,READ,MODEL,model.{L*2}.bias,3,0,{MODEL_ID},{i},{my_chunk}")
            w(core, f"VECTOR,4,{my_chunk}") 
        
        w(core, f"\n# === Gather to Core {CORES_ASSIGNED[0]} for Softmax ===")
        if core != CORES_ASSIGNED[0]:
            if my_chunk > 0:
                tx, ty = get_xy(CORES_ASSIGNED[0])
                w(core, f"NoC,4,0,{tx},{ty},2,{my_offset},{my_chunk}")
        else:
            if my_chunk > 0:
                w(core, f"DMA,4,0,2,{my_offset},{my_chunk}")
            
            # Wait for incoming pieces ONLY from cores that actually have chunks > 0
            num_incoming = sum(1 for idx, c in enumerate(out_chunks) if idx != 0 and c > 0)
            for _ in range(num_incoming):
                w(core, f"NoC,IN")
                
            w(core, f"VECTOR,2,{out_len}") 
            w(core, f"DRAM,WRITE,TOKEN_O,4,0,{OUTPUT_OFFSET},{out_len}")

    for f in files.values():
        f.close()
        
    print(f"Generated instructions for Cores {CORES_ASSIGNED[0]}-{CORES_ASSIGNED[-1]} (Model {MODEL_ID}).")

if __name__ == "__main__":
    generate_model5_instructions()