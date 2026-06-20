import os

# --- Configurations ---
GROUP_ID = "G09"
MODEL_ID = 0
CORES_ASSIGNED = [0, 1, 2, 3, 4, 5]
# Extracted directly from workload_config.json
ARCH = [440, 332, 3475, 768, 3766, 729, 294, 912, 9]

def get_xy(core_id):
    """Convert linear core ID to NoC Mesh coordinates."""
    return core_id % 8, core_id // 8

def split_dim(dim, num_cores):
    """Partition a dimension into contiguous chunks, distributing remainder."""
    base = dim // num_cores
    rem = dim % num_cores
    return [base + 1 if i < rem else base for i in range(num_cores)]

def generate_model0_instructions():
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
        #w(core, f"DRAM,READ,TOKEN_I,7,0,0,{ARCH[0]}")
        #w(core, f"DMA,7,0,0,0,{ARCH[0]}")
        # DRAM, READ, TOKEN_I, buf, buf_offset, token_io_offset, len 
        w(core, f"DRAM,READ,TOKEN_I,0,0,0,{ARCH[0]}")

        w(core, f"ARRAY,IS,model.0.weight,{MODEL_ID},{i},{ARCH[0]},{my_chunk}")
        
        w(core, f"DMA,1,0,2,0,{my_chunk}")
        w(core, f"DRAM,READ,MODEL,model.0.bias,3,0,0,{i},{my_chunk}")
        w(core, f"VECTOR,4,{my_chunk}") # ADD
        w(core, f"DMA,4,0,2,0,{my_chunk}")
        w(core, f"VECTOR,3,{my_chunk}") # SiLU
        w(core, f"DMA,4,0,0,0,{my_chunk}") # Move to Array Input for L1

    # ==========================================
    # LAYERS 1 to 6: ROW SPLIT (Reduce-Scatter)
    # ==========================================
    in_chunks = out_chunks
    for L in range(1, 7):
        out_len = ARCH[L+1]
        out_chunks = split_dim(out_len, len(CORES_ASSIGNED))
        offsets = [sum(out_chunks[:j]) for j in range(len(CORES_ASSIGNED))]
        
        for i, core in enumerate(CORES_ASSIGNED):
            my_chunk = out_chunks[i]
            my_offset = offsets[i]
            
            w(core, f"\n# === Layer {L}: Row Split + Reduce Scatter ===")
            w(core, f"ARRAY,IS,model.{L*2}.weight,{MODEL_ID},{i},{in_chunks[i]},{out_len}")
            
            # Extract *only* the chunk this core is responsible for finalizing
            w(core, f"DMA,1,{my_offset},2,0,{my_chunk}")
            
            # SCATTER: Send the other chunks directly from Array Out to target cores
            for j in range(len(CORES_ASSIGNED)):
                if j != i:
                    target_core = CORES_ASSIGNED[j]
                    tx, ty = get_xy(target_core)
                    w(core, f"# Send Chunk {j} to Core {target_core} [x:{tx}, y:{ty}]")
                    w(core, f"NoC,1,{offsets[j]},{tx},{ty},3,0,{out_chunks[j]}")
            
            # REDUCE: Wait for incoming packets and ADD them sequentially
            for _ in range(len(CORES_ASSIGNED) - 1):
                w(core, f"NoC,IN")
                w(core, f"VECTOR,4,{my_chunk}") # ADD
                w(core, f"DMA,4,0,2,0,{my_chunk}")
                
            # Bias & SiLU Phase
            w(core, f"DRAM,READ,MODEL,model.{L*2}.bias,3,0,0,{i},{my_chunk}")
            w(core, f"VECTOR,4,{my_chunk}")
            w(core, f"DMA,4,0,2,0,{my_chunk}")
            w(core, f"VECTOR,3,{my_chunk}")
            w(core, f"DMA,4,0,0,0,{my_chunk}") # Feed to next layer
            
        in_chunks = out_chunks

    # ==========================================
    # LAYER 7: ROW SPLIT -> REDUCE-SCATTER -> BIAS -> GATHER -> SOFTMAX
    # ==========================================
    L = 7
    out_len = ARCH[L+1] # 9 Classes
    out_chunks = split_dim(out_len, len(CORES_ASSIGNED))
    offsets = [sum(out_chunks[:j]) for j in range(len(CORES_ASSIGNED))]

    for i, core in enumerate(CORES_ASSIGNED):
        my_chunk = out_chunks[i]
        my_offset = offsets[i]
        
        w(core, f"\n# === Layer 7: Row Split + Reduce Scatter ===")
        w(core, f"ARRAY,IS,model.{L*2}.weight,{MODEL_ID},{i},{in_chunks[i]},{out_len}")
        w(core, f"DMA,1,{my_offset},2,0,{my_chunk}")
        
        # SCATTER
        for j in range(len(CORES_ASSIGNED)):
            if j != i:
                target_core = CORES_ASSIGNED[j]
                tx, ty = get_xy(target_core)
                w(core, f"NoC,1,{offsets[j]},{tx},{ty},3,0,{out_chunks[j]}")
        
        # REDUCE
        for _ in range(len(CORES_ASSIGNED) - 1):
            w(core, f"NoC,IN")
            w(core, f"VECTOR,4,{my_chunk}")
            w(core, f"DMA,4,0,2,0,{my_chunk}")
            
        # BIAS PHASE (Read exact slice to prevent out-of-bounds error)
        w(core, f"DRAM,READ,MODEL,model.{L*2}.bias,3,0,0,{i},{my_chunk}")
        w(core, f"VECTOR,4,{my_chunk}") 
        # Resulting local chunk is now residing in Bank 4
        
        w(core, f"\n# === Gather to Core 0 for Softmax ===")
        if core != CORES_ASSIGNED[0]:
            # Cores 1-5 send their chunk from Bank 4 directly to Core 0's Bank 2 at the correct offset
            tx, ty = get_xy(CORES_ASSIGNED[0])
            w(core, f"NoC,4,0,{tx},{ty},2,{my_offset},{my_chunk}")
        else:
            # Core 0 moves its own chunk into Bank 2 at offset 0
            w(core, f"DMA,4,0,2,{my_offset},{my_chunk}")
            
            # Core 0 waits to receive the 5 incoming packets
            for _ in range(len(CORES_ASSIGNED) - 1):
                w(core, f"NoC,IN")
                
            # Now Bank 2 contains the fully assembled 9-element Linear output.
            w(core, f"VECTOR,2,{out_len}") # Run Softmax!
            
            # Write all 9 final probabilities to DRAM
            w(core, f"DRAM,WRITE,TOKEN_O,4,0,0,{out_len}")

    # Close active file handlers
    for f in files.values():
        f.close()
                
    print("Generated instruction files for model 0 successfully.")

if __name__ == "__main__":
    generate_model0_instructions()