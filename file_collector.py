import os

def collect_verilog_files(base_dir):
    verilog_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == ".v":
                verilog_files.append(os.path.join(root, f))
    return sorted(verilog_files)
