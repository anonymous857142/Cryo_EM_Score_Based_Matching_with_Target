import os
import csv
import random


root_dir = "data"
domains = [
    "10291_particles_stack_snr_0005",
]
src_dirs = [os.path.join(root_dir, d) for d in domains]

domain_clean_5A = "clean_images_10291_5A"
src_dir_clean = os.path.join(root_dir, domain_clean_5A)

domain_test = "10291_micrographs"
src_dir_test = os.path.join(root_dir, domain_test)

out_dir = root_dir
os.makedirs(out_dir, exist_ok=True)

# Collect all .mrc files from all source dirs
files = []
for d in src_dirs:
    for f in os.listdir(d):
        if f.endswith(".mrc"):
            files.append(os.path.join(d, f))  # store full paths!
files = sorted(files)

random.shuffle(files) 

clean_files = []
for d in [domain_clean_5A]:
    clean_files.extend([f"{d}/{f}" for f in os.listdir(os.path.join(root_dir, d)) if f.endswith(".mrc")])
test_files = [f for f in os.listdir(src_dir_test) if f.endswith(".mrc") or f.endswith(".tif") or f.endswith(".tiff")]

n_total = len(files)
n_train = int(0.8 * n_total)
n_val = int(0.2 * n_total)

train_files = files[:n_train]
val_files = files[n_train:n_train+n_val]

# Save CSV files for train and val splits, with paths relative to root_dir
def save_csv(file_list, split):
    csv_path = os.path.join(out_dir, f"{split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])  # single column header
        for full_path in file_list:
            # keep path relative to root_dir for portability
            rel_path = os.path.relpath(full_path, root_dir)
            writer.writerow([rel_path])
    print(f"{csv_path} 已保存，共 {len(file_list)} 条")

save_csv(train_files, "train_10291")
save_csv(val_files, "val_10291")

def save_csv_clean(file_list, clean):
    csv_path = os.path.join(out_dir, "clean_10291_5A.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        for path in file_list:
            writer.writerow([path])
    print(f"{csv_path} 已保存，共 {len(file_list)} 条")

save_csv_clean(clean_files, "clean")

def save_csv_test(file_list):
    csv_path = os.path.join(out_dir, "test_10291.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([domain_test])  # 列名改为 micrograph 域
        for name in file_list:
            writer.writerow([f"{domain_test}/{name}"])
    print(f"{csv_path} 已保存，共 {len(file_list)} 条")

save_csv_test(test_files)

"""
domain_tp = "10059_particles_stack_tp"
domain_fp = "10059_particles_stack_fp"
src_dir_tp = os.path.join(root_dir, domain_tp)
src_dir_fp = os.path.join(root_dir, domain_fp)
tp_files = [f for f in os.listdir(src_dir_tp) if f.endswith(".mrcs")]
fp_files = [f for f in os.listdir(src_dir_fp) if f.endswith(".mrcs")]

def save_csv_general(file_list, csv_name, domain_name):
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([domain_name])
        for name in file_list:
            writer.writerow([f"{domain_name}/{name}"])
    print(f"{csv_path} 已保存，共 {len(file_list)} 条")

# save separate csv for FP and TP
save_csv_general(tp_files, "tp.csv", domain_tp)
save_csv_general(fp_files, "fp.csv", domain_fp)
"""

#print(f"总数: {n_total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}, clean={len(clean_files)}, tp={len(tp_files)}, fp={len(fp_files)}")
print(f"总数: {n_total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}, clean={len(clean_files)}")
