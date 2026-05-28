import os
import cv2
import pandas as pd
import numpy as np
import re
from ultralytics import YOLO
from tadpole_core import calculate_morphology, paint_measured_biology

def deep_batch_analyze():
    MODEL_PATH = "best.pt"
    PARENT_DIR = "Data_to_Analyze"
    OUTPUT_PROJECT = "Analysis_Results"
    
    print("🧠 Loading Tadpole AI...")
    model = YOLO(MODEL_PATH)
    model.model.names = {0: 'Eye', 1: 'Head'}

    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    master_results_data = []
    MM_CONVERSION = 1.0 

    for root, dirs, files in os.walk(PARENT_DIR):
        has_images = any(f.lower().endswith(VALID_EXTENSIONS) for f in files)
        
        if has_images:
            folder_name = os.path.basename(root)
            parent_folder_name = os.path.basename(os.path.dirname(root))
            output_folder_name = f"{parent_folder_name}_{folder_name}"
            
            print(f"\n--- 🔍 Found images! Processing folder: {output_folder_name} ---")
            
            results_list = model.predict(
                source=root,
                save=False, 
                save_txt=True,
                conf=0.25,
                device='mps',
                retina_masks=True,
                project=OUTPUT_PROJECT, 
                name=output_folder_name,
                exist_ok=True,
                verbose=False 
            )
            
            output_dir = os.path.join(OUTPUT_PROJECT, output_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, res in enumerate(results_list, 1):
                filename = os.path.basename(res.path)
                print(f"  [{idx}/{len(results_list)}] Extracting data for: {filename}")
                
                stats, flag, final_head, valid_eyes = calculate_morphology(res, pixel_to_mm_ratio=MM_CONVERSION)
                painted_img = paint_measured_biology(res, final_head, valid_eyes)
                cv2.imwrite(os.path.join(output_dir, filename), painted_img)
                
                stats["Filename"] = filename
                stats["Output Folder"] = output_folder_name
                stats["Status Flag"] = flag
                
                master_results_data.append(stats)

    # ==========================================================
    # DATA EXPORT & STATISTICAL SUMMARY
    # ==========================================================
    if master_results_data:
        print("\n✅ All nested folders processed successfully! Compiling data...")
        df = pd.DataFrame(master_results_data)
        
        # Pull Filename to the front, push Status Flag to the back!
        cols = df.columns.tolist()
        if "Filename" in cols:
            cols.remove("Filename")
            cols.insert(0, "Filename")
        if "Status Flag" in cols:
            cols.remove("Status Flag")
            cols.append("Status Flag") 
        df = df[cols]
        
        raw_csv_name = "Objective_1_Morphology_Results.csv"
        df.to_csv(raw_csv_name, index=False)
        print(f"📄 Raw Master Data saved to: {raw_csv_name}")
        
        print("📊 Generating Statistical Folder Summaries...")
        summary_rows = []
        
        for folder, group in df.groupby("Output Folder"):
            clean_group = group[group["Status Flag"] == "OK"]
            error_group = group[group["Status Flag"] != "OK"]
            
            n_total = len(group)
            n_ok = len(clean_group)
            
            failed_imgs = [f"{row['Filename']} ({row['Status Flag']})" for _, row in error_group.iterrows()]
            
            if n_ok > 0:
                mean_head = clean_group["Head Area"].mean()
                std_head = clean_group["Head Area"].std()
                
                mean_eye = clean_group["Interpupillary Distance"].mean()
                std_eye = clean_group["Interpupillary Distance"].std()
                
                mean_total_eye = clean_group["Total Eye Area"].mean()
                std_total_eye = clean_group["Total Eye Area"].std()
                
                mean_ratio = clean_group["Interpupillary Distance Ratio"].mean()
                mean_sym_ratio = clean_group["Orbital Symmetry Ratio"].mean()
                mean_asym_diff = clean_group["Orbital Asymmetry % Difference"].mean()
                
                outliers = []
                for _, row in clean_group.iterrows():
                    z_head = abs((row["Head Area"] - mean_head) / std_head) if pd.notna(std_head) and std_head > 0 else 0
                    z_eye = abs((row["Interpupillary Distance"] - mean_eye) / std_eye) if pd.notna(std_eye) and std_eye > 0 else 0
                    
                    if z_head > 2.0 or z_eye > 2.0:
                        outliers.append(f"{row['Filename']} (Stat Outlier)")
                        
                all_flags = failed_imgs + outliers
                flag_str = ", ".join(all_flags) if all_flags else "None"
                
                summary_rows.append({
                    "Experimental Folder": folder,
                    "Total Images": n_total,
                    "Successful Scans": n_ok,
                    "Mean Head Area (px²)": round(mean_head, 2),
                    "CV% Head Area": round((std_head/mean_head)*100, 1) if std_head else 0.0,
                    "Mean Interpupillary Dist (px)": round(mean_eye, 2),
                    "CV% Interpupillary Dist": round((std_eye/mean_eye)*100, 1) if std_eye else 0.0,
                    "Mean Total Eye Area (px²)": round(mean_total_eye, 2),
                    "CV% Total Eye Area": round((std_total_eye/mean_total_eye)*100, 1) if std_total_eye else 0.0,
                    "Mean Interpupillary Dist Ratio": round(mean_ratio, 2),
                    "Mean Orbital Sym Ratio": round(mean_sym_ratio, 3),
                    "Mean Orbital Asym % Diff": round(mean_asym_diff, 2),
                    "Flags & Outliers (Check these images)": flag_str
                })
            else:
                flag_str = ", ".join(failed_imgs) if failed_imgs else "Unknown Error"
                
                summary_rows.append({
                    "Experimental Folder": folder,
                    "Total Images": n_total,
                    "Successful Scans": 0,
                    "Mean Head Area (px²)": np.nan,
                    "CV% Head Area": np.nan,
                    "Mean Interpupillary Dist (px)": np.nan,
                    "CV% Interpupillary Dist": np.nan,
                    "Mean Total Eye Area (px²)": np.nan,
                    "CV% Total Eye Area": np.nan,
                    "Mean Interpupillary Dist Ratio": np.nan,
                    "Mean Orbital Sym Ratio": np.nan,
                    "Mean Orbital Asym % Diff": np.nan,
                    "Flags & Outliers (Check these images)": flag_str
                })
                
        summary_df = pd.DataFrame(summary_rows)
        
        summary_df = summary_df.sort_values(
            by="Experimental Folder",
            key=lambda col: col.map(lambda s: tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))))
        )
        
        summary_csv_name = "Objective_1_Folder_Summary.csv"
        summary_df.to_csv(summary_csv_name, index=False)
        print(f"📄 Statistical Summary saved to: {summary_csv_name}")
        
        print(f"🎨 Clean painted images saved in: {OUTPUT_PROJECT}")
        print(f"🐸 Total images processed: {len(master_results_data)}")
    else:
        print("\n⚠️ No images were found to process.")

if __name__ == '__main__':
    deep_batch_analyze()