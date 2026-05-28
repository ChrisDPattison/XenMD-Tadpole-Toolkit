import streamlit as st
import pandas as pd
import zipfile
import io
from PIL import Image
from ultralytics import YOLO
import plotly.express as px

from tadpole_core import calculate_morphology, paint_measured_biology

@st.cache_resource
def load_model():
    MODEL_PATH = "best.pt" 
    model = YOLO(MODEL_PATH)
    model.model.names = {0: 'Eye', 1: 'Head'} 
    return model

def load_crisp_logo(image_path, target_height=120):
    img = Image.open(image_path)
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)
    return img.resize((new_width, target_height), Image.LANCZOS)

st.set_page_config(page_title="XenMD: TadpoleToolKit", layout="wide", page_icon="🐸")

col1, col2, col3 = st.columns([1.5, 1.5, 4])
with col1:
    st.image(load_crisp_logo("XenMD.png", target_height=120)) 
with col2:
    st.image(load_crisp_logo("icg_logo.png", target_height=120)) 

st.title("XenMD: TadpoleToolKit")
st.write("Upload your tadpole microscopy images to automatically detect heads and eyes, measure traits, and generate instant analytics.")

st.success("""
**About this model:** Machine learning model trained to segment the head and eyes of *Xenopus tropicalis* tadpoles, and will return the head size, interpupillary distance, and eye sizes for each tadpole.
""")

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.raw_results_data = [] 
    st.session_state.zip_data = None
    st.session_state.image_dictionary = {} 

st.write("### Upload Image Batches")
up_col1, up_col2 = st.columns(2)

with up_col1:
    control_files = st.file_uploader("Upload CONTROLS", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="controls")
with up_col2:
    mutant_files = st.file_uploader("Upload MUTANTS", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="mutants")

all_uploads = []
if control_files:
    for f in control_files:
        all_uploads.append(("Control", f))
if mutant_files:
    for f in mutant_files:
        all_uploads.append(("Mutant", f))

if len(all_uploads) > 0:
    st.success(f"Ready for analysis: {len(control_files) if control_files else 0} Controls and {len(mutant_files) if mutant_files else 0} Mutants.")
    
    if st.button("🚀 Run AI Analysis", type="primary"):
        model = load_model()
        st.session_state.raw_results_data = [] 
        zip_buffer = io.BytesIO()
        image_dict = {} 
        
        my_bar = st.progress(0, text="AI is analyzing your images. Please wait...")
        total_files = len(all_uploads)
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, (condition, file) in enumerate(all_uploads):
                my_bar.progress((idx + 1) / total_files, text=f"Analyzing {file.name} ({idx + 1}/{total_files})")
                
                original_image = Image.open(file).convert("RGB")
                res = model.predict(
                    source=original_image, 
                    conf=0.5, 
                    device='mps', 
                    retina_masks=True, 
                    verbose=False
                )[0]
                
                stats, flag, final_head, valid_eyes = calculate_morphology(res, pixel_to_mm_ratio=1.0)
                
                painted_img_array = paint_measured_biology(res, final_head, valid_eyes) 
                painted_img = Image.fromarray(painted_img_array)
                
                img_byte_arr = io.BytesIO()
                painted_img.save(img_byte_arr, format='JPEG')
                zip_file.writestr(f"analyzed_{file.name}", img_byte_arr.getvalue())
                image_dict[file.name] = {"original": original_image, "painted": painted_img}
                
                st.session_state.raw_results_data.append({
                    "Filename": file.name,
                    "Condition": condition, 
                    "Head Area": stats["Head Area"],
                    "Interpupillary Distance": stats["Interpupillary Distance"],
                    "Left Eye Area": stats["Left Eye Area"],
                    "Right Eye Area": stats["Right Eye Area"],
                    "Total Eye Area": stats["Total Eye Area"],
                    "Orbital Symmetry Ratio": stats["Orbital Symmetry Ratio"],
                    "Orbital Asymmetry % Difference": stats["Orbital Asymmetry % Difference"],
                    "Interpupillary Distance Ratio": stats["Interpupillary Distance Ratio"],
                    "Status Flag": flag 
                })
        
        st.session_state.zip_data = zip_buffer.getvalue()
        st.session_state.image_dictionary = image_dict
        st.session_state.analysis_complete = True
        my_bar.empty() 

if st.session_state.analysis_complete:

    st.markdown("---")
    st.header("🔬 Scale Calibration & Units")
    st.write("Toggle between pixels and real-world units. The table and graphs will update instantly.")
    
    calib_c1, calib_c2, calib_c3 = st.columns(3)
    
    with calib_c1:
        unit_toggle = st.radio("Display Units:", ["Pixels (px)", "Millimeters (mm)"])
        
    with calib_c2:
        ctrl_scale = st.number_input("Control (mm per px)", value=0.00000, step=0.001, format="%.5f", help="Example: 0.005")
        
    with calib_c3:
        mut_scale = st.number_input("Mutant (mm per px)", value=ctrl_scale, step=0.001, format="%.5f")

    use_mm = False
    if unit_toggle == "Millimeters (mm)":
        if ctrl_scale > 0.0 and mut_scale > 0.0:
            use_mm = True
        else:
            st.warning("⚠️ Please enter a conversion scale greater than 0 for both groups to view in Millimeters. Defaulting to Pixels.")
            use_mm = False
            
    dist_unit = "(mm)" if use_mm else "(px)"
    area_unit = "(mm²)" if use_mm else "(px²)"
    
    display_rows = []
    for row in st.session_state.raw_results_data:
        new_row = dict(row) 
        
        if use_mm and (new_row["Status Flag"] == "✅ OK" or new_row["Status Flag"] == "OK"):
            scale = ctrl_scale if new_row["Condition"] == "Control" else mut_scale
            
            new_row["Head Area"] = round(new_row["Head Area"] * (scale ** 2), 4)
            new_row["Left Eye Area"] = round(new_row["Left Eye Area"] * (scale ** 2), 4)
            new_row["Right Eye Area"] = round(new_row["Right Eye Area"] * (scale ** 2), 4)
            new_row["Total Eye Area"] = round(new_row["Total Eye Area"] * (scale ** 2), 4)
            
            new_row["Interpupillary Distance"] = round(new_row["Interpupillary Distance"] * scale, 4)
            
            if new_row["Interpupillary Distance"] > 0:
                new_row["Interpupillary Distance Ratio"] = round(new_row["Head Area"] / new_row["Interpupillary Distance"], 4)
                
        display_rows.append({
            "Filename": new_row["Filename"],
            "Condition": new_row["Condition"],
            f"Head Area {area_unit}": new_row["Head Area"],
            f"Interpupillary Dist {dist_unit}": new_row["Interpupillary Distance"],
            f"Left Eye Area {area_unit}": new_row["Left Eye Area"],
            f"Right Eye Area {area_unit}": new_row["Right Eye Area"],
            f"Total Eye Area {area_unit}": new_row["Total Eye Area"],
            "Orbital Symmetry Ratio": new_row["Orbital Symmetry Ratio"], 
            "Orbital Asymmetry % Diff": new_row["Orbital Asymmetry % Difference"],
            f"Interpupillary Dist Ratio {dist_unit}": new_row["Interpupillary Distance Ratio"],
            "Status Flag": new_row["Status Flag"]
        })

    display_df = pd.DataFrame(display_rows)
    clean_df = display_df[display_df["Status Flag"].isin(["✅ OK", "OK"])].copy()
    
    st.markdown("---")
    st.header("📥 Export Results")
    
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    
    col_dl1, col_dl2, empty_space = st.columns([1.2, 1.2, 5])
    
    with col_dl1:
        st.download_button("📥 Download Data Table (CSV)", data=csv_bytes, file_name="tadpole_measurements.csv", mime="text/csv")
    with col_dl2:
        st.download_button("🎨 Download Painted Images (ZIP)", data=st.session_state.zip_data, file_name="analyzed_tadpoles.zip", mime="application/zip")

    st.markdown("---")
    st.header("🔍 Visual Tadpole Inspector")
    st.write("Select a file to compare the raw microscope image with the exact solid shapes measured by the AI.")
    
    selected_file = st.selectbox("Select Image to Inspect:", [row["Filename"] for row in st.session_state.raw_results_data])
    
    if selected_file:
        img_data = st.session_state.image_dictionary[selected_file]
        insp_col1, insp_col2 = st.columns(2)
        with insp_col1:
            st.image(img_data["original"], caption="Original Image", use_container_width=True)
        with insp_col2:
            st.image(img_data["painted"], caption="Mathematical Polygons", use_container_width=True)

    st.markdown("---")
    st.header("📈 Analytics & Data")
    
    ctrl_df = clean_df[clean_df["Condition"] == "Control"]
    mut_df = clean_df[clean_df["Condition"] == "Mutant"]
    
    st.write("##### 🔵 Control Group")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Uploaded", len(display_df[display_df["Condition"] == "Control"]))
    c2.metric("Without Error*", len(ctrl_df))
    c3.metric(f"Avg Head Area", f"{round(ctrl_df[f'Head Area {area_unit}'].mean(), 2)} {area_unit.strip('()')}" if len(ctrl_df) > 0 else "N/A")
    c4.metric(f"Avg IP Dist", f"{round(ctrl_df[f'Interpupillary Dist {dist_unit}'].mean(), 2)} {dist_unit.strip('()')}" if len(ctrl_df) > 0 else "N/A")
    c2.caption("*Check data table for error")

    st.write("##### 🔶 Mutant Group")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Uploaded", len(display_df[display_df["Condition"] == "Mutant"]))
    m2.metric("Without Error*", len(mut_df))
    m3.metric(f"Avg Head Area", f"{round(mut_df[f'Head Area {area_unit}'].mean(), 2)} {area_unit.strip('()')}" if len(mut_df) > 0 else "N/A")
    m4.metric(f"Avg IP Dist", f"{round(mut_df[f'Interpupillary Dist {dist_unit}'].mean(), 2)} {dist_unit.strip('()')}" if len(mut_df) > 0 else "N/A")
    m2.caption("*Check data table for error")

    if len(clean_df) > 0:
        st.markdown("---")
        st.write("###### Head Area vs. Interpupillary Distance Distribution")
        
        fig = px.scatter(clean_df, 
                         x=f"Head Area {area_unit}", 
                         y=f"Interpupillary Dist {dist_unit}", 
                         hover_data=["Filename", "Status Flag", "Orbital Asymmetry % Diff", f"Interpupillary Dist Ratio {dist_unit}"], 
                         color="Condition", 
                         symbol="Condition",
                         color_discrete_sequence=["#0072B2", "#E69F00"], 
                         labels={"Condition": "Group"})
        
        fig.update_traces(marker=dict(size=10)) 
        st.plotly_chart(fig, use_container_width=True)
        
        plot_html = fig.to_html(include_plotlyjs="cdn")
        
        st.download_button("📉 Download Interactive Plot (HTML)", data=plot_html, file_name="tadpole_distribution_plot.html", mime="text/html")

    st.write("### 📝 Raw Data Table")
    
    with st.expander("📖 Data Dictionary & Formulas"):
        st.markdown("""
        * **Head Area:** The total 2D cross-sectional area of the detected head.
        * **Interpupillary Distance:** The shortest linear distance between the inner edges of the left and right eyes.
        * **Left and Right Eye Area:** The 2D cross-sectional area of the individual eyes.
        * **Total Eye Area:** The sum of the left and right eye areas.
        * **Orbital Symmetry Ratio:** `Left Eye Area / Right Eye Area`. A value of 1.0 indicates perfect developmental symmetry.
        * **Orbital Asymmetry % Diff:** The absolute difference between the two eyes divided by their average, expressed as a percentage.
        * **Interpupillary Distance Ratio:** `Head Area / Interpupillary Distance`. 
        * **Status Flag:** Indicates if the AI successfully measured all traits (`OK`) or encountered an anatomical anomaly (e.g., `Missing Head`).
        """)

    st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.caption("""
**Open Source License:** Powered by YOLOv8. Source code and model weights are freely available under the AGPL-3.0 license. 
""")