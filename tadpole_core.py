# tadpole_core.py
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

def calculate_morphology(res, pixel_to_mm_ratio=1.0):
    stats = {
        "Head Area": np.nan,
        "Interpupillary Distance": np.nan,
        "Left Eye Area": np.nan,
        "Right Eye Area": np.nan,
        "Total Eye Area": np.nan, 
        "Orbital Symmetry Ratio": np.nan, 
        "Orbital Asymmetry % Difference": np.nan,
        "Interpupillary Distance Ratio": np.nan
    }
    flag = "OK"  
    
    final_head_obj = None
    valid_eyes_objs = []

    if res.masks is None:
        return stats, "No detections found", final_head_obj, valid_eyes_objs

    masks = res.masks.xy
    classes = res.boxes.cls.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    names = res.names
    
    raw_heads = []
    raw_eyes = []
    
    for mask_coords, cls, conf in zip(masks, classes, confs):
        if len(mask_coords) >= 3:
            poly = Polygon(mask_coords)
            
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            label = names[int(cls)]
            
            if label == 'Head':
                raw_heads.append({"poly": poly, "conf": float(conf)})
            elif label == 'Eye':
                raw_eyes.append({"poly": poly, "conf": float(conf)})
                
    if len(raw_heads) > 0:
        raw_heads.sort(key=lambda x: x["conf"], reverse=True)
        final_head_obj = raw_heads[0]
    else:
        flag = "Missing Head"
        
    if final_head_obj is not None:
        
        # 1. Find Valid Eyes
        head_search_area = final_head_obj["poly"].buffer(200.0)
        for eye in raw_eyes:
            if head_search_area.intersects(eye["poly"]):
                valid_eyes_objs.append(eye)
                
        # 2. THE SURGICAL SPACKLE
        patched_head = final_head_obj["poly"]
        
        for eye in valid_eyes_objs:
            # Isolate a tight 30-pixel zone around the eye
            local_zone = eye["poly"].buffer(30.0)
            
            # Grab just the broken edge of the head inside that local zone
            local_head = patched_head.intersection(local_zone)
            
            # Snap a rubber band (convex hull) around just the eye and that local edge to create a patch
            local_patch = unary_union([local_head, eye["poly"]]).convex_hull
            
            # Plaster the patch onto the main head. 
            # The rest of the tadpole is completely mathematically untouched!
            patched_head = unary_union([patched_head, local_patch])
            
        if patched_head.geom_type == 'MultiPolygon':
            patched_head = max(patched_head.geoms, key=lambda a: a.area)
            
        final_head_obj["poly"] = patched_head
        
        # 3. Now calculate the corrected, true Head Area
        stats["Head Area"] = round(final_head_obj["poly"].area * (pixel_to_mm_ratio ** 2), 2)
                
    if final_head_obj is not None:
        if len(valid_eyes_objs) == 2:
            valid_eyes_objs.sort(key=lambda x: x["poly"].centroid.x)
            
            left_area = round(valid_eyes_objs[0]["poly"].area * (pixel_to_mm_ratio ** 2), 2)
            right_area = round(valid_eyes_objs[1]["poly"].area * (pixel_to_mm_ratio ** 2), 2)
            
            stats["Left Eye Area"] = left_area
            stats["Right Eye Area"] = right_area
            stats["Total Eye Area"] = round(left_area + right_area, 2)
            
            if right_area > 0:
                stats["Orbital Symmetry Ratio"] = round(left_area / right_area, 3)
                
            if (left_area + right_area) > 0:
                diff = abs(left_area - right_area)
                avg = (left_area + right_area) / 2.0
                stats["Orbital Asymmetry % Difference"] = round((diff / avg) * 100, 2)
            
            dist = valid_eyes_objs[0]["poly"].distance(valid_eyes_objs[1]["poly"])
            stats["Interpupillary Distance"] = round(dist * pixel_to_mm_ratio, 2)
            
            if stats["Interpupillary Distance"] > 0:
                stats["Interpupillary Distance Ratio"] = round(stats["Head Area"] / stats["Interpupillary Distance"], 2)
            else:
                flag = "Eyes Overlapping (Dist=0)"
        else:
            flag = f"Found {len(valid_eyes_objs)} Valid Eyes"
            
    return stats, flag, final_head_obj, valid_eyes_objs


def paint_measured_biology(res, final_head_obj, valid_eyes_objs):
    annotated_img = res.orig_img.copy()
    overlay = annotated_img.copy()

    head_color = (0, 94, 213)    
    eye_color = (66, 228, 240)   
    
    if final_head_obj is not None:
        poly_to_draw = final_head_obj["poly"]
        if poly_to_draw.geom_type == 'MultiPolygon':
            poly_to_draw = max(poly_to_draw.geoms, key=lambda a: a.area)
            
        head_coords = np.array(poly_to_draw.exterior.coords, dtype=np.int32)
        cv2.fillPoly(overlay, [head_coords], color=head_color)

    for eye_obj in valid_eyes_objs:
        poly_to_draw = eye_obj["poly"]
        if poly_to_draw.geom_type == 'MultiPolygon':
            poly_to_draw = max(poly_to_draw.geoms, key=lambda a: a.area)
            
        eye_coords = np.array(poly_to_draw.exterior.coords, dtype=np.int32)
        cv2.fillPoly(overlay, [eye_coords], color=eye_color)

    cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0, annotated_img)

    drawn_label_rects = []

    def check_overlap(rect1, rect2):
        return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])

    def draw_smart_label(img, poly, text, label_color, placement_hint):
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda a: a.area)
            
        minx, miny, maxx, maxy = poly.bounds
        text_scale = 1.2 
        text_thickness = 2
        outline_thickness = 5 
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255) 

        (text_width, text_height), baseline = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        
        rect_width = text_width + 10 
        rect_height = text_height + baseline + 10 
        
        if placement_hint == "head":
            rect_left = int((minx + maxx) / 2) - int(rect_width / 2)
            rect_top = int(miny) - rect_height - 5
            nudge_x, nudge_y = 0, -5
        elif placement_hint == "left_eye":
            rect_left = int(minx) - rect_width + 10
            rect_top = int(maxy) + 5
            nudge_x, nudge_y = -5, 5
        else: 
            rect_left = int(maxx) - 10
            rect_top = int(maxy) + 5
            nudge_x, nudge_y = 5, 5

        rect_left = max(0, min(rect_left, img.shape[1] - rect_width))
        rect_top = max(0, min(rect_top, img.shape[0] - rect_height))

        placed = False
        max_attempts = 15
        attempts = 0
        current_left = rect_left
        current_top = rect_top

        while not placed and attempts < max_attempts:
            test_rect = [current_left, current_top, current_left + rect_width, current_top + rect_height]
            if not any(check_overlap(test_rect, drawn) for drawn in drawn_label_rects):
                placed = True
            else:
                current_left += nudge_x
                current_top += nudge_y
                current_left = max(0, min(current_left, img.shape[1] - rect_width))
                current_top = max(0, min(current_top, img.shape[0] - rect_height))
                attempts += 1
                
        if not placed:
            current_left = rect_left
            current_top = rect_top
            
        rect_bottom = current_top + rect_height
        drawn_label_rects.append([current_left, current_top, current_left + rect_width, rect_bottom])
        
        cv2.rectangle(img, (current_left, current_top), (current_left + rect_width, rect_bottom), (0, 0, 0), -1)
        text_org = (current_left + 5, current_top + text_height + 5) 
        cv2.putText(img, text, text_org, text_font, text_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
        cv2.putText(img, text, text_org, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)

    if final_head_obj is not None:
        poly_to_draw = final_head_obj["poly"]
        if poly_to_draw.geom_type == 'MultiPolygon':
            poly_to_draw = max(poly_to_draw.geoms, key=lambda a: a.area)
            
        head_coords = np.array(poly_to_draw.exterior.coords, dtype=np.int32)
        cv2.polylines(annotated_img, [head_coords], isClosed=True, color=head_color, thickness=2)
        label_text = f"Head {final_head_obj['conf']:.2f}"
        draw_smart_label(annotated_img, poly_to_draw, label_text, head_color, "head")

    for idx, eye_obj in enumerate(valid_eyes_objs):
        poly_to_draw = eye_obj["poly"]
        if poly_to_draw.geom_type == 'MultiPolygon':
            poly_to_draw = max(poly_to_draw.geoms, key=lambda a: a.area)
            
        eye_coords = np.array(poly_to_draw.exterior.coords, dtype=np.int32)
        cv2.polylines(annotated_img, [eye_coords], isClosed=True, color=eye_color, thickness=2)
        label_text = f"Eye {eye_obj['conf']:.2f}"
        hint = "left_eye" if idx == 0 else "right_eye"
        draw_smart_label(annotated_img, poly_to_draw, label_text, eye_color, hint)

    return annotated_img