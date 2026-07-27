import cv2
import numpy as np
from PIL import Image
import streamlit as st
import rawpy
import io
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==============================================================================
# 1. MASSIVE CONTEXT MATRICES (150+ CONDITIONS)
# ==============================================================================

GEO_MATRIX = {
    "Neutral": (0, 1.0), "Urban / City": (-5, 0.85), "Forest / Jungle": (5, 1.15), "Desert": (25, 1.1),
    "Ocean / Beach": (-15, 1.2), "Mountains / Alpine": (-20, 1.05), "Indoor / Studio": (5, 0.95),
    "Cyberpunk Cityscape": (-10, 1.4), "Abandoned Factory": (-5, 0.7), "Neon Alleyway": (10, 1.3),
    "Glacier / Ice Cave": (-35, 1.1), "Redwood Forest": (15, 1.1), "Space Station": (-25, 0.8),
    "Volcanic Crater": (40, 1.2), "Coral Reef": (5, 1.4), "Salt Flat": (15, 0.9),
    "Zen Garden": (5, 1.0), "Bamboo Forest": (0, 1.25), "Martian Surface": (45, 1.2),
    "Lunar Crater": (-20, 0.5), "Post-Apocalyptic Wasteland": (15, 0.6), "Suburban Street": (5, 1.0),
    "Shopping Mall": (-5, 1.1), "Subway Station": (-10, 0.9), "Airport Terminal": (-15, 0.85),
    "Gothic Cathedral": (-20, 0.75), "High-Tech Laboratory": (-30, 0.9), "Antique Library": (25, 0.85),
    "Music Festival": (10, 1.3), "Amusement Park": (5, 1.35), "Haunted Mansion": (-25, 0.6),
    "Casino Floor": (15, 1.25), "Sports Stadium": (-5, 1.1), "Ancient Pyramid": (30, 1.05),
    "Deep Ocean Trench": (-40, 0.5), "Floating Island": (10, 1.2), "Overgrown Highway": (5, 1.1),
    "Cyberpunk Slum": (-15, 1.3), "Victorian London": (-10, 0.7), "Neon Tokyo": (15, 1.4),
    "Favela / Shanty Town": (20, 1.15), "Medieval Castle": (-15, 0.8), "Underground Bunker": (-20, 0.7),
    "Tundra / Permafrost": (-30, 0.85), "Tropical Island": (15, 1.3), "Savanna / Grassland": (25, 1.15),
    "Canyon / Badlands": (20, 1.1), "Swamp / Mangrove": (10, 1.1), "Farmland / Rural": (15, 1.1)
}

WEATHER_MATRIX = {
    "Neutral": (0, 1.0), "Sunny / Clear": (5, 1.1), "Overcast / Rainy": (-15, 0.8), "Snow": (15, 1.05),
    "Stormy / Dark": (-20, 0.6), "Fog / Mist": (-5, 0.7), "Thunderstorm / Lightning": (-25, 0.7),
    "Acid Rain": (10, 1.2), "Toxic Smog / Pollution": (15, 0.6), "Aurora Borealis": (-30, 1.4),
    "Wildfire Smoke": (35, 0.7), "Blood Moon": (40, 0.9), "Pitch Black Night": (-30, 0.4),
    "Sandstorm / Dust": (40, 0.9), "Blizzard / Whiteout": (10, 0.5), "Hail / Sleet": (-15, 0.8),
    "Drizzle / Light Rain": (-10, 0.9), "Rainbow / Post-Rain": (10, 1.25), "Humid / Muggy": (15, 1.1),
    "Windy / Autumn Leaves": (20, 1.15), "Heatwave / Haze": (30, 0.9), "Eclipse / Unearthly": (-20, 0.5),
    "Ash Fall": (-10, 0.4), "Sun Showers": (10, 1.15), "Heavy Monsoon": (-15, 0.85),
    "Freezing Rain": (-20, 0.9), "Sea Mist": (-10, 0.8), "Morning Dew": (5, 1.1),
    "Solar Flare / Radiation": (40, 1.3), "Pollen Storm": (20, 1.2), "Tornado / Hurricane": (-25, 0.6),
    "Frost / Rime": (-15, 0.95), "Golden Mist": (25, 0.9), "Crimson Sky": (35, 1.2),
    "Purple Haze": (-10, 1.1), "Midnight Sun": (15, 0.8), "White Nights": (-5, 0.7),
    "Meteor Shower": (-20, 0.85), "Supercell Cloud": (-30, 0.65), "Mammatus Clouds": (-15, 0.8),
    "God Rays": (20, 1.1), "Dry Lightning": (10, 0.7), "Freezing Fog": (-25, 0.6),
    "Volcanic Lightning": (25, 1.3), "Sweltering Humidity": (15, 1.1), "Crisp Winter": (-20, 1.0),
    "Radioactive Fallout": (15, 1.4), "Time Rift / Anomaly": (-40, 1.5), "Ethereal Fog": (-15, 0.5)
}

# UPDATED: Format -> (Brightness, Gamma, Temperature, Saturation)
LIGHTING_MATRIX = {
    "Neutral": (0, 1.0, 0, 1.0), 
    "Golden Hour": (15, 0.9, 20, 1.25), 
    "Blue Hour / Twilight": (-30, 1.3, -25, 1.1),
    "Midday Sun": (25, 0.8, -8, 0.95), 
    "Neon / Cyberpunk": (-15, 1.2, 0, 1.5), 
    "Night / Low Light": (-60, 1.6, 10, 0.85),
    "Sodium Vapor Lamps": (-20, 1.2, 30, 0.9), 
    "Fluorescent Green (Matrix)": (0, 1.1, -15, 1.3), 
    "Red Room / Darkroom": (-30, 1.4, 50, 1.5), 
    "Lightning Flash": (60, 0.6, -40, 0.5), 
    "Rim Lighting": (10, 0.9, 10, 1.1),
    "Halogen Floodlights": (30, 0.8, -10, 0.8), 
    "Tungsten Bulb": (10, 1.0, 25, 1.1), 
    "Blacklight / UV": (-50, 1.5, -40, 1.6),
    "Disco Ball Reflections": (0, 1.1, 0, 1.4), 
    "Campfire Glow": (10, 1.1, 35, 1.2), 
    "Flashlight Beam": (20, 0.9, -5, 0.9),
    "Car Headlights": (40, 0.8, -10, 1.0), 
    "Police Sirens (Red/Blue)": (10, 1.0, 10, 1.3), 
    "Computer Monitor": (5, 1.1, -15, 0.8),
    "Holographic Projection": (15, 1.0, -25, 1.2), 
    "Fairy Lights": (5, 1.1, 15, 1.1), 
    "Chandelier Sparkle": (10, 1.0, 20, 1.15),
    "Stained Glass Refraction": (-10, 1.2, 5, 1.4), 
    "Lava Glow": (5, 1.1, 45, 1.3), 
    "Bioluminescent Fungi": (-40, 1.4, -30, 1.5),
    "Phosphorescent Algae": (-40, 1.4, -35, 1.4), 
    "Camera Obscura": (-50, 1.5, 10, 0.7), 
    "Silhouette / Backlit": (-60, 1.8, 5, 0.6),
    "Rembrandt Lighting": (0, 1.1, 15, 1.0), 
    "Split Lighting": (-10, 1.2, -5, 0.9), 
    "Butterfly Lighting": (15, 0.9, 5, 1.0),
    "Interrogation Room": (-10, 1.3, -20, 0.6), 
    "Stadium Floodlights": (40, 0.7, -15, 1.1), 
    "Searchlight Beam": (30, 0.8, -10, 0.8),
    "UFO Abduction Beam": (50, 0.6, -30, 1.4), 
    "Cinematic Orange / Teal": (0, 1.1, 10, 1.25), 
    "Candlelight": (-20, 1.2, 40, 1.1),
    "Muzzle Flash / Gunshot": (30, 0.7, 20, 1.1), 
    "Bioluminescent Ocean": (-50, 1.5, -45, 1.5), 
    "Ethereal Glow": (20, 0.8, -10, 0.8),
    "Harsh Flash / Paparazzi": (50, 0.7, -5, 0.9), 
    "Starlight / Astrophotography": (-60, 1.6, -20, 0.7), 
    "Window Light / Diffused": (10, 0.9, 5, 1.05), 
    "Underwater Caustic": (-20, 1.2, -50, 1.2), 
    "Sunrise / Dawn": (15, 0.9, 15, 1.15),
    "Sunset / Dusk": (10, 1.0, 25, 1.2), 
    "Moonlight / Silver": (-40, 1.4, -35, 0.6), 
    "Overhead Fluorescent": (20, 0.9, -15, 0.8)
}

# ==============================================================================
# 2. NVIDIA SEGFORMER ENGINE + LUMINOSITY MASKING
# ==============================================================================

@st.cache_resource
def load_ai_model():
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return processor, model

class MultiLayerGrader:
    def __init__(self, image_array, is_raw=False):
        self.original_image = image_array.astype(np.uint8)
        self.is_raw = is_raw
        self.adjustment_scale = 1.0 if is_raw else 0.4
        
        self.sky_mask = None
        self.subject_mask = None
        self.env_mask = None

    def generate_semantic_masks(self):
        processor, model = load_ai_model()
        pil_img = Image.fromarray(self.original_image)
        inputs = processor(images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits, size=self.original_image.shape[:2], mode="bilinear", align_corners=False
        )
        
        segmentation_map = logits.argmax(dim=1)[0].numpy()
        
        raw_sky_mask = (segmentation_map == 2).astype(np.float32)
        raw_subject_mask = ((segmentation_map == 12) | (segmentation_map == 126)).astype(np.float32)
        raw_env_mask = (1.0 - raw_sky_mask - raw_subject_mask)
        raw_env_mask = np.clip(raw_env_mask, 0.0, 1.0)
        
        self.sky_mask = np.stack([raw_sky_mask]*3, axis=2)
        self.subject_mask = np.stack([raw_subject_mask]*3, axis=2)
        self.env_mask = np.stack([raw_env_mask]*3, axis=2)

    def apply_shadow_grade(self, composited_image, total_w):
        luma = cv2.cvtColor(composited_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        shadow_mask = np.clip(1.0 - (luma / 0.4), 0.0, 1.0)
        shadow_mask_3d = np.stack([shadow_mask]*3, axis=2)
        
        shadow_layer = composited_image.copy().astype(np.float32)
        
        if total_w < 0:
            shadow_layer[:,:,0] -= 25 * self.adjustment_scale  
            shadow_layer[:,:,1] += 10 * self.adjustment_scale  
            shadow_layer[:,:,2] += 30 * self.adjustment_scale  
        else:
            shadow_layer[:,:,0] += 25 * self.adjustment_scale  
            shadow_layer[:,:,1] += 5  * self.adjustment_scale  
            shadow_layer[:,:,2] -= 20 * self.adjustment_scale  
            
        shadow_layer = np.clip(shadow_layer, 0, 255)
        
        final_img = (composited_image * (1.0 - shadow_mask_3d)) + (shadow_layer * shadow_mask_3d)
        
        return final_img.astype(np.uint8)

    def adjust_temperature(self, image, base_warmth_factor, scale_modifier=1.0):
        warmth_factor = base_warmth_factor * self.adjustment_scale * scale_modifier
        img_float = image.astype(np.float32)
        if warmth_factor > 0:
            img_float[:, :, 0] += warmth_factor * 1.05 
            img_float[:, :, 2] -= warmth_factor        
        elif warmth_factor < 0:
            img_float[:, :, 2] -= warmth_factor * 1.05 
            img_float[:, :, 0] += warmth_factor        
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def adjust_hsv(self, image, base_saturation, scale_modifier=1.0):
        sat = 1.0 + (base_saturation - 1.0) * self.adjustment_scale * scale_modifier
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= sat
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # NEW: Adjusts actual exposure and contrast in LAB Lightness Channel
    def adjust_luminance(self, image, brightness_shift, gamma):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Apply absolute brightness shift
        l = l + (brightness_shift * self.adjustment_scale)
        
        # Apply Gamma curve (Contrast/Midtone shift)
        l = l / 255.0
        l = np.clip(l, 0.0, 1.0) ** gamma
        l = l * 255.0
        
        lab = cv2.merge((np.clip(l, 0, 255), a, b)).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def composite_image(self, environment, weather, lighting):
        self.generate_semantic_masks()
        
        geo_w, geo_s = GEO_MATRIX.get(environment, (0, 1.0))
        wea_w, wea_s = WEATHER_MATRIX.get(weather, (0, 1.0))
        
        # Pull new 4-parameter lighting config
        lig_b, lig_g, lig_w, lig_s = LIGHTING_MATRIX.get(lighting, (0, 1.0, 0, 1.0))

        total_w = geo_w + wea_w + lig_w
        total_s = geo_s * wea_s * lig_s

        # Subject Base Processing
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4,4)).apply(l)
        subject = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
        subject = self.adjust_temperature(subject, 5)
        subject = self.adjust_luminance(subject, lig_b, lig_g)
        
        # Environment Base Processing
        env = self.original_image.copy()
        env = self.adjust_temperature(env, total_w)
        env = self.adjust_hsv(env, total_s)
        env = self.adjust_luminance(env, lig_b, lig_g)

        # Sky Base Processing
        sky = self.original_image.copy()
        sky = self.adjust_temperature(sky, total_w, scale_modifier=1.5) 
        sky = self.adjust_hsv(sky, total_s, scale_modifier=1.2)
        sky = self.adjust_luminance(sky, lig_b, lig_g)
        
        base_composite = (subject * self.subject_mask) + (env * self.env_mask) + (sky * self.sky_mask)
        base_composite = base_composite.astype(np.uint8)
        
        final_array = self.apply_shadow_grade(base_composite, total_w)
        
        return final_array

# ==============================================================================
# 3. STREAMLIT FRONTEND
# ==============================================================================

st.set_page_config(page_title="CHROMATIX", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3, h4 { color: #FFFFFF !important; font-family: 'Courier New', Courier, monospace; text-transform: uppercase; font-weight: bold; }
    hr { border-top: 1px solid #333333; }
    .stButton>button { background-color: transparent !important; color: #FFFFFF !important; border: 1px solid #FFFFFF !important; border-radius: 0px !important; text-transform: uppercase; }
    .stButton>button:hover { background-color: #FFFFFF !important; color: #000000 !important; }
    div[data-baseweb="select"] > div { background-color: transparent !important; border: 1px solid #555555 !important; border-radius: 0px !important; color: #FFFFFF !important; }
    div[data-testid="stFileUploader"] { border: 1px dashed #FFFFFF !important; border-radius: 0px !important; background-color: transparent !important; }
    .status-box { border: 1px solid #FFFFFF; padding: 15px; font-size: 0.9rem; text-transform: uppercase; }
    img { border-radius: 0px !important; border: 1px solid #333333; }
</style>
""", unsafe_allow_html=True)

st.title("CHROMATIX")
st.markdown("<hr>", unsafe_allow_html=True)

# Initialize Session State Memory 
if "master_image" not in st.session_state:
    st.session_state.master_image = None
if "source_image" not in st.session_state:
    st.session_state.source_image = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("SELECT CONDITIONS")
    # 1. FILE UPLOADER MOVED OUTSIDE THE FORM (Triggers instantly)
    uploaded_file = st.file_uploader("[ SELECT IMAGE ]", type=["jpg", "png", "jpeg", "cr2", "nef"])
    
    # 2. THE RENDER FORM
    with st.form("grading_form"):
        environment = st.selectbox("Geography", sorted(list(GEO_MATRIX.keys())))
        weather = st.selectbox("Weather", sorted(list(WEATHER_MATRIX.keys())))
        lighting = st.selectbox("Lighting", sorted(list(LIGHTING_MATRIX.keys())))
        st.markdown("<br>", unsafe_allow_html=True)
        render_button = st.form_submit_button("[ INITIATE RENDER ]")

with col2:
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_raw_input = file_extension in ['cr2', 'nef', 'arw', 'dng']

        # Only decode the image if it's a completely new file
        if st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("Decoding Media Matrices..."):
                if is_raw_input:
                    uploaded_file.seek(0)
                    with rawpy.imread(uploaded_file) as raw:
                        image_array = raw.postprocess(use_camera_wb=True)
                else:
                    raw_image = Image.open(uploaded_file).convert('RGB')
                    image_array = np.array(raw_image)
                    
                # MEMORY OPTIMIZATION: Downscale to prevent RAM crash
                max_dimension = 1280
                height, width = image_array.shape[:2]
                if max(height, width) > max_dimension:
                    scale = max_dimension / float(max(height, width))
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Save new image to session state and reset the render
                st.session_state.source_image = image_array
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.master_image = None

        # If the user clicks the submit button, run the AI math
        if render_button:
            st.markdown(f'''
            <div class="status-box">
                TARGET........: {uploaded_file.name}<br>
                NETWORK.......: NVIDIA SEGFORMER (150 CLASSES)<br>
                COMPOSITING...: TRI-LAYER AI + LUMINOSITY MASKS<br>
                RESOLUTION....: SCALED FOR CLOUD MEMORY LIMITS
            </div><br>
            ''', unsafe_allow_html=True)

            grader = MultiLayerGrader(st.session_state.source_image, is_raw=is_raw_input)
            with st.spinner(f"Running Pixel Classification & Shadow Mapping..."):
                final_image = grader.composite_image(environment, weather, lighting)
                st.session_state.master_image = final_image

        # DISPLAY LOGIC
        if st.session_state.master_image is not None:
            # Show both Source and Render side-by-side
            st.markdown("SOURCE IMAGE VS FINAL IMAGE")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(st.session_state.source_image, caption="SOURCE IMAGE", use_container_width=True)
            with img_col2:
                st.image(st.session_state.master_image, caption="FINAL IMAGE", use_container_width=True)
                
            st.markdown("<hr>", unsafe_allow_html=True)
            buffer = io.BytesIO()
            Image.fromarray(st.session_state.master_image).save(buffer, format="JPEG", quality=95)
            st.download_button(
                label="[ EXPORT FINAL IMAGE ]",
                data=buffer.getvalue(),
                file_name=f"chroma_multilayer.jpg",
                mime="image/jpeg",
            )
        else:
            # Only the image is uploaded, no render has happened yet
            st.markdown('<div class="status-box" style="margin-bottom: 20px;">SYSTEM STATUS : MEDIA LOADED.<br>CONFIGURE MATRICES AND INITIATE RENDER.</div>', unsafe_allow_html=True)
            st.image(st.session_state.source_image, caption="SOURCE PREVIEW", use_container_width=True)

    else:
        # Default empty state
        st.markdown('<div class="status-box">SYSTEM STATUS : IDLE<br>AWAITING MEDIA INGESTION</div>', unsafe_allow_html=True)
