import fitz  # PyMuPDF
from datetime import datetime
from PIL import Image
import os


data = {
    "latitude": 42.653265,
    "longitude": -83.318236,
    "elevation": 287.7,
    "accuracy": 4.3,

    "pressure": 1030,
    "temperature": 18,
    "humidity": 42,
    "air_quality": "Fresh air",

    "soil_moisture": 37,
    "subsurface_temp": 36,

    "soil_type": "Red soil",
    "soil_texture": "Granular",

    "rock_type": "Sedimentary rock",
    "rock_char": "Porous, Grainy",
    "rock_comp": "Si, O, Al",

    "wavelength": 570,
    "absorbance": 0.311,
    "transmittance": 34.57,

    "strata_texture": "Fine",
    "water_retention": "High",
    "formation": "River beds, Floodplains",

    "strata_texture": "Fine",
    "water_retention": "High",
    "environment": "River beds, Floodplains",
    "water_history": "Indicates past river activity, fertile for farming"
}


# CREATE PDF
doc = fitz.open()
page = doc.new_page(width=595, height=842)  # A4

if os.path.exists("logo.png"):
    page.insert_image(
        fitz.Rect(30, 20, 100, 90),
        filename="logo.png"
    )

page.insert_text(
    (150, 55),
    "SITE ANALYSIS REPORT",
    fontsize=20,
    fontname="hebo"
)

#page.insert_text(
    #(430, 85),
    #f"Date: {datetime.now().strftime('%d-%m-%Y')}",
    #fontsize=9
#)

if os.path.exists("site1.png"):
    page.insert_image(
        fitz.Rect(40, 110, 555, 240),
        filename="site1.png"
    )


# LOCATION INFO
y = 180
page.insert_text((50, y), f"Latitude: {data['latitude']}", fontsize=10)
page.insert_text((50, y+15), f"Longitude: {data['longitude']}", fontsize=10)
page.insert_text((50, y+30), f"Elevation: {data['elevation']} meters", fontsize=10)
page.insert_text((50, y+45), f"Accuracy: ±{data['accuracy']} meters", fontsize=10)


# ATMOSPHERIC ANALYSIS
page.insert_text((50, 350-90), "ATMOSPHERIC ANALYSIS:", fontsize=14, fontname="hebo")
page.insert_text((50, 375-90), f"Air Quality: {data['air_quality']}", fontsize=10)
page.insert_text((50, 390-90), f"Pressure: {data['pressure']} hPa", fontsize=10)
page.insert_text((50, 405-90), f"Temperature: {data['temperature']} °C", fontsize=10)
page.insert_text((50, 420-90), f"Humidity: {data['humidity']} %", fontsize=10)

# SOIL ANALYSIS
page.insert_text((320, 350-90), "SOIL ANALYSIS:", fontsize=14, fontname="hebo")
page.insert_text((320, 375-90), f"Moisture: {data['soil_moisture']} %", fontsize=10)
page.insert_text((320, 390-90), f"Subsurface Temp: {data['subsurface_temp']} °C", fontsize=10)


# MICROSCOPIC IMAGE ANALYSIS
page.insert_text((50, 460-90), "MICROSCOPIC IMAGE ANALYSIS:", fontsize=14, fontname="hebo")

# ---- SOIL IMAGE ----
page.insert_text((50, 490-90), "SOIL SAMPLE:", fontsize=12, fontname="hebo")

if os.path.exists("soil.png"):
    page.insert_image(
        fitz.Rect(50, 510-90, 220, 650-90),
        filename="soil.png"
    )

page.insert_text((50, 665-90), f"Type: {data['soil_type']}", fontsize=10)
page.insert_text((50, 680-90), f"Texture: {data['soil_texture']}", fontsize=10)

# ---- ROCK IMAGE ----
page.insert_text((320, 490-90), "ROCK SAMPLE:", fontsize=12, fontname="hebo")

if os.path.exists("rock.png"):
    page.insert_image(
        fitz.Rect(320, 510-90, 490, 650-90),
        filename="rock.png"
    )

page.insert_text((320, 665-90), f"Type: {data['rock_type']}", fontsize=10)
page.insert_text((320, 680-90), f"Characteristics: {data['rock_char']}", fontsize=10)
page.insert_text((320, 695-90), f"Composition: {data['rock_comp']}", fontsize=10)

# SPECTROSCOPIC ANALYSIS
page.insert_text((50, 730-90), "SPECTROSCOPIC ANALYSIS:", fontsize=14, fontname="hebo")
page.insert_text((50, 755-90), f"Wavelength: {data['wavelength']} nm", fontsize=10)
page.insert_text((50, 770-90), f"Absorbance: {data['absorbance']}", fontsize=10)
page.insert_text((50, 785-90), f"Transmittance: {data['transmittance']} %", fontsize=10)

# STRATIGRAPHIC ANALYSIS
page.insert_text((320, 730-90), "STRATIGRAPHIC PROFILE:", fontsize=14, fontname="hebo")
page.insert_text((320, 755-90), f"Texture: {data['strata_texture']}", fontsize=10)
page.insert_text((320, 770-90), f"Water Retentivity: {data['water_retention']}", fontsize=10)
page.insert_text((320, 785-90), f"Environment Formation: {data['environment']}", fontsize=10)
page.insert_text((320, 800-90), f"Water History: {data['water_history']}", fontsize=10)

# SAVE
doc.save("Site_Analysis_Report_Final.pdf")
doc.close()

print("✅ PDF with logo & soil images generated successfully!")
