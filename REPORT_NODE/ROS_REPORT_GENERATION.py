import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import json
import base64
import os
import fitz
from datetime import datetime


class pdf_node(Node):
    def __init__(self):
        super().__init__("pdf_node")
        self.pub_approve = self.create_publisher(String, 'approve', 10)
        self.sub_data = self.create_subscription(
            String,
            'pdf_data',
            self.callback,
            10
        )

        self.approved = False
        self.pdf_done = False

        # ---------------- DATA STORE ----------------
        self.data_store = {
            "latitude": None,
            "longitude": None,
            "elevation": None,
            "accuracy": None,

            "pressure": None,
            "temperature": None,
            "humidity": None,
            "air_quality": None,

            "soil_moisture": None,
            "subsurface_temp": None,
            "soil_type": None,
            "soil_texture": None,

            "rock_type": None,
            "rock_char": None,
            "rock_comp": None,

            "wavelength": None,
            "absorbance": None,
            "transmittance": None,

            "strata_texture": None,
            "water_retention": None,
            "environment": None,
            "water_history": None,
        }

        # ---------------- IMAGE SLOTS ----------------
        self.image_slots = {
            "soil": {"received": False, "path": "soil.png"},
            "rock": {"received": False, "path": "rock.png"},
            "field": {"received": False, "path": "site1.png"}
        }

        t = threading.Thread(target=self.wait_for_approval, daemon=True)
        t.start()
        self.get_logger().info("Workstation PDF node started. Waiting for approval...")

    # =================================================
    # USER APPROVAL
    # =================================================
    def wait_for_approval(self):
        usr = input(">>>>> Enter [ c ] to Start Create PDF : ").strip()
        if usr.lower() == 'c':
            msg = String()
            msg.data = 'c'
            self.pub_approve.publish(msg)
            self.approved = True
            self.init_pdf()
            self.get_logger().info("Approved. Waiting for data & images from Jetson...")

    # =================================================
    # PDF INITIALIZATION (YOUR FORMAT)
    # =================================================
    def init_pdf(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=595, height=842)

        if os.path.exists("logo.png"):
            self.page.insert_image(
                fitz.Rect(30, 20, 100, 90),
                filename="logo.png"
            )

        self.page.insert_text(
            (150, 55),
            "SITE ANALYSIS REPORT",
            fontsize=20,
            fontname="hebo"
        )

        self.page.insert_text(
            (430, 85),
            f"Date: {datetime.now().strftime('%d-%m-%Y')}",
            fontsize=9
        )

    # =================================================
    # DATA + IMAGE CALLBACK
    # =================================================
    def callback(self, msg):
        if not self.approved or self.pdf_done:
            return

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid JSON received")
            return

        # ---------- IMAGE HANDLING ----------
        if "type" in data and "image" in data:
            self.handle_image(data)
        else:
            self.handle_data(data)

        # ---------- CHECK COMPLETION ----------
        if self.ready_for_pdf():
            self.generate_pdf()
            self.pdf_done = True
            self.get_logger().info("PDF GENERATED SUCCESSFULLY")

    # =================================================
    # HANDLE NORMAL DATA
    # =================================================
    def handle_data(self, incoming):
        for key, value in incoming.items():
            if key in self.data_store:
                self.data_store[key] = value
                self.get_logger().info(f"Stored {key} = {value}")

    # =================================================
    # HANDLE IMAGE JSON
    # =================================================
    def handle_image(self, data):
        img_type = data["type"]  # soil / rock / field

        if img_type not in self.image_slots:
            self.get_logger().warn(f"Unknown image type: {img_type}")
            return

        img_bytes = base64.b64decode(data["image"])
        img_path = self.image_slots[img_type]["path"]

        with open(img_path, "wb") as f:
            f.write(img_bytes)

        self.image_slots[img_type]["received"] = True
        self.get_logger().info(f"{img_type} image received")

    # =================================================
    # CHECK FUNCTIONS
    # =================================================
    def all_data_received(self):
        return all(v is not None for v in self.data_store.values())

    def all_images_received(self):
        return all(slot["received"] for slot in self.image_slots.values())

    def ready_for_pdf(self):
        return self.all_data_received() and self.all_images_received()

    # =================================================
    # PDF GENERATION (YOUR EXACT LAYOUT)
    # =================================================
    def generate_pdf(self):
        d = self.data_store
        p = self.page

        # FIELD IMAGE
        p.insert_image(
            fitz.Rect(40, 110, 555, 240),
            filename=self.image_slots["field"]["path"]
        )

        y = 180
        p.insert_text((50, y), f"Latitude: {d['latitude']}", fontsize=10)
        p.insert_text((50, y+15), f"Longitude: {d['longitude']}", fontsize=10)
        p.insert_text((50, y+30), f"Elevation: {d['elevation']} meters", fontsize=10)
        p.insert_text((50, y+45), f"Accuracy: ±{d['accuracy']} meters", fontsize=10)

        # ATMOSPHERIC
        p.insert_text((50, 260), "ATMOSPHERIC ANALYSIS:", fontsize=14, fontname="hebo")
        p.insert_text((50, 285), f"Air Quality: {d['air_quality']}", fontsize=10)
        p.insert_text((50, 300), f"Pressure: {d['pressure']} hPa", fontsize=10)
        p.insert_text((50, 315), f"Temperature: {d['temperature']} °C", fontsize=10)
        p.insert_text((50, 330), f"Humidity: {d['humidity']} %", fontsize=10)

        # SOIL IMAGE
        p.insert_image(
            fitz.Rect(50, 420, 220, 580),
            filename=self.image_slots["soil"]["path"]
        )
        p.insert_text((50, 600), f"Type: {d['soil_type']}", fontsize=10)
        p.insert_text((50, 615), f"Texture: {d['soil_texture']}", fontsize=10)

        # ROCK IMAGE
        p.insert_image(
            fitz.Rect(320, 420, 490, 580),
            filename=self.image_slots["rock"]["path"]
        )
        p.insert_text((320, 600), f"Type: {d['rock_type']}", fontsize=10)
        p.insert_text((320, 615), f"Char: {d['rock_char']}", fontsize=10)
        p.insert_text((320, 630), f"Comp: {d['rock_comp']}", fontsize=10)

        # SAVE
        self.doc.save("Site_Analysis_Report_Final.pdf")
        self.doc.close()


def main(args=None):
    rclpy.init(args=args)
    node = pdf_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
