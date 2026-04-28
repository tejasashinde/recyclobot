import base64
import json
import io
import re
import os
import tempfile
import gradio as gr
from PIL import Image
from io import BytesIO
from google import genai
from openai import OpenAI
from google.genai import types
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth

# ------------------ #
# Utility Functions
# ------------------ #

def generate_pdf(item_name, status, instructions, reasoning, classification, impact):
    """Generate a PDF report with proper wrapping and pagination."""

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file_path = tmp_file.name
    tmp_file.close()

    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Margins
    left_margin = 50
    right_margin = 50
    top_margin = 50
    bottom_margin = 50

    max_width = width - left_margin - right_margin
    y = height - top_margin

    # Word wrapping function
    def write_line(text, font="Helvetica", size=11, line_spacing=15):
        nonlocal y

        c.setFont(font, size)

        words = str(text).split()
        line = ""

        for word in words:
            test_line = f"{line} {word}".strip()
            text_width = stringWidth(test_line, font, size)

            if text_width <= max_width:
                line = test_line
            else:
                c.drawString(left_margin, y, line)
                y -= line_spacing

                # Page break check
                if y < bottom_margin:
                    c.showPage()
                    c.setFont(font, size)
                    y = height - top_margin

                line = word

        if line:
            c.drawString(left_margin, y, line)
            y -= line_spacing

    # Section helper
    def add_section(title, content):
        nonlocal y

        # Title
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, title)
        y -= 20

        # Content
        write_line(content)

        y -= 20

        # Page break safeguard
        if y < bottom_margin:
            c.showPage()
            y = height - top_margin

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, y, f"RecycloBot Report: {item_name}")
    y -= 30

    # Sections
    add_section("Item Recyclability Summary", status)
    add_section("Instructions on What Exactly to Do with It", instructions)
    add_section("Why This Matters", reasoning)
    add_section("Smart Item Classification Tags", classification)
    add_section("Environmental Impact Score", impact)

    c.save()

    return file_path

def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def parse_nebius_response(content: str):
    """Parse the response content from Nebius."""

    item_name = re.search(r"Item Name:\s*\*?\*?\s*(.*)", content)
    status = re.search(r"1\.\s*Recyclability Status:\s*\*?\*?\s*(.*)", content)
    instructions = re.search(r"2\.\s*Instructions:\s*\*?\*?\s*(.*?)(?:\s*\*?\*\s*3\.|3\.|$)", content, re.DOTALL)
    reasoning = re.search(r"3\.\s*Reasoning:\s*\*?\*?\s*(.*?)(?:\s*\*?\*\s*4\.|4\.|$)", content, re.DOTALL)
    tags = re.search(r"4\.\s*Smart Item Classification Tags:\s*\*?\*?\s*(.*?)(?:\s*\*?\*\s*5\.|5\.|$)", content, re.DOTALL)
    impact = re.search(r"5\.\s*Environmental Impact Score:\s*\*?\*?\s*(.*)", content, re.DOTALL)

    return {
        "item_name": item_name.group(1).strip() if item_name else "This item",
        "status": status.group(1).strip() if status else "Unknown",
        "instructions": instructions.group(1).strip() if instructions else "",
        "reasoning": reasoning.group(1).strip() if reasoning else "",
        "tags": tags.group(1).strip() if tags else "",
        "impact": impact.group(1).strip() if impact else ""
    }

def parse_gemini_response(response_text: str):
    """Parse the JSON string response from Gemini provider."""

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        data = {}

    return {
        "item_name": data.get("Item Name", "This item").strip(),
        "status": data.get("1. Recyclability Status", "").strip(),
        "instructions": data.get("2. Instructions", "").strip(),
        "reasoning": data.get("3. Reasoning", "").strip(),
        "tags": data.get("4. Smart Item Classification Tags", "").strip(),
        "impact": data.get("5. Environmental Impact Score", "").strip()
    }

def build_prompt(item_description: str, location: str):
    """Build system and user prompts."""
    system_prompt = (
        "You are a recycling and waste management expert. "
        "Your job is to help users determine whether an item is recyclable, and if not, guide them on responsible disposal based on their location. "
        "Be specific, practical, and locally relevant.\n\n"
        "Always format your response as follows:\n"
        "Item Name: <clear name>\n"
        "1. Recyclability Status: Recyclable / Not Recyclable / Depends\n"
        "2. Instructions: What should the user do with the item?\n"
        "3. Reasoning: Why is this the right action in the selected location?\n"
        "4. Smart Item Classification Tags: Provide structured tags such as:\n"
        "   - Category (e.g., e-waste, plastic, glass, organic, hazardous)\n"
        "   - Material type (e.g., lithium battery, PET plastic, aluminum, mixed)\n"
        "   - Disposal method (e.g., curbside recycling, e-waste drop-off, landfill, special handling)\n"
        "   - Risk level (low / medium / high)\n\n"
        "5. Environmental Impact Score: Estimate environmental impact in a simple format such as:\n"
        "   - CO2 impact (approximate savings or emissions if improperly disposed)\n"
        "   - Pollution risk (low / medium / high)\n"
        "   - Short explanation of environmental consequence if mismanaged"
    )

    description = f"The user is located in: {location}.\n"
    description += f'They described the item as: "{item_description}".' if item_description else "No description provided."

    return system_prompt, description

def validate_inputs(api_key, item_image, location):
    """Validate required inputs."""
    if not api_key or api_key.strip() == "":
        raise gr.Error("🔐 API Key is required.")
    if not item_image:
        raise gr.Error("📷 Please upload image of the item.")
    if not location or location.strip() == "":
        raise gr.Error("🌍 Please enter your region.")

# ------------------------- #
# Main Processing Function
# ------------------------- #

def recyclo_advisor(item_description, item_image, location, api_key, provider):
    """Main advisor logic: processes the image and description using the chosen provider."""
    validate_inputs(api_key, item_image, location)

    try:
        system_prompt, user_text = build_prompt(item_description, location)
        user_content = [{"type": "text", "text": user_text}]

        if item_image:
            img_b64 = image_to_base64(item_image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        if provider == "Nebius":
            client = OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
                api_key=api_key
            )
            response = client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=messages,
                max_tokens=2048,
                temperature=0.6,
                top_p=0.9
            )
            full_response = response.choices[0].message.content.strip()
            result = parse_nebius_response(full_response)

        else:  # Gemini
            client = genai.Client(api_key=api_key)
            prompt = system_prompt + "\n" + user_text

            image_obj = None
            for part in user_content:
                if part["type"] == "image_url":
                    b64_data = part["image_url"]["url"].split(",")[1]
                    image_bytes = base64.b64decode(b64_data)
                    image_obj = Image.open(io.BytesIO(image_bytes))
                    break

            if not image_obj:
                raise ValueError("No image provided for Gemini provider")

            config = types.GenerateContentConfig(response_mime_type="application/json")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_obj, prompt],
                config=config
            )
            result = parse_gemini_response(response.text.strip())

        # Label formatting
        status = result["status"]
        item_name = result["item_name"]

        if "Not Recyclable" in status:
            label = f"❌ {item_name} is Not Recyclable"
        elif "Depends" in status:
            label = f"⚠️ {item_name} recyclability Depends"
        else:
            label = f"✅ {item_name} is Recyclable"

        markdown_summary = f"""
        ### ♻️ **Recyclability Report for `{item_name}`**
        **Recyclability Status:**  
        {status}
        """

        pdf_path = generate_pdf(
            item_name=item_name,
            status=status,
            instructions=result["instructions"],
            reasoning=result["reasoning"],
            classification=result["tags"],
            impact=result["impact"],
        )

        return (
            label,
            markdown_summary.strip(),
            result["instructions"],
            result["reasoning"],
            result["tags"],
            result["impact"],
            pdf_path
        )

    except Exception as e:
        print(f"[Error] {e}")
        return ( "❌ Error calling API Endpoint", "", "", "", "", "", None )

# ----------- #
# Gradio UI
# ----------- #

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ♻️ RecycloBot – AI-Powered Recycling Adviser")
    gr.Markdown("Upload an item photo or describe it to get location-aware recycling guidance.")

    with gr.Row():
        with gr.Column(scale=1):
            provider_dropdown = gr.Dropdown(
                choices=["Nebius", "Gemini"],
                value="Nebius",
                label="🧠 Select API Provider"
            )

            api_key = gr.Textbox(
                label="🔐 API Key (Nebius)",
                placeholder="Paste your API key here",
                type="password"
            )

            def update_label(provider):
                return gr.update(label=f"🔐 API Key ({provider})")

            provider_dropdown.change(fn=update_label, inputs=provider_dropdown, outputs=api_key)

            item_image = gr.Image(label="📷 Upload Image (Optional)", type="pil")

            item_description = gr.Textbox(
                label="📝 Describe the Item (OPTIONAL)",
                placeholder="e.g., 'USB cable', 'Greasy pizza box'"
            )

            location = gr.Textbox(
                label="🌍 Your Region",
                placeholder="Please input your location. e.g., Country and/or state name",
                value="USA"
            )

            submit_btn = gr.Button("🚀 Analyze Item")

            examples = gr.Examples(
                examples=[["broken phone", "img/broken_phone.jpg", "India"],
                          ["charger", "img/charger.jpg", "London, England"]],
                inputs=[item_description, item_image, location],
                label="🧪 Try an Example"
            )

        with gr.Column(scale=1):
            status_output = gr.Label(label="♻️ Item Recyclability Summary")
            summary_output = gr.Markdown(label="📋 Recyclability Report")
            with gr.Accordion("📍 Smart Item Classification Tags", open=False):
                classification = gr.Markdown()
            with gr.Accordion("🌍 Environmental Impact Score", open=False):
                impact = gr.Markdown()
            with gr.Accordion("ℹ️ Instructions on What Exactly to Do with It", open=False):
                instructions_output = gr.Markdown()
            with gr.Accordion("📦 Why This Matters", open=False):
                reasoning_output = gr.Markdown()
            pdf_output = gr.File(label="⬇️ Download PDF Report")

    submit_btn.click(
        fn=recyclo_advisor,
        inputs=[item_description, item_image, location, api_key, provider_dropdown],
        outputs=[status_output, summary_output, instructions_output, reasoning_output, classification, impact, pdf_output]
    )

demo.launch()