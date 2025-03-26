import os
import json
import base64
import requests
import logging
from appwrite.client import Client
from appwrite.services.databases import Databases
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Appwrite client
client = Client()
client.set_endpoint("https://cloud.appwrite.io/v1")
client.set_project(os.getenv("APPWRITE_PROJECT_ID"))
client.set_key(os.getenv("APPWRITE_API_KEY"))
databases = Databases(client)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro-vision")

def download_image_as_base64(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except requests.RequestException as e:
        logger.error(f"Failed to download image: {str(e)}")
        return None

def check_content(task_description, text_content, image_url):
    prompt = f"""
    This is the description of the task to be performed: {task_description}

    Below are the content and images that need to be moderated:
    - Content: {text_content}

    Please check if the content and poster are suitable for the task.
    If not, please explain the reason.
    """

    parts = [prompt]

    if image_url:
        base64_image = download_image_as_base64(image_url)
        if base64_image:
            parts.append({"image": {"base64": base64_image}})

    try:
        response = model.generate_content(
            parts,
            safety_settings=[
                (HarmCategory.HARM_CATEGORY_HATE_SPEECH, HarmBlockThreshold.BLOCK_NONE),
                (HarmCategory.HARM_CATEGORY_HARASSMENT, HarmBlockThreshold.BLOCK_NONE),
                (HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmBlockThreshold.BLOCK_NONE),
                (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, HarmBlockThreshold.BLOCK_NONE),
            ]
        )
        return response.text if hasattr(response, 'text') else "AI analysis failed."
    except Exception as e:
        logger.error(f"AI processing error: {str(e)}")
        return "AI analysis encountered an error."

def handler(req):
    try:
        req_body = json.loads(req)
        logger.info(f"Event Data: {json.dumps(req_body, indent=2)}")

        doc_id = req_body["payload"]["$id"]
        collection_id = req_body["payload"]["$collectionId"]
        db_id = req_body["payload"]["$databaseId"]

        task_description = req_body["payload"].get("task", "").strip()
        text_content = req_body["payload"].get("text", "").strip()
        image_url = req_body["payload"].get("image_url", "").strip()

        if not task_description:
            logger.warning(f"Document {doc_id} has no task description, skipping.")
            return json.dumps({"status": "skipped", "reason": "missing task description"})

        logger.info(f"Processing document: {doc_id}")

        ai_response = check_content(task_description, text_content, image_url)
        logger.info(f"AI Response: {ai_response}")

        try:
            databases.update_document(db_id, collection_id, doc_id, {
                "ai_analysis": ai_response
            })
            logger.info(f"Document {doc_id} updated successfully.")
        except Exception as db_error:
            logger.error(f"Database Update Error: {str(db_error)}")
            return json.dumps({"status": "error", "message": "Failed to update database"})

        return json.dumps({"status": "success", "ai_response": ai_response})

    except Exception as e:
        logger.error(f"Handler Error: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})
