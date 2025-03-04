import os
import json
from appwrite.client import Client
from appwrite.services.databases import Databases
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv

dotenv.load_dotenv()

client = Client()
client.set_endpoint("https://cloud.appwrite.io/v1")
client.set_project(os.getenv("APPWRITE_PROJECT_ID"))
client.set_key(os.getenv("APPWRITE_API_KEY"))

databases = Databases(client)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro-vision")

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
        parts.append({"image": {"url": image_url}})

    response = model.generate_content(
        parts,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    return response.text if response.text else "Unable to analyze content."

def handler(req):
    try:
        req_body = json.loads(req)
        print(f"Event Data: {json.dumps(req_body, indent=2)}")  

        doc_id = req_body["payload"]["$id"]
        collection_id = req_body["payload"]["$collectionId"]
        db_id = req_body["payload"]["$databaseId"]

        task_description = req_body["payload"].get("task", "").strip()
        text_content = req_body["payload"].get("text", "").strip()
        image_url = req_body["payload"].get("image_url", "").strip()  

        if not task_description:
            print(f"Document {doc_id} has no task description, skipping.")
            return json.dumps({"status": "skipped", "reason": "missing task description"})

        print(f"Received document: {doc_id}")
        print(f"Task: {task_description}")
        print(f"Text: {text_content}")
        print(f"Image URL: {image_url}")

        ai_response = check_content(task_description, text_content, image_url)
        print(f"AI Response: {ai_response}")

        try:
            databases.update_document(db_id, collection_id, doc_id, {
                "ai_analysis": ai_response
            })
            print(f"Document {doc_id} updated successfully.")
        except Exception as db_error:
            print(f"Database Update Error: {str(db_error)}")
            return json.dumps({"status": "error", "message": "Failed to update database"})

        return json.dumps({"status": "success", "ai_response": ai_response})

    except Exception as e:
        print(f"Error: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})
