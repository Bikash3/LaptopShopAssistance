# Laptop Recommendation Chat Assistant
> Conversational assistant that collects user requirements and recommends laptops from a local catalogue. Built with Flask and a chat service connector.

## Table of Contents
* [General Info](#general-information)
* [Stepwise Process](#stepwise-process)
  * [Step 1: Objective](#step-1---objective)
  * [Step 2: Requirements and Setup](#step-2---requirements-and-setup)
  * [Step 3: Project Structure and Key Files](#step-3---project-structure-and-key-files)
  * [Step 4: Conversation and Function Flow](#step-4---conversation-and-function-flow)
  * [Step 5: Endpoints and Usage](#step-5---endpoints-and-usage)
  * [Step 6: Data and Validation](#step-6---data-and-validation)
  * [Step 7: Moderation and Safety](#step-7---moderation-and-safety)
  * [Step 8: Testing and Debugging](#step-8---testing-and-debugging)
  * [Step 9: Deployment Notes](#step-9---deployment-notes)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Info
> This project implements a conversational laptop shopping assistant. The assistant interacts with a user to extract a small profile (GPU intensity, display quality, portability, multitasking, processing speed, budget) and returns ranked laptop recommendations from a local CSV catalogue.

Primary outputs:
- Interactive web UI (Flask) for chat.
- Top-3 laptop recommendations (JSON / HTML table).
- Logged conversation in memory for the session.

## Stepwise Process

### Step 1 - Objective
- Collect a concise user profile through conversation.
- Map the profile to a small set of categorical requirements (low / medium / high) and a numeric budget.
- Filter and rank laptops from updated_laptop.csv and present top matches with provenance.

### Step 2 - Requirements and Setup
1. Files required (place in project root):
   - OPENAI_API_Key.txt — single-line API key.
   - updated_laptop.csv — catalogue with a `laptop_feature` column (stringified dict) and `Price`.
   - templates/index.html — front-end chat UI expected by Flask.
2. Python environment:
   - Install: flask, openai, pandas, ipython, python-dotenv (optional), spacy if required by other code.
   - Example (Windows PowerShell):
     - python -m venv .venv
     - .venv\Scripts\Activate.ps1
     - pip install -r requirements.txt
3. Start the app:
   - python app.py
   - Default host: 0.0.0.0, debug=True (adjust for production).

### Step 3 - Project Structure and Key Files
- app.py — Flask application and route handlers for chat flow:
  - "/" : renders chat interface.
  - "/invite" : receives user messages and advances conversation.
  - "/end_conv" : resets conversation state.
- functions.py — helper logic and connectors:
  - initialize_conversation() : system prompt with extraction instructions.
  - get_chat_completions() : wrapper around chat API and function-calling logic.
  - compare_laptops_with_user() : filters and scores laptops from CSV.
  - recommendation_validation(), format_laptop_recommendations(), extract_laptop_list(), string_to_list() : utilities.
- updated_laptop.csv — product data (Price, laptop_feature, other specs).
- OPENAI_API_Key.txt — API key file.
- templates/index.html — front-end (not provided in repo by code listing).

### Step 4 - Conversation and Function Flow
1. Conversation initialization:
   - System message guides the assistant to ask clarifying questions and fill a 6-key profile dictionary.
2. User messages are appended to the conversation list and sent to get_chat_completions().
3. Function calling:
   - A function schema (get_laptop_recommendation) is provided to the chat service.
   - When the model returns a call to that function, app code invokes compare_laptops_with_user() using parsed arguments.
   - The function returns top-k rows as JSON (max 3 recommended laptops).
4. Post-processing:
   - The app formats the returned list into HTML table and asks the user to confirm ("Did I get all your requirements correctly?").
   - On confirmation, a secondary conversation is started to craft final descriptions for the selected products.

### Step 5 - Endpoints and Usage
- GET / : open chat UI (index.html). The UI should POST user messages to /invite.
- POST /invite : processes user message, performs moderation, calls chat completions, and returns redirected view of conversation.
- POST or GET /end_conv : resets conversation and returns to initial state.

Typical user interaction:
1. User enters requirements (use case, preferences, budget).
2. Assistant asks clarifying questions as needed.
3. Assistant returns a compact list of candidate laptops and asks for confirmation.
4. Upon confirmation, assistant summarizes selected products with specs and prices.

### Step 6 - Data and Validation
- updated_laptop.csv must include:
  - Price column (string numeric with optional commas).
  - laptop_feature column containing a stringified dictionary mapping the named keys to 'low'/'medium'/'high' values.
  - Other spec columns used in formatted display.
- Scoring:
  - Budget filter applied first.
  - For each matched laptop, keys (excluding Budget) are compared: laptop value meeting or exceeding user requirement increments score.
  - Top laptops are sorted by Score and top 3 are returned.
- Validation:
  - recommendation_validation() keeps only entries with Score > 2 before finalizing recommendations.

### Step 7 - Moderation and Safety
- All user inputs and assistant outputs pass through moderation_check() using the moderation API.
- Flagged content aborts the conversation and prompts restart.
- System prompt enforces restricted, deterministic value ranges for profile keys (low/medium/high) and numeric budget threshold.

### Step 8 - Testing and Debugging
- Local tests:
  - Verify OPENAI_API_Key.txt is present and valid.
  - Run app.py and interact via the provided templates/index.html.
  - Check console logs for printed objects (top_3_laptops, parsed lists).
- Function-level checks:
  - Call compare_laptops_with_user() with sample user requirement string to verify JSON output.
  - Use string_to_list() and extract_laptop_list() to validate parsing logic for various model output formats.
- Edge cases:
  - Non-numeric or malformed budget strings.
  - Missing laptop_feature keys in CSV rows — compare_laptops_with_user() handles missing keys by fallback.

### Step 9 - Deployment Notes
- Remove debug=True and bind to a production WSGI server (gunicorn / waitress) for deployment.
- Secure the API key (use environment variables or a secrets manager instead of a plaintext file).
- Add rate limiting and caching to reduce repeated calls to the chat service.
- Sanitize and validate uploaded catalogue files before use in production.

## Technologies Used
- Python
- Flask
- openai Python client
- pandas
- IPython.display (used in development)
- HTML templates for front-end

## Acknowledgements
- Implementation based on local catalogue matching and conversational extraction patterns.
- Utilities adapted to handle variable model output formats and function-calling schema.

## Contact
### Created by
  * Bikash Sarkar

// ...existing code...