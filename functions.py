import openai
import ast
import re
import pandas as pd
import json
from IPython.display import display, HTML

# Add Function Schema and Mappings for function calling
function_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_laptop_recommendation",
            "description": "Get laptop recommendations based on user requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "GPU_intensity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "Display_quality": {"type": "string", "enum": ["low", "medium", "high"]},
                    "Portability": {"type": "string", "enum": ["low", "medium", "high"]},
                    "Multitasking": {"type": "string", "enum": ["low", "medium", "high"]},
                    "Processing_speed": {"type": "string", "enum": ["low", "medium", "high"]},
                    "Budget": {"type": "string"}
                },
                "required": [
                    "GPU_intensity", "Display_quality", "Portability",
                    "Multitasking", "Processing_speed", "Budget"
                ]
            }
        }
    }
]

def get_laptop_recommendation(
    GPU_intensity, Display_quality, Portability, Multitasking, Processing_speed, Budget
):
    user_req = {
        "GPU intensity": GPU_intensity,
        "Display quality": Display_quality,
        "Portability": Portability,
        "Multitasking": Multitasking,
        "Processing speed": Processing_speed,
        "Budget": Budget
    }
    return compare_laptops_with_user(str(user_req))

function_map = {
    "get_laptop_recommendation": get_laptop_recommendation
}

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    example_user_dict = {'GPU intensity': "high",
                        'Display quality':"high",
                        'Portability': "low",
                        'Multitasking': "high",
                        'Processing speed': "high",
                        'Budget': "150000"}

    example_user_req = {'GPU intensity': "_",
                        'Display quality': "_",
                        'Portability': "_",
                        'Multitasking': "_",
                        'Processing speed': "_",
                        'Budget': "_"}

    system_message = f"""
    You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this
    {{'GPU intensity': 'values','Display quality': 'values','Portability': 'values','Multitasking': 'values','Processing speed': 'values','Budget': 'values'}}
    The value for 'Budget' should be a numerical value extracted from the user's response.
    The values for all keys, except 'Budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    All the values in the example dictionary are only representative values.
    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised:
    - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    - The value for 'Budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range.
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    Follow the chain-of-thoughts below and only output the final updated python dictionary for the keys as described in {example_user_req}. \n
    {delimiter}
    Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear. Ask followup questions to understand their needs.
    You are trying to fill the values of all the keys {{'GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget'}} in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    If the necessary information has been extracted, only then proceed to the next step. \n
    Otherwise, rephrase the question to capture their profile clearly. \n

    {delimiter}
    Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys.
    Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.
    {delimiter}

    {delimiter}
    Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions.
    {delimiter}

    {delimiter}
    Here is a sample conversation between the user and assistant:
    User: "Hi, I am an editor."
    Assistant: "Great! As an editor, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
    User: "I primarily work with After Effects."
    Assistant: "Thank you for providing that information. Working with After Effects involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
    User: "Yes, sometimes I work with 4K videos as well."
    Assistant: "Thank you for the information. Processing 4K vidoes will require a good processor and high GPU. I think we have already determined earlier that you need a high GPU. To ensure I have a complete understanding of your needs, I have one more question: Are you frequently on the go and require a laptop that is lightweight and easy to carry, or do you primarily work from a stationary location?"
    User: "Yes, sometimes I travel but do not carry my laptop."
    Assistant:"Could you kindly let me know your budget for the laptop? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 1.5lakh inr"
    Assistant: "{example_user_dict}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation

def get_chat_completions(input, json_format = False):
    MODEL = 'gpt-3.5-turbo'
    system_message_json_output = """<<. Return output in JSON format to the key output.>>"""
    # If the output is required to be in JSON format
    if json_format == True:
        # Append the input prompt to include JSON response as specified by OpenAI
        input[0]['content'] += system_message_json_output

        # JSON return type specified
        chat_completion_json = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            response_format={ "type": "json_object" },
            tools=function_schemas,
            tool_choice="auto",
            seed = 1234
        )

        message_json = chat_completion_json.choices[0].message
        if hasattr(message_json, 'tool_calls') and message_json.tool_calls:
            function_name = message_json.tool_calls[0].function.name
            function_args = json.loads(message_json.tool_calls[0].function.arguments)
            output = function_name(**function_args)
        elif message_json.content:
            output = message_json.content
        else:
            output = "Sorry, I couldn't process your request. Please try again."
        output = json.loads(output)  # Convert the JSON string to a Python dictionary
    # No JSON return type specified
    else:
        chat_completion = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            tools = function_schemas,
            tool_choice="auto",
            seed = 1234
        )

        message = chat_completion.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            function_name = message.tool_calls[0].function.name
            function_args = json.loads(message.tool_calls[0].function.arguments)
            output = function_map[function_name](**function_args)
        elif message.content:
            output = message.content
        else:
            output = "Sorry, I couldn't process your request. Please try again."

    return output

def iterate_llm_response(funct, debug_response, num = 10):
    """
    Calls a specified function repeatedly and prints the results.
    This function is designed to test the consistency of a response from a given function.
    It calls the function multiple times (default is 10) and prints out the iteration count,
    the function's response(s).
    Args:
        funct (function): The function to be tested. This function should accept a single argument
                          and return the response value(s).
        debug_response (dict): The input argument to be passed to 'funct' on each call.
        num (int, optional): The number of times 'funct' will be called. Defaults to 10.
    Returns:
        This function only returns the results to the console.
    """
    i = 0  # Initialize counter

    while i < num:  # Loop to call the function 'num' times

        response = funct(debug_response)  # Call the function with the debug response

        # Print the iteration number, result, and reason from the response
        print("Iteration: {0}".format(i))
        print(response)
        print('-' * 50)  # Print a separator line for readability
        i += 1  # Increment the counter

def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    response = openai.moderations.create(input=user_input)

    # Extract the moderation result from the API response.
    moderation_output = response.results[0].flagged

    # Check if the input was flagged by the moderation system.
    if response.results[0].flagged == True:
        # If flagged, return "Flagged"
        return "Flagged"
    else:
        # If not flagged, return "Not Flagged"
        return "Not Flagged"

def compare_laptops_with_user(user_req_string):
    laptop_df = pd.read_csv('updated_laptop.csv')
    # convert string value of user_req_string to json and store in user_requirements
    user_requirements = ast.literal_eval(user_req_string)
    # Extracting user requirements from the input string (assuming it's a dictionary)
    # Since the function parameter already seems to be a string, we'll use it directly instead of extracting from a dictionary

    # Extracting the budget value from user_requirements and converting it to an integer
    budget_raw = user_requirements.get('Budget', '0').replace(',', '').split()[0]
    if budget_raw == '_':
        budget = 50000
    else:
        try:
            budget = int(budget_raw)
        except ValueError:
            budget = 50000

    # Creating a copy of the DataFrame and filtering laptops based on the budget
    filtered_laptops = laptop_df.copy()
    filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',', '').astype(int)
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()
    # filtered_laptops
    # Mapping string values 'low', 'medium', 'high' to numerical scores 0, 1, 2
    mappings = {'low': 0, 'medium': 1, 'high': 2}

    # Creating a new column 'Score' in the filtered DataFrame and initializing it to 0
    filtered_laptops['Score'] = 0

    # Iterating over each laptop in the filtered DataFrame to calculate scores based on user requirements
    for index, row in filtered_laptops.iterrows():
        user_product_match_str = row['laptop_feature']
        laptop_values = user_product_match_str
        laptop_values = ast.literal_eval(user_product_match_str)
        score = 0

        # Comparing user requirements with laptop features and updating scores
        for key, user_value in user_requirements.items():
            if key == 'Budget':
                continue  # Skipping budget comparison
            laptop_value = laptop_values.get(key, None)
            laptop_mapping = mappings.get(laptop_value, -1)
            user_mapping = mappings.get(user_value, -1)
            if laptop_mapping >= user_mapping:
                score += 1  # Incrementing score if laptop value meets or exceeds user value

        filtered_laptops.loc[index, 'Score'] = score  # Updating the 'Score' column in the DataFrame

    # Sorting laptops by score in descending order and selecting the top 3 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)
    top_laptops_json = top_laptops.to_json(orient='records')  # Converting the top laptops DataFrame to JSON format

    # top_laptops
    return top_laptops_json

def recommendation_validation(laptop_recommendation):
    if isinstance(laptop_recommendation, list):
        data = laptop_recommendation
    else:
        data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1

def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue in the user message \
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

    """
    user_message = f""" These are the user's products: {products}"""
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]
    return conversation

def string_to_list(s):
    # If already a list, return as is
    if isinstance(s, list):
        return s
    # Try JSON first (expects double quotes)
    try:
        result = json.loads(s)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    # Try ast.literal_eval (handles single quotes, Python dict/list)
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    # If all fails, return empty list
    return []

def is_laptop_recommendation_list(data):
    required_keys = {"Brand", "Model Name", "Price"}
    return (
        isinstance(data, list) and
        len(data) > 0 and
        isinstance(data[0], dict) and
        required_keys.issubset(data[0].keys())
    )
def format_laptop_recommendations(laptop_list):
    if not laptop_list:
        return "<p>No recommendations found.</p>"
    html = '<table border="1"><tr>'
    # Show only a few key columns
    columns = ["Brand", "Model Name", "CPU Manufacturer", "Core", "RAM Size", "Storage Type", "Display Size", "Graphics Processor", "OS", "Price", "Score"]
    for col in columns:
        html += f"<th>{col}</th>"
    html += "</tr>"
    for laptop in laptop_list:
        html += "<tr>"
        for col in columns:
            html += f"<td>{laptop.get(col, '')}</td>"
        html += "</tr>"
    html += "</table>"
    return html
def extract_laptop_list(response_str):
    # Try to find the first list in the string using regex
    match = re.search(r'(\[.*\])', response_str, re.DOTALL)
    if match:
        list_str = match.group(1)
        try:
            # Try JSON first
            return json.loads(list_str)
        except Exception:
            try:
                # Fallback to Python literal
                return ast.literal_eval(list_str)
            except Exception:
                return []
    return []
