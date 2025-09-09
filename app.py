import openai
import ast
import re
import pandas as pd
import json
from IPython.display import display, HTML
from flask import Flask, redirect, url_for, render_template, request
from functions import *

openai.api_key = open('OPENAI_API_Key.txt', 'r').read().strip()

app = Flask(__name__)

conversation_bot = []

conversation = initialize_conversation()
introduction = get_chat_completions(conversation)
conversation_bot.append({'bot': introduction})
top_3_laptops = None


@app.route("/")
def default_func():
    global conversation_bot, conversation, top_3_laptops
    return render_template("index.html", name_xyz = conversation_bot)

@app.route("/end_conv", methods = ['POST', 'GET'])
def end_conv():
    global conversation_bot, conversation, top_3_laptops
    conversation_bot = []
    conversation = initialize_conversation()
    introduction = get_chat_completions(conversation)
    conversation_bot.append({'bot': introduction})
    top_3_laptops = None
    return redirect(url_for('default_func'))

@app.route("/invite", methods = ['POST'])
def invite():
    global conversation_bot, conversation, top_3_laptops, conversation_reco, laptop_list

    user_input = request.form["user_input_message"].strip()
    prompt = (
        ". Remember that you are a intelligent laptop shopping assistant. "
        "You should help and answer only with the queries related to laptops. "
        "If the queries are not related to laptops, just say something like you can help only with queries related to laptops etc."
    )

    # Moderation check for user input
    if moderation_check(user_input) == 'Flagged':
        display("Sorry, this message has been flagged. Please restart your conversation.")
        return redirect(url_for('end_conv'))

    # Check if last bot message was a confirmation question
    if conversation_bot and "did i get all your requirements" in conversation_bot[-1].get('bot', '').lower():
        answer = user_input.lower()
        conversation.append({"role": "user", "content": answer})
        conversation_bot.append({'user': answer})
        if answer == "yes":
            # Proceed to fetch recommendations
            conversation_bot.append({'bot': "Thank you for confirming!"})
            # You should have stored the last valid requirements somewhere; here we use the last assistant message
            if is_laptop_recommendation_list(laptop_list):
                top_3_laptops = laptop_list
                print("Top 3 laptops selected:", type(top_3_laptops), len(top_3_laptops), top_3_laptops)
                validated_reco = recommendation_validation(top_3_laptops)
                if not validated_reco:
                    conversation_bot.append({'bot': "Sorry, we do not have laptops that match your requirements."})
                else:
                    conversation_reco = initialize_conv_reco(validated_reco)
                    conversation_reco.append({"role": "user", "content": "This is my user profile" + str(validated_reco)})
                    recommendation = get_chat_completions(conversation_reco)
                    if moderation_check(recommendation) == 'Flagged':
                        display("Sorry, this message has been flagged. Please restart your conversation.")
                        return redirect(url_for('end_conv'))
                    conversation_reco.append({"role": "assistant", "content": str(recommendation)})
                    conversation_bot.append({'bot': recommendation})
            else:
                conversation_bot.append({'bot': "Sorry, I could not process your requirements. Please try again."})
        elif answer == "no":
            conversation_bot.append({'bot': "Sorry about that. Please specify your requirements again."})
        else:
            conversation_bot.append({'bot': "Please answer with Yes or No."})
        return redirect(url_for('default_func'))

    # 3. Main flow: get assistant response and check for recommendations
    if top_3_laptops is None:
        conversation.append({"role": "user", "content": user_input + prompt})
        conversation_bot.append({'user': user_input})
        response_assistant = get_chat_completions(conversation)
        if moderation_check(response_assistant) == 'Flagged':
            display("Sorry, this message has been flagged. Please restart your conversation.")
            return redirect(url_for('end_conv'))
        

        parsed_response = string_to_list(response_assistant)
        if is_laptop_recommendation_list(parsed_response):
            laptop_list = extract_laptop_list(response_assistant)
            if laptop_list:
                formatted = format_laptop_recommendations(laptop_list)
                conversation_bot.append({'bot': formatted})
                conversation_bot.append({'bot': "Did I get all your requirements correctly? Please answer with a Yes or No."})
            else:
                conversation_bot.append({'bot': response_assistant})
        else:
            conversation.append({"role": "assistant", "content": str(response_assistant)})
            conversation_bot.append({'bot': response_assistant})
    else:
        # If recommendations already exist, continue the conversation
        print("Top 3 laptops already present:", top_3_laptops)
        conversation_reco.append({"role": "user", "content": user_input})
        conversation_bot.append({'user': user_input})

        response_asst_reco = get_chat_completions(conversation_reco)
        if moderation_check(response_asst_reco) == 'Flagged':
            print("Sorry, this message has been flagged. Please restart your conversation.")
            return redirect(url_for('end_conv'))

        conversation.append({"role": "assistant", "content": response_asst_reco})
        conversation_bot.append({'bot': response_asst_reco})

    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

    