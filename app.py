from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import openai
import json
from typing import List
from pydantic import BaseModel
from fireworks.client import Fireworks
import pandas as pd

fireworks_key = "mHqGgCqsjMNbt8uWv3GUt6SXRur1HyZc0hLpxdthDHtzdLHf"
# client = Fireworks(api_key=fireworks_key)
client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = fireworks_key
)

df = pd.read_csv('./data.csv')

class StateResults(BaseModel):
    state_codes: List[str]
    
class CityResults(BaseModel):
    city_names: List[str]

class ZipResults(BaseModel):
    zip_codes: List[str]

def get_states_in_region(region_name):
    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=fireworks_key,
    )
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        response_format={"type": "json_object", "schema": StateResults.schema_json()},
        messages=[
            {
                "role": "user",
                "content": f'Give a JSON list of two-letter state codes in the {region_name} of the US',
            },
        ],
    )

    print(chat_completion)
    return json.loads(chat_completion.choices[0].message.content)

def get_cities_in_region(region_name):
    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=fireworks_key,
    )
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        response_format={"type": "json_object", "schema": CityResults.schema_json()},
        messages=[
            {
                "role": "user",
                "content": f'Give a JSON list of five city names (with state code) in the {region_name} region of the US',
            },
        ],
    )

    print(chat_completion)
    return json.loads(chat_completion.choices[0].message.content)

def get_cities_in_state(state_name):
    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=fireworks_key,
    )
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        response_format={"type": "json_object", "schema": CityResults.schema_json()},
        messages=[
            {
                "role": "user",
                "content": f'Give a JSON list of five city names in the state of {state_name} in the US',
            },
        ],
    )

    print(chat_completion)
    return json.loads(chat_completion.choices[0].message.content)

def get_zips_in_city(city_name):
    return list(df[df['place'].str.contains("Los Angeles", na=False)]['zcta5'])

def get_locations_data(input_geographic_unit, input_geographic_names, output_geographic_unit):    
    if input_geographic_unit == 'REGION':
        print('going from region')
        if output_geographic_unit == 'STATE':
            print('going to state')
            states = []
            for name in input_geographic_names:
                states.extend(get_states_in_region(name)['state_codes'])
            return states
        else:
            print('going to city')
            cities = []
            for name in input_geographic_names:
                cities.extend(get_cities_in_region(name)['city_names'])
            return cities

        # now we need to loop through candidates and get data
    elif input_geographic_unit == 'STATE':
        print('going from state')
        cities = []
        for name in input_geographic_names:
            cities.extend(get_cities_in_state(name)['city_names'])
        return cities
    elif input_geographic_unit == 'CITY':
        print('going from city to zips')
        zips = []
        for name in input_geographic_names:
            zips.extend(get_zips_in_city(name))
        return zips

def get_state_stats(df, geo_output):
    result = {}

    for state in geo_output:
        result[state] = {}
        
        # Calculate the median of the "median_income" column for the filtered DataFrame
        df['household_median_income'] = pd.to_numeric(df['household_median_income'], errors='coerce')
        
        # Optional: Drop rows where 'median_income' is NaN
        df.dropna(subset=['household_median_income'], inplace=True)
        
        # Filter the DataFrame, treating NA values in 'place' as False
        filtered_df = df[df['state'].str.contains(state, na=False)]
        
        # Calculate the median of the 'median_income' column for the filtered DataFrame
        median_income = filtered_df['household_median_income'].median()
        
        result[state]['median_income'] = median_income

    return result

def get_city_stats(df, geo_output):
    result = {}

    for city in geo_output:
        city = city.split(',')[0]
        result[city] = {}
        
        # Calculate the median of the "median_income" column for the filtered DataFrame
        df['household_median_income'] = pd.to_numeric(df['household_median_income'], errors='coerce')
        
        # Optional: Drop rows where 'median_income' is NaN
        df.dropna(subset=['household_median_income'], inplace=True)
        
        # Filter the DataFrame, treating NA values in 'place' as False
        filtered_df = df[df['place'].str.contains(city, na=False)]
        
        # Calculate the median of the 'median_income' column for the filtered DataFrame
        median_income = filtered_df['household_median_income'].median()
        
        result[city]['median_income'] = median_income

    return result

def get_zip_stats(df, geo_output):
    result = {}

    for zip in geo_output:
        result[zip] = {}
        
        # Calculate the median of the "median_income" column for the filtered DataFrame
        df['household_median_income'] = pd.to_numeric(df['household_median_income'], errors='coerce')
        
        # Optional: Drop rows where 'median_income' is NaN
        df.dropna(subset=['household_median_income'], inplace=True)
        
        # Filter the DataFrame, treating NA values in 'place' as False
        filtered_df = df[df['zcta5'].astype(str).str.contains(str(zip), na=False)]
        
        # Calculate the median of the 'median_income' column for the filtered DataFrame
        median_income = filtered_df['household_median_income'].median()
        
        result[zip]['median_income'] = median_income

    return result

def get_chat_text(message, output):
    client = Fireworks(api_key=fireworks_key)

    response = client.chat.completions.create(
      model="accounts/fireworks/models/llama-v2-7b-chat",
      messages=[{
        "role": "user",
        "content": f"The user asked `{message}`. They are especially concerned with income. Use this data to help them: {str(output)}",
      }],
    )
    
    text_output = response.choices[0].message.content

    return text_output

def handle_message(message):
    print('handling message....')
    tools = [    
        {
            "type": "function",
            "function": {
                "name": "get_locations_data",
                "description": "Get data for zip codes, cities, or states in a particular geographic region.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_geographic_unit": {
                            "type": "string",
                            "enum": ["REGION", "STATE", "CITY"],
                            "description": "The input unit, which can be REGION, STATE or CITY."
                        },
                        "input_geographic_names": {
                            "type": "array",
                            "description": "An array of input geographic names - either a REGION name, an array of two-letter STATE codes, or CITY names."
                        },
                        "output_geographic_unit": {
                            "type": "string",
                            "enum": ["STATE", "CITY", "ZIP"],
                            "description": "The outputted geographic unit - either STATE, CITY, or ZIP. If the user does not specify, we assume if the input_geographic_unit is region, this value will be state, and if the input_geographic_unit is state, this will be city, and if the input_geographic_unit is city, this will be zip."
                        }
                    },
                    "required": ["input_geographic_unit", "input_geographic_names", "output_geographic_unit"],
                },
            },
        }
    ]
    
    messages = [
        {"role": "system", "content": f"You are a helpful US real estate assistant with access to functions. Use them if required."},
        {"role": "user", "content": message}
    ]

    print('running chat completion to get function....')
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/firefunction-v1",
        messages=messages,
        tools=tools,
        temperature=0.1
    )
    print(chat_completion)
    output = json.loads(chat_completion.choices[0].message.model_dump_json())
    
    if output['tool_calls']:
        function = output['tool_calls'][0]['function']
        geo_output = globals()[function['name']](**json.loads(function['arguments']))
    
        output_unit = json.loads(function['arguments'])['output_geographic_unit']
    
        if output_unit == 'STATE':
            output = get_state_stats(df, geo_output)
        if output_unit == 'CITY':
            output = get_city_stats(df, geo_output)
        if output_unit == 'ZIP':
            output = get_zip_stats(df, geo_output)
    
        text = get_chat_text(message, output)

        return {"text": text, "json_output": output}

#######################################################################################################

app = Flask(__name__)
CORS(app) 

@app.route('/question', methods=['POST'])
def handle_question():
    data = json.loads(request.data.decode('utf-8'))
    print('data:', data)

    response = handle_message(data['question'])
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
