from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import urllib.request

MODEL_URL = "https://github.com/Aastha2785/GoodKitchen/releases/download/v1.0/best_model.pth"
MODEL_PATH = "best_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 15)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

inventory = []

@app.route('/')
def index():
    return render_template('index.html', inventory=inventory)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
        vegetable = class_names[pred]

    if vegetable not in inventory:
        inventory.append(vegetable)

    return jsonify({'vegetable': vegetable, 'inventory': inventory})

@app.route('/add_manual', methods=['POST'])
def add_manual():
    data = request.get_json()
    items = data.get('items', [])
    for item in items:
        if item and item not in inventory:
            inventory.append(item)
    return jsonify({'inventory': inventory})

@app.route('/remove', methods=['POST'])
def remove():
    data = request.get_json()
    item = data.get('item', '')
    if item in inventory:
        inventory.remove(item)
    return jsonify({'inventory': inventory})

@app.route('/suggest', methods=['POST'])
def suggest():
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    data = request.get_json()
    items = data.get('inventory', inventory)

    if not items:
        return jsonify({'recipes': 'Please add some vegetables first!'})

    vegetables = ', '.join(items)
    prompt = f"""I have these vegetables: {vegetables}. 
    Suggest 3 North Indian and 2 South Indian recipes I can make with these.
    For each recipe mention: Recipe name, which vegetables from my list it uses.
    Keep it brief and friendly."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    recipes = response.choices[0].message.content
    return jsonify({'recipes': recipes})

@app.route('/recipe_search', methods=['POST'])
def recipe_search():
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    data = request.get_json()
    recipe_name = data.get('recipe_name', '')
    user_inventory = data.get('inventory', inventory)

    if not recipe_name:
        return jsonify({'result': 'Please enter a recipe name!'})

    prompt = f"""For the recipe "{recipe_name}", list ALL ingredients needed with quantities for 2 servings:

1. Vegetables required
2. Spices & masalas  
3. Dairy & oils
4. Other ingredients (lentils, rice, flour etc.)

After the full list, you MUST write this line (mandatory, do not skip):
INGREDIENTS_LIST: ingredient1, ingredient2, ingredient3, ingredient4, ingredient5...

The INGREDIENTS_LIST line must contain every single ingredient as lowercase words separated by commas, no quantities, no symbols."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = response.choices[0].message.content
    print("RAW:", result[-500:])

    ingredients = []
    missing = []

    for line in result.split('\n'):
        line_stripped = line.strip()
        if 'INGREDIENTS_LIST:' in line_stripped:
            raw = line_stripped.split('INGREDIENTS_LIST:')[1].strip()
            ingredients = [i.strip().lower() for i in raw.split(',') if i.strip()]
            inv_lower = [i.lower() for i in user_inventory]
            missing = [i for i in ingredients if not any(i in inv_item or inv_item in i for inv_item in inv_lower)]
            break

    display = '\n'.join([l for l in result.split('\n') if 'INGREDIENTS_LIST:' not in l])

    print("MISSING:", missing)

    return jsonify({'result': display, 'missing': missing})
@app.route('/full_recipe', methods=['POST'])
def full_recipe():
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    
    data = request.get_json()
    recipe_name = data.get('recipe_name', '')
    
    prompt = f"""Give me a detailed step by step recipe for "{recipe_name}".
    Include:
    1. Ingredients with exact quantities
    2. Step by step cooking instructions
    3. Cooking time
    4. Serving suggestions
    Keep it clear and easy to follow."""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return jsonify({'result': response.choices[0].message.content})

@app.route('/clear', methods=['POST'])
def clear():
    inventory.clear()
    return jsonify({'inventory': inventory})

if __name__ == '__main__':
    app.run(debug=True)