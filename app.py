from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)

# Initialize LLM
API_KEY = os.getenv("MISTRAL_API_KEY", "lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc")
llm = ChatMistralAI(api_key=API_KEY, temperature=0.1)
parser = JsonOutputParser()

# Emotion detection prompt template
EMOTION_PROMPT_TEMPLATE = """
Analyze the following text and classify it into **all possible emotions** you detect 
(not just happy/sad/angry). Consider nuanced emotions like contentment, anticipation, 
curiosity, trust, disgust, surprise, fear, etc.

IMPORTANT RULES:
1. If no emotion is detected, return: {{"message": "No, I cannot find emotions."}}
2. Otherwise, return ONLY a JSON object with emotions as keys and percentages as values
3. Percentages MUST be integers that sum to exactly 100
4. Include 2-6 emotions maximum for better visualization

Example format:
{{
  "sadness": 35,
  "hope": 25,
  "fear": 15,
  "anger": 15,
  "excitement": 10
}}

Text: "{text}"

JSON Response:
"""

def detect_emotions(text):
    """Emotion detection using LLM with your prompt template"""
    if not text or not text.strip():
        return {"message": "No, I cannot find emotions."}
    
    prompt = PromptTemplate(template=EMOTION_PROMPT_TEMPLATE, input_variables=["text"])
    chain = prompt | llm | parser
    
    try:
        emotions = chain.invoke({"text": text})
        
        # Handle "no emotions" message
        if "message" in emotions:
            return emotions
        
        # Clean and validate emotions
        clean_emotions = {}
        for emotion, value in emotions.items():
            try:
                clean_emotions[emotion.lower()] = int(float(value))
            except:
                continue
        
        if not clean_emotions:
            return {"message": "No, I cannot find emotions."}
            
        # Ensure sum to exactly 100
        total = sum(clean_emotions.values())
        if total > 0 and total != 100:
            factor = 100.0 / total
            clean_emotions = {k: round(v * factor) for k, v in clean_emotions.items()}
            
            # Handle rounding errors to ensure exact sum of 100
            diff = 100 - sum(clean_emotions.values())
            if diff != 0:
                max_emotion = max(clean_emotions, key=clean_emotions.get)
                clean_emotions[max_emotion] += diff
        
        return clean_emotions
        
    except Exception as e:
        print(f"LLM Error: {str(e)}")  # Debug print
        # Fallback: try to extract JSON from response manually
        try:
            response_text = str(e)
            if "Invalid json output:" in response_text:
                # Extract the part after "Invalid json output:"
                json_part = response_text.split("Invalid json output:")[1].split("For troubleshooting")[0].strip()
                # Try to find JSON in the text
                import json
                import re
                json_match = re.search(r'\{.*\}', json_part, re.DOTALL)
                if json_match:
                    emotions = json.loads(json_match.group())
                    if "message" in emotions:
                        return emotions
                    return emotions
        except:
            pass
        
        return {"message": "No, I cannot find emotions."}

def create_pie_chart(emotions):
    """Create enhanced pie chart for multiple emotions"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a more diverse color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
    
    wedges, texts, autotexts = ax.pie(
        emotions.values(), 
        labels=emotions.keys(), 
        autopct='%1.1f%%',
        colors=colors, 
        startangle=90,
        explode=[0.05] * len(emotions),  # Slightly separate each slice
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Emotion Distribution - Pie Chart", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_bar_chart(emotions):
    """Create enhanced bar chart for multiple emotions"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort emotions by percentage for better visualization
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_emotions)))
    
    bars = ax.bar(
        range(len(sorted_emotions)), 
        sorted_emotions.values(), 
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_emotions.values())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, 
                f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize axes
    ax.set_xticks(range(len(sorted_emotions)))
    ax.set_xticklabels([emotion.title() for emotion in sorted_emotions.keys()], 
                       rotation=45, ha='right', fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    ax.set_title("Emotion Distribution - Bar Chart", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(sorted_emotions.values()) * 1.15)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page"""
    emotions = {}
    charts = {}
    error_message = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        
        if not text:
            error_message = "Please enter some text"
        else:
            # Detect emotions
            emotions = detect_emotions(text)
            
            # Generate charts if emotions found
            if "message" not in emotions and emotions:
                try:
                    charts["pie"] = fig_to_base64(create_pie_chart(emotions))
                    charts["bar"] = fig_to_base64(create_bar_chart(emotions))
                except Exception as e:
                    error_message = f"Error generating charts: {str(e)}"

    return render_template("index.html", 
                         emotions=emotions, 
                         charts=charts, 
                         error_message=error_message)

@app.route("/api/emotions", methods=["POST"])
def api_emotions():
    """API endpoint for emotion detection"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        emotions = detect_emotions(text)
        return jsonify({"emotions": emotions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)