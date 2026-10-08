from flask import Flask, render_template, request
from playwright.sync_api import sync_playwright
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai
import torch

app = Flask(__name__)


# =========================================================
# 1. LOAD FAKE NEWS MODEL
# =========================================================

MODEL_NAME = "hamzab/roberta-fake-news-classification"

print("Loading AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

model.eval()

print("AI model loaded successfully!")


# =========================================================
# 2. CONNECT GEMINI
# =========================================================

print("Connecting to Gemini...")

gemini_client = genai.Client()

print("Gemini connected successfully!")


# =========================================================
# 3. PLAYWRIGHT - EXTRACT NEWS FROM URL
# =========================================================

def extract_news_from_url(url):

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36"
            )
        )

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            page.wait_for_timeout(3000)

            title = page.title()

            # Try article tag first
            articles = page.locator(
                "article"
            ).all_inner_texts()

            if articles:

                article_text = max(
                    articles,
                    key=len
                )

            else:

                # Try main tag
                main = page.locator("main")

                if main.count() > 0:

                    article_text = (
                        main.first.inner_text()
                    )

                else:

                    # Fallback to body
                    article_text = (
                        page.locator(
                            "body"
                        ).inner_text()
                    )

            return (
                title,
                article_text[:12000],
                None
            )

        except Exception as e:

            return (
                None,
                None,
                str(e)
            )

        finally:

            browser.close()


# =========================================================
# 4. ROberta FAKE NEWS PREDICTION
# =========================================================

def predict_news(title, text):

    input_text = (
        "<title>"
        + title
        + "<content>"
        + text
        + "<end>"
    )

    inputs = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    fake_probability = (
        probabilities[0].item()
    )

    real_probability = (
        probabilities[1].item()
    )

    if fake_probability > real_probability:

        prediction = "Likely Fake News"

        confidence = (
            fake_probability * 100
        )

    else:

        prediction = "Likely Real News"

        confidence = (
            real_probability * 100
        )

    return (
        prediction,
        round(confidence, 2)
    )


# =========================================================
# 5. GEMINI AI EXPLANATION
# =========================================================

def generate_ai_explanation(
    title,
    text,
    prediction,
    confidence
):

    prompt = f"""
You are an AI assistant for a Fake News Detection System.

Analyze the news article below and explain the
machine learning prediction in simple language.

NEWS TITLE:
{title}

NEWS CONTENT:
{text[:8000]}

MACHINE LEARNING PREDICTION:
{prediction}

MODEL CONFIDENCE:
{confidence}%

Please explain:

1. What the article is mainly about.
2. Why the machine learning model may have given
   this prediction.
3. Important claims or information in the article.
4. Whether the article should be verified further.

IMPORTANT RULES:

- Do not say the article is definitely fake.
- Do not say the article is definitely real.
- The machine learning result is only an
  AI-assisted prediction.
- Do not invent facts.
- Do not invent sources.
- Keep the explanation simple.
"""

    # Try multiple currently supported Gemini models
    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]

    for model_name in models_to_try:

        try:

            print(
                "Trying Gemini:",
                model_name
            )

            response = (
                gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            )

            if response.text:

                print(
                    "Gemini response received from:",
                    model_name
                )

                return response.text

        except Exception as e:

            print(
                "Gemini error:",
                model_name,
                e
            )

            continue

    return (
        "Gemini is temporarily unavailable. "
        "The machine learning prediction was "
        "generated successfully, but the AI "
        "explanation could not be generated. "
        "Please try again later."
    )


# =========================================================
# 6. FLASK HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    explanation = None
    title = None
    extracted_text = None
    confidence = None
    error = None

    if request.method == "POST":

        input_type = request.form.get(
            "input_type"
        )


        # =================================================
        # NEWS TEXT
        # =================================================

        if input_type == "text":

            news = request.form.get(
                "news",
                ""
            ).strip()

            if news:

                title = "User Provided News"

                extracted_text = news

                try:

                    # ML prediction
                    result, confidence = (
                        predict_news(
                            title,
                            news
                        )
                    )

                    # Gemini explanation
                    explanation = (
                        generate_ai_explanation(
                            title,
                            news,
                            result,
                            confidence
                        )
                    )

                except Exception as e:

                    error = (
                        "AI model error: "
                        + str(e)
                    )

            else:

                error = (
                    "Please enter some news text."
                )


        # =================================================
        # NEWS URL
        # =================================================

        elif input_type == "url":

            url = request.form.get(
                "url",
                ""
            ).strip()

            if url:

                # Playwright extraction
                (
                    title,
                    extracted_text,
                    error
                ) = extract_news_from_url(
                    url
                )

                if not error:

                    try:

                        # ML prediction
                        result, confidence = (
                            predict_news(
                                title,
                                extracted_text
                            )
                        )

                        # Gemini explanation
                        explanation = (
                            generate_ai_explanation(
                                title,
                                extracted_text,
                                result,
                                confidence
                            )
                        )

                    except Exception as e:

                        error = (
                            "AI model error: "
                            + str(e)
                        )

            else:

                error = (
                    "Please enter a valid "
                    "news URL."
                )


    # =====================================================
    # SEND RESULT TO HTML
    # =====================================================

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        title=title,
        extracted_text=extracted_text,
        confidence=confidence,
        error=error
    )


# =========================================================
# 7. START APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )