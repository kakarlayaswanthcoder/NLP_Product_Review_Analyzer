import streamlit as st
import joblib

from nlp_utils import preprocess_text
from translator import translate_text



model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")



st.set_page_config(
    page_title="NLP Product Review Analyzer",
    page_icon="🧠",
    layout="wide"
)




st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #aaaaaa;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)



st.markdown(
    '<div class="main-title">🧠 NLP Product Review Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze product reviews using Natural Language Processing, '
    'Machine Learning and Multilingual Translation.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()



st.markdown(
    '<div class="section-title">📝 Enter Product Review</div>',
    unsafe_allow_html=True
)

review = st.text_area(
    "Write your product review:",
    placeholder=(
        "Example: This phone has an amazing camera "
        "and excellent battery life."
    ),
    height=160
)


st.markdown(
    '<div class="section-title">🌐 Translation</div>',
    unsafe_allow_html=True
)

language = st.selectbox(
    "Choose target language:",
    [
        "Telugu",
        "Hindi",
        "Tamil",
        "Kannada",
        "Malayalam",
        "English"
    ]
)


language_codes = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "English": "en"
}



if st.button("🔍 Analyze Review", use_container_width=True):


    if review.strip() == "":
        st.warning("⚠️ Please enter a product review.")

    else:

        

        processed_tokens = preprocess_text(review)

        processed_text = " ".join(processed_tokens)


       

        review_features = vectorizer.transform(
            [processed_text]
        )


       

        prediction = model.predict(review_features)[0]



        try:

            probabilities = model.predict_proba(
                review_features
            )[0]

            confidence = max(probabilities) * 100

        except AttributeError:

            confidence = 0


        
        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Sentiment Analysis'
            '</div>',
            unsafe_allow_html=True
        )


        if prediction == "positive":

            st.success("😊 POSITIVE REVIEW")

        elif prediction == "neutral":

            st.warning("😐 NEUTRAL REVIEW")

        else:

            st.error("😡 NEGATIVE REVIEW")


      

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Sentiment",
                prediction.upper()
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        with col3:

            st.metric(
                "Word Count",
                len(processed_tokens)
            )


      

        st.write("Prediction Confidence")

        st.progress(
            min(int(confidence), 100)
        )


       

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🧠 NLP Preprocessing'
            '</div>',
            unsafe_allow_html=True
        )


      

        st.write("Original Review:")

        st.info(review)


      

        st.write("Tokens after preprocessing:")

        st.code(
            processed_tokens
        )


       

        st.write("🔑 Important Words")


        if processed_tokens:

            unique_words = list(
                dict.fromkeys(processed_tokens)
            )

            st.write(
                ", ".join(unique_words)
            )

        else:

            st.write(
                "No important words found."
            )


      
        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🌐 Translation'
            '</div>',
            unsafe_allow_html=True
        )


        target_language = language_codes[language]


        try:

           

            if target_language == "en":

                translated_text = review


           

            else:

                translated_text = translate_text(
                    review,
                    target_language
                )


            st.info(
                translated_text
            )


        except Exception as e:

            st.error(
                "❌ Translation failed. "
                "Please check your internet connection."
            )




st.divider()

st.caption(
    "NLP Product Review Analyzer | "
    "Python • NLTK • Scikit-learn • Streamlit"
)