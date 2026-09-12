import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Enterprise Knowledge Assistant")

st.write(
    "Ask questions about the Microsoft 2025 Annual Report."
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8001/ask"


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question:",
    placeholder="What does Microsoft say about AI?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=120
                )


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    st.subheader("Answer")

                    st.write(
                        result["answer"]
                    )


                    # ------------------------------------------------
                    # SOURCES
                    # ------------------------------------------------

                    sources = result.get(
                        "sources",
                        []
                    )

                    if sources:

                        st.subheader("Sources")

                        for source in sources:

                            st.write(
                                f"📄 {source['source']} — "
                                f"Page {source['page']}"
                            )

                    else:

                        st.info(
                            "No supporting sources found."
                        )


                # ------------------------------------------------
                # API ERROR
                # ------------------------------------------------

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure the API is running on port 8001."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )


            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )
