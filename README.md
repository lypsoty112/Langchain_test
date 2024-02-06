# Streamlit Stock Recommendation App

This is a simple Streamlit app that provides stock recommendation based on a given company and strategy. The recommendation is generated using the provided Python file that utilizes OpenAI's language model and other tools.

## Features

- Allows users to input a company name and a strategy.
- Utilizes OpenAI's language model to generate a stock recommendation based on the input.
- Provides reasoning behind the recommendation and allows users to provide an alternate strategy.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/lypsoty112/Langchain_test.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the project directory:

    ```bash
    cd langchain_test
    ```

2. Create a `.env` file based on the `.env.template` file and fill in the necessary API keys (see instructions below).

3. Run the Streamlit app:

    ```bash
    streamlit run src/app.py
    ```

4. Open your web browser and go to `http://localhost:8501` to access the app.

5. Enter the company name and strategy, then click the "Run" button to get the stock recommendation.

## Customization

- Modify the `app.py` file to customize the Streamlit app's behavior or appearance.
- Adjust the color scheme and layout in the `app.py` file according to your preferences.

## Environment Variables

This project uses environment variables to store sensitive information such as API keys. You need to create a `.env` file in the root directory of the project based on the `.env.template` file provided. The `.env` file should contain the following variables:

```env
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_CSE_ID="your_google_cse_id_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

Replace `"your_openai_api_key_here"`, `"your_google_cse_id_here"`, and `"your_google_api_key_here"` with your actual API keys.

## Credits

- This app utilizes OpenAI's language model and other tools for generating stock recommendations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
