import dotenv
from langchain_openai import OpenAI
from langchain.tools import BaseTool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain
import streamlit as st
import matplotlib.pyplot as plt


dotenv.load_dotenv()

llm = OpenAI(temperature=0.7)

def run(
    company: str,
    strategy: str,
): 
    search_results = search(company, create_search_agent())
    
    template = """
Given the following information about {company}, formulate a buy/sell/hold recommendation for the stock. Add your reasoning for the recommendation. 
You're allowed to formulate another strategy if you think it's more appropriate.

<<STRATEGY>>
{strategy}

<<INPUT>>
{information}

<<OUTPUT>>
{expected_output}
    """

    recommendation_schema = ResponseSchema(
        name="Recommendation",
        description="Is the recommended move a buy, sell or a hold given the provided strategy?",
    )
    reasoning_schema = ResponseSchema(
        name="Reasoning",
        description="Reasoning behind the recommendation and the strategy used.",
    )

    alternate_strategy_schema = ResponseSchema(
        name="Alternate Strategy",
        description="If you think the recommended strategy is not appropriate, provide an alternate strategy.",
    )


    output_parser = StructuredOutputParser.from_response_schemas(
        [recommendation_schema, reasoning_schema, alternate_strategy_schema]
    )

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=template,
        input_variables=["company", "strategy", "information"],
        partial_variables={"expected_output": format_instructions},
        input_types={
            "company": "string",
            "strategy": "string",
            "information": "dict[string, string]",
            "expected_output": "string",
        },
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    for i in range(3):
        try: 
            result =  chain.invoke(
                input={
                    "company": company,
                    "strategy": strategy,
                    "information": search_results,
                }
            )["text"]
        except Exception as e:
            print(e)
            if i == 2:
                raise e
            continue
    
    return result


def create_search_agent():
    class SearchTool(BaseTool):
        name = "Google Search"
        description = "Search Google for recent results. Input is a string."

        def _run(self, query: str):
            search = GoogleSearchAPIWrapper()
            return search.results(query=query, num_results=3)

        def _arun(self, query: str):
            raise NotImplementedError


    tools = [SearchTool()]

    prompt = hub.pull("hwchase17/react")

    searchAgent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools,
    )

    return AgentExecutor(agent=searchAgent, tools=tools, verbose=False, handle_parsing_errors=True)


def search(company: str, search_executor: AgentExecutor) -> dict[str, str]:
    templates = {
        "positive_news": "Provide an overview of the most recent positive news that might have an impact on the stock price of {company}.",
        "negative_news": "Provide an overview of the most recent negative news that might have an impact on the stock price of {company}.",
        "financial": "Provide an overview of the most recent financial information about {company}.",
        "general_opinion": "Provide an overview of the general opinion investors have about {company}.",
    }

    prompts = {}
    for key, template in templates.items():
        prompts[key] = PromptTemplate.from_template(template)

    results = {}
    for key, prompt in prompts.items():
        results[key] = search_executor.invoke(
            input={"input": prompt.invoke(input={"company": company})}
        )["output"]

    return results

def main():
    # Set configuration
    st.set_page_config(
        page_title="Stock Recommendation App",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Apply custom styling
    st.title("Stock Recommendation App")

    # Sidebar inputs
    company = st.sidebar.text_input("Company", "Apple")
    strategy = st.sidebar.text_area("Strategy", "Invest for the long term.")

    if st.sidebar.button("Run"):
        st.write("Running recommendation for:", company)
        st.write("Strategy:", strategy)

        # Run the function
        output = run(company, strategy)

        # Display the output
        st.subheader("Recommendation:")
        st.write(output["Recommendation"])

        st.subheader("Reasoning:")
        st.write(output["Reasoning"])

        st.subheader("Alternate Strategy:")
        st.write(output["Alternate Strategy"])

if __name__ == "__main__":
    main()
    