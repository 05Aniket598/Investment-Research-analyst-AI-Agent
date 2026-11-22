#!/usr/bin/env python
import os
import warnings
from datetime import datetime

from investment_analyst_agent.crew import InvestmentResearchAnalyst
from investment_analyst_agent.prefetch import run_prefetch

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DISABLE_PROXY"] = "true"
os.environ["LITELLM_SILENT"] = "true"


def run():
    """
    Step 1 (Python, no LLM): run the SQL and the annual-report search.
    Step 2 (Gemini, about 4 requests): the crew reasons over that data and writes report.md.
    """
    stock = "TCS"
    peers = ["INFY"]

    data = run_prefetch(stock, peers)   # prints progress, saves knowledge/prefetch_TCS.md

    inputs = {
        "stock": stock,
        "peer_companies": ", ".join(peers),
        "report_date": datetime.now().strftime("%d %B %Y"),
        **data,   # financial_data, valuation_data, commentary_performance, commentary_risk, data_notes
    }
    try:
        result = InvestmentResearchAnalyst().crew().kickoff(inputs=inputs)
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


# from investment_research_analyst.tools.rag_tool import AnnualReportSearchTool
# print(AnnualReportSearchTool()._run("key risks", "TCS", "FY26"))

# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "topic": "AI LLMs",
#         'current_year': str(datetime.now().year)
#     }
#     try:
#         InvestmentResearchAnalyst().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")

# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         InvestmentResearchAnalyst().crew().replay(task_id=sys.argv[1])

#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")

# def test():
#     """
#     Test the crew execution and returns the results.
#     """
#     inputs = {
#         "topic": "AI LLMs",
#         "current_year": str(datetime.now().year)
#     }

#     try:
#         InvestmentResearchAnalyst().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while testing the crew: {e}")

# def run_with_trigger():
#     """
#     Run the crew with trigger payload.
#     """
#     import json

#     if len(sys.argv) < 2:
#         raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

#     try:
#         trigger_payload = json.loads(sys.argv[1])
#     except json.JSONDecodeError:
#         raise Exception("Invalid JSON payload provided as argument")

#     inputs = {
#         "crewai_trigger_payload": trigger_payload,
#         "topic": "",
#         "current_year": ""
#     }

#     try:
#         result = InvestmentResearchAnalyst().crew().kickoff(inputs=inputs)
#         return result
#     except Exception as e:
#         raise Exception(f"An error occurred while running the crew with trigger: {e}")
