from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, SeleniumScrapingTool

from dotenv import load_dotenv
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

load_dotenv()

# Creating tools for the agent

web_search_tool = SerperDevTool()
website_scraping_tool = ScrapeWebsiteTool()
selenium_scraping_tool = SeleniumScrapingTool()

toolkit = [web_search_tool, website_scraping_tool, selenium_scraping_tool]

import os
from crewai import LLM

# Disable LiteLLM proxy / logging noise
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DISABLE_LOGGING"] = "true"

# # ===================== CLOUD LLM =====================
# gemini_llm = LLM(
#     model="gemini-3-flash",
#     provider="google",
#     temperature=0.1,
#     max_tokens=12800,
# )

# # ===================== LOCAL LLMs (CPU-safe) =====================

# phi3_llm = LLM(
#     model="ollama/phi3:mini",
#     provider="ollama",
#     base_url="http://localhost:11434",
#     temperature=0.0,
#     max_tokens=12800,
#     extra_kwargs={
#         "num_ctx": 12800,
#         "num_thread": 6
#     }
# )

# qwen_llm = LLM(
#     model="ollama/qwen2.5:3b",
#     provider="ollama",
#     base_url="http://localhost:11434",
#     temperature=0.1,
#     max_tokens=12800,
#     extra_kwargs={
#         "num_ctx": 12800,
#         "num_thread": 6
#     }
# )

import time

from crewai import LLM
import os

gemini3_llm = LLM(
    provider="google",
    model="gemini-3-flash",
    temperature=0.15,
    max_tokens=-1,
    api_key=os.getenv("GEMINI_API_KEY")
)

flash_llm = LLM(
    provider="google",
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=-1,
    api_key=os.getenv("GEMINI_API_KEY")
)

gemma_llm = LLM(
    provider="google",
    model="gemma-3-12b",
    temperature=0.0,
    max_tokens=-1,
    api_key=os.getenv("GEMINI_API_KEY")
)
from google.api_core.exceptions import ResourceExhausted


def get_llm_with_retry(task_type):
    if task_type not in ["analysis", "risk", "valuation"]:
        return flash_llm

    try:
        return gemini3_llm
    except ResourceExhausted:
        time.sleep(2)
        try:
            return gemini3_llm
        except ResourceExhausted:
            return flash_llm





# =======================Building Crewagent============================

@CrewBase
class InvestmentResearchAnalyst() :
    """InvestmentResearchAnalyst Crew"""

    agents = List[BaseAgent]
    tasks = List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def DataEngineer(self) -> Agent:
        return Agent(
            config = self.agents_config["DataEngineer"],
            verbose = True,
            tools = toolkit,
            llm = flash_llm
        )
    
    
    @agent
    def FinancialAnalysisAgent(self) -> Agent:
        return Agent(
            config = self.agents_config["FinancialAnalysisAgent"],
            verbose = True,
            llm = get_llm_with_retry("analysis"),
            # tools = toolkit
        )
    
    @agent
    def RiskAnalysisAgent(self) -> Agent:
        return Agent(
            config = self.agents_config["RiskAnalysisAgent"],
            verbose = True,
            llm = get_llm_with_retry("risk"),
            # tools = toolkit
        )
    
    @agent
    def ValuationAnalysisAgent(self) -> Agent:
        return Agent(
            config = self.agents_config["ValuationAnalysisAgent"],
            verbose = True,
            llm = get_llm_with_retry("valuation"),
            # tools = toolkit
        )
    
    @agent
    def ManagerAgent(self) -> Agent:
        return Agent(
            config = self.agents_config["ManagerAgent"],
            verbose = True,
            llm = flash_llm,
            # tools = toolkit
        )

# ========================Task=========================================

    @task
    def data_collection_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_collection_task"]
        )
    
    @task
    def fundamental_performance_and_peer_comparison_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_performance_and_peer_comparison_analysis"],
            context=[self.data_collection_task()]
        )
    
    @task
    def multi_dimensional_risk_identification_and_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["multi_dimensional_risk_identification_and_assessment"],
            context=[self.data_collection_task(),self.fundamental_performance_and_peer_comparison_analysis()]
        )
    
    @task
    def equity_valuation_and_mispricing_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["equity_valuation_and_mispricing_assessment"],
            context=[self.data_collection_task(),
                     self.fundamental_performance_and_peer_comparison_analysis(),
                     self.multi_dimensional_risk_identification_and_assessment()]
        )
    
    @task
    def integrated_equity_research_report_generation(self) -> Task:
        return Task(
            config=self.tasks_config["integrated_equity_research_report_generation"],
            context=[self.data_collection_task(),
                     self.fundamental_performance_and_peer_comparison_analysis(),
                     self.multi_dimensional_risk_identification_and_assessment(),
                     self.equity_valuation_and_mispricing_assessment()],
            output_file='report.md'
        )
    

    #===============Crew=================

    @crew
    def crew(self) ->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential
        )
    