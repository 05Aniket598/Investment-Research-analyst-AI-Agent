import os
from typing import List

from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

load_dotenv()

# Reduce LiteLLM logging noise
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DISABLE_LOGGING"] = "true"

# ===================== Design =====================
# Data fetching has NO LLM: prefetch.py runs the SQL and the annual-report search in
# Python, and main.py passes the results to the tasks as inputs. So no agent has tools.
#
# Gemini requests per run (free limit: 20 per day per model):
#   Financial analyst  -> 1 request  (analyst_llm_1)
#   Risk analyst       -> 1 request  (analyst_llm_2)
#   Valuation analyst  -> 1 request  (analyst_llm_3)
#   Manager            -> 1 request  (manager_llm)
#   Total: 4 requests normally (a retry after a bad answer adds 1 more).
# Each model gets only 1 request per run, so a model allows about 20 runs per day.
# If you point two agents at the SAME model name, that model's daily limit is shared.
#
# Model names come from your .env, so you can change them without editing this file.
# The defaults are the names from your last crew.py. Check them in Google AI Studio
# (rate-limit page): I cannot verify that these exact names exist.

def _llm(env_name: str, default: str, max_tokens: int) -> LLM:
    return LLM(
        provider="google",
        model=os.getenv(env_name, default),
        temperature=0.1,
        max_tokens=max_tokens,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

# Reasoning models spend part of max_tokens on thinking, so the analysts get more room.
analyst_llm_1 = _llm("ANALYST1_MODEL", "gemini-3.1-flash-lite", 16000)    # Financial analysis
analyst_llm_2 = _llm("ANALYST2_MODEL", "gemini-3.6-flash", 16000)  # Risk analysis
analyst_llm_3 = _llm("ANALYST3_MODEL", "gemini-3.6-flash", 16000)  # Valuation
manager_llm = _llm("MANAGER_MODEL", "gemini-3.6-flash", 8000)  # Final report

# ===================== Crew =====================

@CrewBase
class InvestmentResearchAnalyst:
    """Investment Research Analyst Crew (no tools, data comes from prefetch.py)"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ===================== Agents =====================
    # max_iter=2: if the answer format is wrong, allow at most 1 retry (saves your quota).

    @agent
    def FinancialAnalysisAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["FinancialAnalysisAgent"],
            verbose=True,
            llm=analyst_llm_1,
            allow_delegation=False,
            max_iter=2,
        )

    @agent
    def RiskAnalysisAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["RiskAnalysisAgent"],
            verbose=True,
            llm=analyst_llm_2,
            allow_delegation=False,
            max_iter=2,
        )

    @agent
    def ValuationAnalysisAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["ValuationAnalysisAgent"],
            verbose=True,
            llm=analyst_llm_3,
            allow_delegation=False,
            max_iter=2,
        )

    @agent
    def ManagerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["ManagerAgent"],
            verbose=True,
            llm=manager_llm,
            allow_delegation=False,
            max_iter=2,
        )

    # ===================== Tasks =====================

    @task
    def fundamental_performance_and_peer_comparison_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_performance_and_peer_comparison_analysis"]
        )

    @task
    def multi_dimensional_risk_identification_and_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["multi_dimensional_risk_identification_and_assessment"],
            context=[self.fundamental_performance_and_peer_comparison_analysis()],
        )

    @task
    def equity_valuation_and_mispricing_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["equity_valuation_and_mispricing_assessment"],
            context=[
                self.fundamental_performance_and_peer_comparison_analysis(),
                self.multi_dimensional_risk_identification_and_assessment(),
            ],
        )

    @task
    def integrated_equity_research_report_generation(self) -> Task:
        return Task(
            config=self.tasks_config["integrated_equity_research_report_generation"],
            context=[
                self.fundamental_performance_and_peer_comparison_analysis(),
                self.multi_dimensional_risk_identification_and_assessment(),
                self.equity_valuation_and_mispricing_assessment(),
            ],
            output_file="report.md",
        )

    # ===================== Crew =====================

    @crew
    def crew(self) -> Crew:
        # SMOKE_TEST=1 runs only the first task (1 Gemini request)
        tasks = self.tasks[:1] if os.getenv("SMOKE_TEST") == "1" else self.tasks
        return Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )