from .calculator_env import CalculatorEnv
from .retriever_env import RetrieverEnv
from .list_manipulation_env import ListManipulationEnv
from .statistics_env import StatisticsEnv
from .counting_env import CountingEnv
from .matrix_env import MatrixEnv
from .prime_env import PrimeEnv
from .historical_year_env import HistoricalYearEnv
from .game_rule_env import GameRuleEnv
from .hash_env import HashEnv
from .decoding_env import DecodingEnv
from .datetime_env import DateTimeEnv
from .code_executor_env import CodeExecutorEnv
from .schedule_env import ScheduleEnv
from .regex_match_env import RegexMatchEnv


ENV_REGISTRY = {
    "CalculatorEnv": CalculatorEnv,
    "RetrieverEnv": RetrieverEnv,
    "ListManipulationEnv": ListManipulationEnv,
    "StatisticsEnv": StatisticsEnv,
    "CountingEnv": CountingEnv,
    "MatrixEnv": MatrixEnv,
    "PrimeEnv": PrimeEnv,
    "HistoricalYearEnv": HistoricalYearEnv,
    "GameRuleEnv": GameRuleEnv,
    "HashEnv": HashEnv,
    "DecodingEnv": DecodingEnv,
    "DateTimeEnv": DateTimeEnv,
    "CodeExecutorEnv": CodeExecutorEnv,
    "ScheduleEnv": ScheduleEnv,
    "RegexMatchEnv": RegexMatchEnv,
}
