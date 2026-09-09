from importlib import import_module


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = '.main' if name == 'autojudge' else '.pipeline' if name in (
        'Pipeline', 'PipelineBuilder', 'AgentNode') else '.agent_pool'
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "autojudge",
    "Pipeline",
    "PipelineBuilder",
    "AgentNode",
    "AgentPool",
    "DefaultAgents",
    "RESEARCH_POOL",
    "PLANNING_POOL",
    "ALL_AGENTS_POOL",
]
