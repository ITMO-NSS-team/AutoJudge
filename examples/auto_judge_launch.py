import asyncio

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from automas.utils.langfuse_utils import ainvoke_with_lf
from maseval import get_langfuse_download_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(".env")

async def main(name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097"):
    pool_gen = PoolGenerator()
    graph_gen = GraphGenerator()
    lf = get_langfuse_download_client()

    traces_page1 = lf.api.trace.list(name=name, limit=50, page=1)
    
    parsed_traces = [parse_langfuse_task(lf.api.trace.get(task.id)) for task in traces_page1.data]
    
    for idx, query in tqdm(enumerate(parsed_traces)):
        judge_input = str({"query": query.user_query, "history_for_evaluating": query.agent_states})
        pool = await pool_gen.create_pool(judge_input)
        graph = await graph_gen.create_graph(pool, judge_input)

        builder = PipelineBuilder()
        pipeline = builder.create_from_pool(pool, graph).build()
        pipeline.to_mermaid_lr(visualize=True)

        result, trace_id = await ainvoke_with_lf(pool, pipeline, judge_input, graph)

        print(f"Result: {result}")
        print(f"Langfuse trace ID: {trace_id}")
        print(f"Launch № {idx} from {len(parsed_traces)}")


if __name__ == "__main__":
    asyncio.run(main())
