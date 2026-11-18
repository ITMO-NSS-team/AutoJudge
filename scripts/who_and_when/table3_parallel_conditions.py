"""Run frozen conditions concurrently with one shared request and budget limit."""
import ast
import asyncio
import inspect
import table3_ablation as experiment

clients = []
slots = asyncio.Semaphore(12)
budget_lock = asyncio.Lock()
reserved = 0.0
stopped = False

class SharedClient(experiment.ConservativeClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = asyncio.Semaphore(6)
        clients.append(self)
        experiment.p.write_json(self.path.parent / 'transport_override.json', {
            'recorded_at': experiment.p.now(), 'conditions_run_sequentially': False,
            'max_concurrent_requests_total': 12,
            'max_concurrent_requests_per_active_condition': 6,
            'reason': 'User requested concurrent conditions; frozen scientific inputs and durable receipts retained; shared budget includes all in-flight reservations.'})

    async def request(self, payload, task, arm, stage, attempt):
        global reserved, stopped
        async with slots:
            call_id = experiment.p.digest([self.manifest['run_id'], task, arm, stage, attempt])
            cached = (self.path / 'calls' / (call_id + '.json')).exists()
            reserve = 0.0 if cached else self.reservation(payload)
            async with budget_lock:
                if stopped or sum(c.spent for c in clients) + reserved + reserve > self.max_cost:
                    stopped = True
                    raise experiment.HaltRun('Shared campaign budget or stop reached')
                reserved += reserve
            try:
                return await super().request(payload, task, arm, stage, attempt)
            except experiment.HaltRun:
                stopped = True
                raise
            finally:
                async with budget_lock:
                    reserved -= reserve

async def drain(tasks):
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result

# Preserve all validation, checkpoint and output logic from the frozen runner.
# Only lift the inner condition loop into concurrent tasks, with a shared budget.
tree = ast.parse(inspect.getsource(experiment.run))
class ConcurrentConditions(ast.NodeTransformer):
    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == 'name' and isinstance(node.iter, ast.Name) and node.iter.id == 'names':
            body = node.body
            class ConditionBody(ast.NodeTransformer):
                def visit_Continue(self, n): return ast.copy_location(ast.Return(value=None), n)
            body = [ConditionBody().visit(n) for n in body]
            body = [n for n in body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'others' for t in n.targets))]
            for n in body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'max_cost' for t in n.targets):
                    n.value = ast.Name(id='max_cost',ctx=ast.Load())
            function = ast.AsyncFunctionDef(name='run_condition',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='name')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
            call = ast.parse('await drain([run_condition(name) for name in names])').body[0]
            return [function,call]
        return self.generic_visit(node)
tree = ast.fix_missing_locations(ConcurrentConditions().visit(tree))
experiment.ConservativeClient = SharedClient
experiment.__dict__['drain'] = drain
exec(compile(tree, __file__, 'exec'), experiment.__dict__)
if __name__ == '__main__':
    experiment.main()
