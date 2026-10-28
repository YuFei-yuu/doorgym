import ast,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from legdiag_random5.diagnostic import LegDiagnostic
d=LegDiagnostic()
assert d.observe(4,.08,0.,['j'],{'j':.4}) is None
assert d.observe(5,.10,0.,['j'],{'j':.4})['diagnostic_step']==5
assert d.observe(6,.12,0.,['j'],{'j':.4}) is None
assert d.observe(7,.14,None,[],{}) is None
assert d.observe(14,.28,.16,['j'],{'j':.4}) is not None
assert len(d.events)==2 and not d.metadata()['abort_enabled']
# Prove the loop is otherwise AST-identical: only bookkeeping and old abort replacement.
base=ast.parse((ROOT/'migration/wc4_act/success_random5/runtime.py').read_text())
new=ast.parse((ROOT/'migration/wc4_act/legdiag_random5/runtime.py').read_text())
def clean(tree):
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            return None if n.module=='legdiag_random5.diagnostic' else n
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id in ['leg_diagnostic','diagnostic'] for t in n.targets):return None
            return self.generic_visit(n)
        def visit_If(self,n):
            if isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='diagnostic':return None
            if isinstance(n.test,ast.BoolOp) and any(isinstance(x,ast.Compare) and isinstance(x.left,ast.Name) and x.left.id=='foot_bad_since' for x in n.test.values):return None
            return self.generic_visit(n)
        def visit_keyword(self,n):
            return None if n.arg=='leg_posture_diagnostic' else self.generic_visit(n)
    return ast.dump(Strip().visit(tree),include_attributes=False)
assert clean(base)==clean(new)
print('PASS: event edge/reset behavior; all other runtime AST unchanged')
