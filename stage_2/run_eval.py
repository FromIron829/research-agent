import sys, json
from pathlib import Path

STAGE2 = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE2))
from agent import answer as run_agent     # reuse your Stage 2 agent

eval_q = json.loads((STAGE2 / "eval" / "eval_set.json").read_text())["questions"]
answers = []
for q in eval_q:
    print(f"running {q['id']} : {q['question'][:60]}...")
    res = run_agent(q["question"])
    answers.append({"id": q["id"], "question": q["question"], "answer": res["answer"]})

(STAGE2 / "eval" / "agent_answers.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False))
print(f"saved {len(answers)} answers")
