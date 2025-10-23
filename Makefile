.PHONY: splits features train eval plot api

CFG = configs/baseline.yaml

splits:
	harmonyai splits --config $(CFG)

features:
	harmonyai features --config $(CFG)

train:
	harmonyai train --config $(CFG)

eval:
	harmonyai eval --config $(CFG)

plot:
	harmonyai plot --config $(CFG)

api:
	uvicorn harmony_ai.api:app --reload --port 8000
