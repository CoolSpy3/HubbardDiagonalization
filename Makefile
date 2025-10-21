.PHONY: clean debug default run

default: run

clean:
	@rm -rf output

debug:
	@mkdir -p output
	@julia --project=. -m HubbardDiagonalization --debug 2>&1 | tee output/debug.log

run:
	@julia --project=. -m HubbardDiagonalization
