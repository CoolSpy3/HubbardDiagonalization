.PHONY: clean debug default run setup

default: run

clean:
	@rm -rf output

debug:
	@mkdir -p output
	@julia --project=. -m HubbardDiagonalization --debug 2>&1 | tee output/debug.log

run:
	@julia --project=. -m HubbardDiagonalization

setup:
	@julia --project=. -e 'import Pkg; Pkg.instantiate()'
