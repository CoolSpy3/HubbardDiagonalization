.PHONY: clean debug default format run setup test_n2

default: run

clean:
	@rm -rf output

debug:
	@mkdir -p output
	@julia --project=. -m HubbardDiagonalization --debug 2>&1 | tee output/debug.log

format:
	@julia --project=. -e 'import JuliaFormatter; JuliaFormatter.format("."; always_for_in = true)'

run:
	@julia --project=. -m HubbardDiagonalization

setup:
	@julia --project=. -e 'import Pkg; Pkg.instantiate()'

test_n2:
	@julia --project=. -e 'include("tests/n2_grids/TestN2.jl"); using .TestN2'
