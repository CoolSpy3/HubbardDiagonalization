.PHONY: clean default run

default: run

clean:
	@rm -rf output

run:
	@julia --project=. -e 'using HubbardDiagonalization'
