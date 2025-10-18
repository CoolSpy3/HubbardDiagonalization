.PHONY: clean debug default run

default: run

clean:
	@rm -rf output

debug:
	@echo -e "\x1b[1;33m WARNING:\x1b[22m This is going to produce a *lot* of output! \x1b[0m"
	@echo -e "\x1b[0;33m If you're running this for a non-trivial case, consider re-running in normal mode. \x1b[0m"
	@echo -e "\x1b[1;31m YOU HAVE BEEN WARNED! \x1b[0m"
	@mkdir -p output
# Use julia's sleep function as a cross-platform way to give the user a moment to read the warning.
	@julia --project=. -e \
		'sleep(5); import Logging; Logging.global_logger(Base.CoreLogging.ConsoleLogger(stdout, Logging.Debug)); using HubbardDiagonalization' \
		| tee output/debug.log

run:
	@julia --project=. -e 'using HubbardDiagonalization'
