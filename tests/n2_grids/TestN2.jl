module TestN2

export main

import HubbardDiagonalization
import HubbardDiagonalization.Graphs

import CSV

using Logging
using Statistics
using ZipFile

function read_csv(name::String, zip_reader::ZipFile.Reader)
    matching_files = filter(f -> endswith(f.name, name), zip_reader.files)
    @assert length(matching_files) == 1 "Expected exactly one file matching $name, found $(length(matching_files))"
    file = matching_files[1]
    return CSV.File(file, header = false) |> CSV.Tables.matrix
end

function (@main)(args)
    warn_on_nan = false

    # Setup mappings between result names and csv files
    test_observables = Dict(
        "Density" => "Densities.csv",
        "Filled States" => "Doubleoccupancies.csv",
        "Energy" => "Energies.csv",
        "Entropy" => "Entropies.csv",
        "Local Moment" => "ninis.csv",
    )

    if !warn_on_nan
        @warn "NaN warnings are disabled."
    end

    grids_zip = ZipFile.Reader("./tests/n2_grids/N2_grids.zip")

    graph = Graphs.linear_chain(4)

    generated_statistics = Dict{String,Vector{Any}}("N" => [], "U" => [], "T" => [])
    for observable_name in keys(test_observables)
        generated_statistics[observable_name] = []
    end

    for test_set in grids_zip.files
        parsed_name = match(r"2x2_N(\d+)_U(\d+)_Tmu.zip", test_set.name)
        if parsed_name === nothing
            continue
        end

        N = parse(Int, parsed_name.captures[1])
        U = parse(Float64, parsed_name.captures[2])

        observables = HubbardDiagonalization.default_observables(N, graph)

        @info "Running tests for N=$N, U=$U..."

        # Reading Nested Zip Files: https://stackoverflow.com/a/44877369
        buffer = IOBuffer(read(test_set))
        test_data = ZipFile.Reader(buffer)

        u_vals = read_csv("mu_vals.csv", test_data)[:, 1]
        T_vals = read_csv("T_vals.csv", test_data)[:, 1]

        # Map file names to expected data
        expected_data = Dict{String,AbstractMatrix{Float64}}()

        # Read in the test files
        for results_file in unique(values(test_observables))
            expected = read_csv(results_file, test_data)
            @assert size(expected) == (length(T_vals), length(u_vals)) "Expected data size mismatch for $results_file"
            expected_data[results_file] = expected
        end

        for (i, T) in enumerate(T_vals)
            test_config = HubbardDiagonalization.TestConfiguration(
                num_colors = N,
                t = 1.0,
                T = T,
                u_test = 0.0,
                U = U / 4.0,
            )

            # Temporarily disable info logging for cleaner test output
            disable_logging(Logging.Info)
            results = HubbardDiagonalization.diagonalize_and_compute_observables(
                u_vals,
                test_config,
                graph,
                observables...,
            )
            disable_logging(Logging.Debug)

            push!(generated_statistics["N"], N)
            push!(generated_statistics["U"], U)
            push!(generated_statistics["T"], T)
            for (observable_name, results_file) in test_observables
                expected = expected_data[results_file][i, :]
                computed = results[observable_name]

                @assert length(expected) == length(computed) "Length mismatch for observable $observable_name at N=$N, U=$U, T=$T"

                if warn_on_nan && any(isnan, expected)
                    @warn "Expected data contains NaN values for N=$N, U=$U, T=$T, observable=$observable_name at " *
                          "$(u_vals[findfirst(isnan, expected)]) < u < $(u_vals[findlast(isnan, expected)]). " *
                          "These values will be ignored in the comparison."
                end
                if warn_on_nan && any(isnan, computed)
                    @warn "Computed data contains NaN values for N=$N, U=$U, T=$T, observable=$observable_name at " *
                          "$(u_vals[findfirst(isnan, computed)]) < u < $(u_vals[findlast(isnan, computed)]). " *
                          "These values will be ignored in the comparison."
                end

                # Filter out NaN values for comparison
                valid_indices = @. !isnan(expected) && !isnan(computed)
                expected = expected[valid_indices]
                computed = computed[valid_indices]

                difference = expected .- computed
                max_difference = maximum(abs.(difference))
                mean_difference = mean(abs.(difference))
                std_difference = std(abs.(difference))

                push!(
                    generated_statistics[observable_name],
                    [mean_difference, std_difference, max_difference],
                )
            end
        end
    end

    # Stop Julia from garbage-collecting the reader (https://github.com/fhs/ZipFile.jl/issues/14#issuecomment-1135397765)
    close(grids_zip)

    @info "Writing results..."

    # Format the statistics into something more human-readable
    formatted_statistics = Dict{String,Vector{Any}}()
    for (name, values) in generated_statistics
        if name in keys(test_observables)
            formatted_statistics[name*" Difference"] = map(values) do value
                mean_difference = value[1]
                std_difference = value[2]
                max_difference = value[3]
                return "$mean_difference ± $std_difference (max: $max_difference)"
            end
        else
            formatted_statistics[name] = values
        end
    end

    Base.Filesystem.mkpath("output")
    CSV.write("output/n2_test_results.csv", formatted_statistics)

    # Print a summary
    summary = "Summary of Results:\n"
    for (observable_name, statistics) in generated_statistics
        if !(observable_name in keys(test_observables))
            continue
        end

        mean_difference = mean(s[1] for s in statistics)
        max_difference = mean(s[3] for s in statistics)

        summary *= "  $observable_name differed by: $mean_difference on average with a max of $max_difference\n"
    end
    @info summary

    @info "Done!"
end

end
