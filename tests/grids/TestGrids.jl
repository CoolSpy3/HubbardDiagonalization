module TestGrids

export main

import HubbardDiagonalization
import HubbardDiagonalization.Graphs
import HubbardDiagonalization.CSVUtil

import CSV

using Logging
using Statistics
using ZipFile

function (@main)(args)
    warn_on_nan = false

    datasets = ["./tests/grids/N2_grids.zip", "./tests/grids/2x2_ED_Repulsive_SUN_OBC.zip"]
    # Setup mappings between result names and csv files
    test_observables = Dict(
        "Density" => "Densities.csv",
        "Double Occupancies" => "Doubleoccupancies.csv",
        "Energy" => "Energies.csv",
        "Entropy" => "Entropies.csv",
    )

    if !warn_on_nan
        @warn "NaN warnings are disabled."
    end

    graph = Graphs.linear_chain(4)

    generated_statistics = Dict{String,Vector{Any}}("N" => [], "U" => [])
    for observable_name in keys(test_observables)
        generated_statistics[observable_name] = []
    end

    for zip_file in datasets
        grids_zip = ZipFile.Reader(zip_file)

        for test_set in grids_zip.files
            parsed_name = match(r"2x2_N(\d+)_U(\d+)(_Tmu.zip|/$)", test_set.name)
            if parsed_name === nothing
                continue
            end

            N = parse(Int, parsed_name.captures[1])
            U = parse(Float64, parsed_name.captures[2])

            is_nested_zip = endswith(test_set.name, ".zip")
            # Reading Nested Zip Files: https://stackoverflow.com/a/44877369
            zip_reader =
                is_nested_zip ? ZipFile.Reader(IOBuffer(read(test_set))) : grids_zip
            prefix = is_nested_zip ? "" : test_set.name
            run_test_set(
                graph,
                generated_statistics,
                zip_reader,
                prefix,
                N,
                U,
                test_observables,
                warn_on_nan,
            )
        end

        # Stop Julia from garbage-collecting the reader (https://github.com/fhs/ZipFile.jl/issues/14#issuecomment-1135397765)
        close(grids_zip)
    end

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
    CSV.write("output/grids_test_results.csv", formatted_statistics)

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

function run_test_set(
    graph::Graphs.Graph,
    generated_statistics::Dict{String,Vector{Any}},
    zip_reader::ZipFile.Reader,
    prefix::String,
    N::Int,
    U::Float64,
    test_observables::Dict{String,String},
    warn_on_nan::Bool,
)
    test_config = HubbardDiagonalization.TestConfiguration(
        num_colors = N,
        t = 2.0,
        u_test = 0.0,
        U = U,
    )

    observables = HubbardDiagonalization.default_observables(test_config, graph)

    @info "Running tests for N=$N, U=$U..."

    u_vals = CSVUtil.read_zipped_csv(prefix, "mu_vals.csv", zip_reader)[:, 1]
    T_vals = CSVUtil.read_zipped_csv(prefix, "T_vals.csv", zip_reader)[:, 1]

    # Map file names to expected data
    expected_data = Dict{String,AbstractMatrix{Float64}}()

    # Read in the test files
    for results_file in unique(values(test_observables))
        expected = CSVUtil.read_zipped_csv(prefix, results_file, zip_reader)
        @assert size(expected) == (length(T_vals), length(u_vals)) "Expected data size mismatch for $results_file"
        expected_data[results_file] = expected
    end

    # Temporarily disable info logging for cleaner test output
    disable_logging(Logging.Info)
    results = HubbardDiagonalization.diagonalize_and_compute_observables(
        T_vals,
        u_vals,
        test_config,
        graph,
        observables...,
    )
    disable_logging(Logging.Debug)

    push!(generated_statistics["N"], N)
    push!(generated_statistics["U"], U)
    for (observable_name, results_file) in test_observables
        expected = expected_data[results_file]
        computed = results[observable_name]

        @assert size(expected) == size(computed) "Size mismatch for observable $observable_name at N=$N, U=$U"

        if warn_on_nan && any(isnan, expected)
            @warn "Expected data contains NaN values for N=$N, U=$U, observable=$observable_name at " *
                  "$(u_vals[findfirst(isnan, expected)]) < u < $(u_vals[findlast(isnan, expected)]). " *
                  "These values will be ignored in the comparison."
        end
        if warn_on_nan && any(isnan, computed)
            @warn "Computed data contains NaN values for N=$N, U=$U, observable=$observable_name at " *
                  "$(u_vals[findfirst(isnan, computed)]) < u < $(u_vals[findlast(isnan, computed)]). " *
                  "These values will be ignored in the comparison."
        end

        # Filter out NaN values for comparison
        valid_indices = @. !isnan(expected) && !isnan(computed)
        expected = expected[valid_indices]
        computed = computed[valid_indices]

        if observable_name == "Energy" || observable_name == "Entropy"
            computed ./= Graphs.num_sites(graph)  # Normalize per site
        end

        difference = @. abs(expected - computed)
        max_difference = maximum(difference)
        mean_difference = mean(difference)
        std_difference = std(difference)

        push!(
            generated_statistics[observable_name],
            [mean_difference, std_difference, max_difference],
        )
    end
end

end
