module HubbardDiagonalization

export main

# Include our submodules
include("Graphs.jl")
include("StateEnumeration.jl")
include("SymmetricMatrices.jl")

using .Graphs
using .StateEnumeration
using .SymmetricMatrices

# Import libraries
const use_unicode_plots = false

import CSV
import LinearAlgebra
import Logging
import Plots
import TOML

# Set up plotting backend
using Plots
if use_unicode_plots
	import UnicodePlots
	unicodeplots()
end

function (@main)(args)
	# Install our own logger for the duration of the program
	old_logger = Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Debug))
	if "--debug" in args
		@warn "Running in debug mode!"
		sleep(5)  # Give user time to see the warning
	else
		# In normal mode disable debug logging
		Logging.disable_logging(Logging.Debug)
	end

	# Load Parameters
	config = TOML.parsefile("SimulationConfig.toml")
	params = config["parameters"]
	num_colors::Int = params["num_colors"]

	t::Float64 = params["t"]
	T::Float64 = params["T"]
	u_test::Float64 = params["u_test"]
	U::Float64 = params["U"]

	plot_config = config["plot"]
	u_min::Float64 = plot_config["u_min"]
	u_max::Float64 = plot_config["u_max"]
	u_step::Float64 = plot_config["u_step"]
	plot_width::Int = plot_config["width"]
	plot_height::Int = plot_config["height"]

	graph_config = config["graph"]
	graph = linear_chain(graph_config["num_sites"])
	num_sites = Graphs.num_sites(graph)

	@info "Initialized with t=$t, T=$T, u_step=$u_step, U=$(U)"
	@debug begin "  Graph edges: $(Graphs.edges(graph))" end

	B = 1/T

	# Observables
	observables = Dict{String, Function}()

	observables["Num_Particles"] = state -> sum(count_ones(color) for color in state)
	observables["Double Occupancy"] = state -> count_ones(reduce(&, state))

	observables["Energy"] = _ -> 0.0  # Will be handled specially

	# Observables that can be calculated from other observables
	derived_observables = Dict{String, Function}()
	derived_observables["Local Moment"] = observable_data -> @. observable_data["Num_Particles"] - 2 * observable_data["Double Occupancy"]
	derived_observables["Density"] = observable_data -> observable_data["Num_Particles"] ./ Graphs.num_sites(graph)
	derived_observables["Entropy"] = _ -> Float64[]  # Will be handled specially

	# Additional Plots that can be directly calculated
	overlays = Dict{String, Function}()

	if num_sites == 1
		# Single-site Hubbard model exact solutions
		# e0(n, u) = U * binomial(n, 2) - (u #= + (U / 2) * (num_colors - 1) =#) * n
		e0(n, u) = U * binomial(n, 2) - (u + (U / 2) * (num_colors - 1)) * n
		weighted_sum(u, f) = sum(binomial(num_colors, n) * f(n) * exp(-B * e0(n, u)) for n in 0:num_colors)
		z0(u) = weighted_sum(u, n -> 1)
		rho(u) = (1/z0(u)) * weighted_sum(u, n -> n)
		# energy(u) = (1/z0(u)) * weighted_sum(u, n -> e0(n, u) #= + (u + (U/2) * (num_colors - 1)) *  n =#)
		energy(u) = (1/z0(u)) * weighted_sum(u, n -> e0(n, u) + (u + (U/2) * (num_colors - 1)) * n)
		overlays["Actual Energy"] = energy
		# overlays["Actual Entropy"] = u -> log(z0(u)) + B * (energy(u) #= - (u + (U/2) * (num_colors - 1)) * rho(u) =#)
		overlays["Actual Entropy"] = u -> log(z0(u)) + B * (energy(u) - (u + (U/2) * (num_colors - 1)) * rho(u))
	end

	"""
		create_observable_data_map(include_derived::Bool, include_overlays::Bool)

	include_derived: Whether to include derived observables in the map.
	include_overlays: Whether to include overlay functions in the map.

	A convenience function to create an empty map from observable names to data vectors.
	"""
	# We're going to need a few of these. Might as well make it a function.
	function create_observable_data_map(include_derived::Bool, include_overlays::Bool)
		map = Dict{String, Vector{Float64}}()
		for observable_name in keys(observables)
			map[observable_name] = Float64[]
		end
		if include_derived
			for derived_name in keys(derived_observables)
				map[derived_name] = Float64[]
			end
		end
		if include_overlays
			for overlay_name in keys(overlays)
				map[overlay_name] = Float64[]
			end
		end
		return map
	end

	@info "Defined observables: $(union(keys(observables), keys(derived_observables), keys(overlays)))"

	N_max_fermions = num_colors * num_sites
	@debug "N_max_fermions=$N_max_fermions"

	# Initialize some containers to store the generated data
	weights = Float64[]  # Weights for each state
	n_fermion_data = Int[]  # Number of fermions for each state (used for re-weighting)
	observable_data = create_observable_data_map(false, false)

	@info "Computing Hamiltonian blocks and observables..."
	# The number of fermions and the color configuration are conserved over tunneling,
	# so we can break the Hamiltonian into blocks labeled by these two quantities
	for N_fermions in 0:N_max_fermions
		for color_configuration in color_configurations(N_fermions, num_sites, num_colors)
			# Size of the Hamiltonian block
			L = prod(binomial(num_sites, n) for n in color_configuration)
			H = SymmetricMatrix(L)  # Use custom "SymmetricMatrix" type to save memory at the cost of speed
			observables_basis = create_observable_data_map(false, false)  # Compute the observables for each state as we build the matrix

			# Compute Hamiltonian matrix elements between all pairs of states
			# enumerate_multistate returns elements in a consistent order, so
			# as long as we're consistent, the matrix elements will be in the right place
			# state_i and state_j are arrays of integers, where each integer is a bitmask
			# representing the occupation of each site for a given color
			for (i, state_i) in enumerate(enumerate_multistate(num_sites, color_configuration))
				# Note: We're going to cut this inner loop off early since the matrix is symmetric
				for (j, state_j) in enumerate(enumerate_multistate(num_sites, color_configuration))
					@debug begin "Computing H[$i,$j] between states:\n  state_i=$(digits.(state_i, base=2, pad=num_sites))\n  state_j=$(digits.(state_j, base=2, pad=num_sites))" end
					if i == j  # Because enumerate_multistate is consistent, if the indices are equal, the states are equal
						# Diagonal element
						H[i,i] = -u_test * N_fermions

						# Interaction term
						# Consider two colors at a time to interact
						for interacting_colors in enumerate_states(num_colors, 2)
							# Count number of pairs of fermions on the same site
							color_mask = digits(interacting_colors, base=2, pad=num_colors)
							# Set bits for colors **not** in the interaction to 1
							color_mask = 1 .- color_mask
							# For all colors not in the interaction, set all bits to 1 (mark all sites as occupied)
							color_mask = color_mask .* ((2 ^ num_sites) - 1)  # 1 -> (111...1), 0 -> 0
							occupied_sites = state_i .| color_mask
							# Take the bitwise AND across all colors to find sites occupied by both colors
							occupied_sites = reduce(&, occupied_sites)
							# Add interaction energy for each pair of fermions on the same site
							H[i,i] += U * count_ones(occupied_sites)
						end

						break  # No need to compute upper-triangular elements
					else
						# Off-diagonal element
						H[i,j] = 0.0

						# Hopping term
						# First, compute the difference between the two states
						# This is a bitmask where bits are 1 if a fermion appeared or disappeared
						# Because the total number of fermions is fixed, if only two
						# bits are set, then one fermion hopped from one site to another
						diff = state_i .⊻ state_j
						# Figure out which color hopped
						# If more than one color hopped, then each individual hopping term is zero
						# so their sum is also zero
						hopped_color = 0
						for color in 1:num_colors
							if diff[color] != 0
								if count_ones(diff[color]) != 2 || hopped_color != 0
									# More than one color or more than one fermion hopped,
									# so this matrix element is zero
									hopped_color = -1
									break
								end
								hopped_color = color
							end
						end
						@assert hopped_color != 0  # States must be different
						if hopped_color == -1
							continue
						end

						# Get the sites involved in the hop
						hopped_sites = digits(diff[hopped_color], base=2, pad=num_sites)
						site_1 = findfirst(hopped_sites .== 1)
						site_2 = findlast(hopped_sites .== 1)
						@assert site_1 != site_2

						@debug begin "Considering hop of color $hopped_color from site $site_1 to site $site_2" end

						# Verify that the hop is allowed by the graph
						if has_edge(graph, site_1, site_2)
							# If so, the sign will be flipped if an odd number of
							# spots are occupied *between* the two sites.
							# Note that this refers to the representation of the
							# spots not how they're related by the graph

							occupied_sites = state_i[hopped_color] & state_j[hopped_color]
							# Create a mask with 1s in all bits between site_1 and site_2.
							# Note that julia's one-based indexing actually works out here
							# If site=2 (so it's referring to the 2nd bit in the bitmask),
							# Then, (1 << site) = 0b100, and (1 << site) - 1 = 0b011
							# (Keep in mind that digits is interpreting this in little-endian,
							# so the second-least-significant bit is the "2nd" bit)
							# With this in mind, we can calculate the mask by taking the
							# mask for all bits at or below site_2 (which is larger than site_1)
							# and ANDing it with all of the bits above site_1 (which is just the
							# same algorithm negated)
							# Note that because the hop occurs between site_1 and site_2,
							# both site_1 and site_2 are 0 in occupied_sites, so it doesn't
							# matter if we include them in the mask or not.
							bitween_mask = ((1 << site_2) - 1) & ~((1 << site_1) - 1)
							sign = iseven(count_ones(occupied_sites & bitween_mask)) ? 1 : -1
							@debug begin "  Hop is allowed by graph! sign=$sign" end
							H[i,j] = sign * (-t)
						end
					end
				end

				# Now that we've constructed the row for state_i, compute the observables
				for (observable_name, observable_function) in observables
					# Pre-compute the observable for this basis state
					push!(observables_basis[observable_name], observable_function(state_i))
				end
			end

			num_permutations = num_configuration_permutations(color_configuration)

			@debug begin "N_fermions=$N_fermions, color_configuration=$(color_configuration), L=$L, num_configuration_permutations=$(num_permutations), H=$H" end

			# Diagonalize the Hamiltonian block
			H_symmetric_view = LinearAlgebra.Symmetric(H, :L)
			# Annoyingly, eigen() forces us to store all the eigenvectors
			# in memory at once, but I can't find a good way around this
			# Even the builtin `eigvals`/`eigvecs` functions are just
			# wrappers around this.
			eigen_data = LinearAlgebra.eigen(H_symmetric_view)

			@debug begin
				msg = "observable_basis_data:\n"
				for (name, data) in observables_basis
					msg *= "  $name: $data\n"
				end
				msg
			end

			# Compute and store observables for each eigen-state
			for (eigen_val, eigen_vec) in zip(eigen_data.values, eachcol(eigen_data.vectors))
				@debug begin "  eigen_val=$eigen_val, eigen_vec=$eigen_vec" end

				# eigen() returns normalized eigenvectors, so we don't need to do any normalization here

				# Weight data of this state in the partition function
				weight = num_permutations * exp(-B * eigen_val)
				push!(weights, weight)
				push!(n_fermion_data, N_fermions)

				# Compute each observable for this state
				for (observable_name, observable_basis_data) in observables_basis
					if observable_name == "Energy"
						# The energy is just the eigenvalue
						push!(observable_data[observable_name], eigen_val)
					else
						# Because we already computed the observables for each basis state,
						# we can just do a weighted sum over those based on the eigenvector components
						observable_value = sum(@. observable_basis_data * eigen_vec * eigen_vec)
						push!(observable_data[observable_name], observable_value)
					end
				end
			end
		end
	end

	@info "Computed data for $(length(weights)) states."

	@info "Computing derived observables..."

	for (observable_name, observable_function) in derived_observables
		observable_data[observable_name] = observable_function(observable_data)
	end

	@debug begin
		msg = "Computed data:\n"
		msg *= "  weights: $weights\n"
		msg *= "  n_fermion_data: $n_fermion_data\n"
		for (name, data) in observable_data
			msg *= "  $name: $data\n"
		end
		msg
	end

	@info "Computing observables over range of u..."

	u_range = u_min:u_step:u_max
	u_shift = (U/2) * (num_colors - 1)  # Shift observables so that density=N/2 at u=0
	# Create a new container to store the observable values at each u
	computed_observable_values = create_observable_data_map(true, true)
	for u in u_range
		# The value that has to be added to u_test to shift to the desired u
		u_datapoint_shift = u - u_test + u_shift

		# Re-weight the data according to the new u value
		weight_correction = exp.((-B * -u_datapoint_shift) .* n_fermion_data)

		# Compute the partition function
		Z = sum(weight_correction .* weights)

		@debug begin "u=$u, weight_corrections=$weight_correction, Z=$Z" end

		# Compute each observable
		for (observable_name, observable_values) in observable_data
			if observable_name == "Energy"
				# Now, update the energy. The free energy is H + u * N, so it works out to
				observable_values = (-u_test * n_fermion_data) .+ observable_values
				@debug begin "  Updated Energy data: $observable_values" end

				# While we're here, calculate the entropy
				# The expectation value of the Hamiltonian depends on u, so we have to shift it here
				internal_energy_values = (-u_datapoint_shift * n_fermion_data) .+ observable_values
				internal_energy_expectation = sum(@. weight_correction * weights * internal_energy_values) / Z
				entropy_expectation = internal_energy_expectation * B + log(Z)
				push!(computed_observable_values["Entropy"], entropy_expectation)
				@debug begin "  Entropy: $entropy_expectation (Internal Energy Values: $internal_energy_values Internal Energy Expectation: $internal_energy_expectation)" end
			elseif observable_name == "Entropy"
				# Calculated above
				continue
			end

			# Compute the expectation value of each observable
			expectation_value = sum(@. weight_correction * weights * observable_values) / Z
			@debug begin "  $observable_name: $expectation_value" end
			push!(computed_observable_values[observable_name], expectation_value)
		end

		for (overlay_name, overlay_function) in overlays
			push!(computed_observable_values[overlay_name], overlay_function(u))
		end

		# Do this at the end to ensure we know the energy value
	end

	@info "Exporting observable data..."

	Base.Filesystem.mkpath("output")
	CSV.write("output/observable_data.csv", merge(Dict("u" => u_range), computed_observable_values))

	@info "Plotting observables..."

	# Initialize the plot
	graph = plot(
			xlabel="u",
			ylabel="Observable Value",
			title="t=$t, T=$T, U=$U, num_sites=$(num_sites), num_colors=$(num_colors)",
			legend=:topright,
			size=(plot_width, plot_height)
		)
	# Plot each observable
	for (name, values) in computed_observable_values
		graph = plot!(graph, u_range, values, labels=name)
	end

	# Save the plot
	savefig(graph, "output/observable_data.png")
	# display(graph)

	@info "Done."

	# Restore the old logger after we're done
	Logging.global_logger(old_logger)

	return 0
end

end
