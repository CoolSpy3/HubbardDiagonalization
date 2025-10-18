include("graphs.jl")
include("state_enumeration.jl")
include("symmetric_matrices.jl")

import LinearAlgebra

using .Graphs
using .StateEnumeration
using .SymmetricMatrices

const use_unicode_plots = false

try
	import CSV
	import Plots
	if use_unicode_plots
		import UnicodePlots
	end
catch ImportError
	@info "Installing required packages..."
	import Pkg

	Pkg.add("CSV")
	import CSV
	Pkg.add("Plots")
	import Plots

	if use_unicode_plots
		Pkg.add("UnicodePlots")
		import UnicodePlots
	end
end
using Plots
if use_unicode_plots
	unicodeplots()
end

function (@main)(args)
	# Parameters
	num_colors = 3

	t = 0.0
	T = 0.07
	u_test = 0.0
	U = 1.0

	u_min = -5
	u_max = 5
	u_step = 0.01

	@info "Initialized with t=$t, T=$T, u_step=$u_step, U=$(U)"

	B = 1/T

	# Observables
	observables = Dict{String, Function}()

	# observables["Energy"] = (eigen_value, _) -> eigen_value
	observables["Num_Particles"] = state -> sum(count_ones(color) for color in state)
	observables["Filled States"] = state -> count_ones(reduce(&, state))

	"""
		create_observable_data_map()

	A convenience function to create an empty map from observable names to data vectors.
	"""
	# We're going to need a few of these. Might as well make it a function.
	function create_observable_data_map()
		map = Dict{String, Vector{Float64}}()
		for observable_name in keys(observables)
			map[observable_name] = Float64[]
		end
		return map
	end

	@info "Defined observables: $(keys(observables))"

	@info "Creating graph..."
	graph = linear_chain(1)

	num_sites = Graphs.num_sites(graph)
	N_max_fermions = num_colors * num_sites
	@debug "N_max_fermions=$N_max_fermions"

	# Initialize some containers to store the generated data
	weights = Float64[]  # Weights for each state
	n_fermion_data = Int[]  # Number of fermions for each state (used for re-weighting)
	observable_data = create_observable_data_map()

	@info "Computing Hamiltonian blocks and observables..."
	# The number of fermions and the color configuration are conserved over tunneling,
	# so we can break the Hamiltonian into blocks labeled by these two quantities
	for N_fermions in 0:N_max_fermions
		for color_configuration in color_configurations(N_fermions, num_sites, num_colors)
			# Size of the Hamiltonian block
			L = prod(binomial(num_sites, n) for n in color_configuration)
			H = SymmetricMatrix(L)  # Use custom "SymmetricMatrix" type to save memory at the cost of speed
			observables_basis = create_observable_data_map()  # Compute the observables for each state as we build the matrix

			# Compute Hamiltonian matrix elements between all pairs of states
			# enumerate_multistate returns elements in a consistent order, so
			# as long as we're consistent, the matrix elements will be in the right place
			# state_i and state_j are arrays of integers, where each integer is a bitmask
			# representing the occupation of each site for a given color
			for (i, state_i) in enumerate(enumerate_multistate(num_sites, color_configuration))
				# Note: We're going to cut this inner loop off early since the matrix is symmetric
				for (j, state_j) in enumerate(enumerate_multistate(num_sites, color_configuration))
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

						# Verify that the hop is allowed by the graph
						if has_edge(graph, site_1, site_2)
							# If so, the sign will be flipped if an odd number of
							# spots are *between* the two sites.
							# Note that this refers to the representation of the
							# spots not how they're related by the graph
							sign = iseven(site_1 - site_2) ? -1 : 1
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
			eig = LinearAlgebra.eigen(H_symmetric_view)

			# Compute and store observables for each eigenstate
			for (eigen_val, eigen_vec) in zip(eig.values, eachcol(eig.vectors))
				@debug begin "  eigen_val=$eigen_val, eigen_vec=$eigen_vec" end

				# Weight data of this state in the partition function
				weight = num_permutations * exp(-B * eigen_val)
				push!(weights, weight)
				push!(n_fermion_data, N_fermions)

				# Compute each observable for this state
				# Because we already computed the observables for each basis state,
				# we can just do a weighted sum over those based on the eigenvector components
				for (observable_name, observable_basis_data) in observables_basis
					push!(observable_data[observable_name], sum(observable_basis_data .* eigen_vec))
				end
			end
		end
	end

	@info "Computed data for $(length(weights)) states."

	@info "Computing derived observables..."

	# Compute derived observables
	# observable_data["m^2"] = @. observable_data["Num_Particles"] - 2 * observable_data["Doublons"]

	@info "Computing observables over range of u..."

	u_range = u_min:u_step:u_max
	# Create a new container to store the observable values at each u
	observable_values = create_observable_data_map()
	for u in u_range
		# Re-weight the data according to the new u value
		weight_correction = exp.(-B * (-(u - u_test)) .* n_fermion_data)

		# Compute the partition function
		Z = sum(weight_correction .* weights)

		@debug begin "u=$u, Z=$Z, weights=$weights, weight_corrections=$weight_correction" end

		# Compute each observable
		for (observable_name, observable_data) in observable_data
			@debug begin "  $observable_name: $observable_data" end
			# Compute the expectation value of each observable
			expectation_value = sum(weight_correction .* weights .* observable_data) / Z
			push!(observable_values[observable_name], expectation_value)
		end
	end

	@info "Exporting observable data..."

	Base.Filesystem.mkpath("output")
	CSV.write("output/observable_data.csv", merge(Dict("u" => u_range), observable_values))

	@info "Plotting observables..."

	# Initialize the plot
	graph = plot(
			xlabel="u",
			ylabel="Observable Value",
			title="t=$t, T=$T, U=$U, num_sites=$(num_sites), num_colors=$(num_colors)",
			legend=:topright,
			size=(800,600)
		)
	# Plot each observable
	for (name, values) in observable_values
		graph = plot!(graph, u_range, values, labels=name)
	end

	# Save the plot
	savefig(graph, "output/observables_u.png")
	# display(graph)

	@info "Done."
end
