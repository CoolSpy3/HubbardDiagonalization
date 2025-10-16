"Defines a SymmetricMatrix type that stores only the lower-triangular part of the matrix."
module SymmetricMatrices

export SymmetricMatrix

import LinearAlgebra

"A matrix of floats that is symmetric and memory-efficient."
struct SymmetricMatrix <: AbstractMatrix{Float64}
	data::Vector{Float64}
	size::Int
end

"""
	SymmetricMatrix(size::Int)

Create a `size x size` symmetric matrix initialized to zero.
"""
function SymmetricMatrix(size::Int)
	return SymmetricMatrix(Base.zeros(Float64, size * (size + 1) ÷ 2), size)
end

# Make the matrix lower-triangular for more efficient iteration
"""
	index(mat::SymmetricMatrix, i::Int, j::Int)

Get the index in the internal data vector corresponding to the (i,j) entry of the matrix.
"""
function index(mat::SymmetricMatrix, i::Int, j::Int)
	@assert i <= mat.size && j <= mat.size
	# Convert upper-triangular indices to lower-triangular
	# (Because the matrix is symmetric, they refer to the same value)
	if j > i
		i,j = j,i
	end
	# Work around julia's 1-based indexing
	i -= 1
	j -= 1
	# The number of elements in the ith row (0-indexed) is i+1, so the number of elements in
	# all previous rows is the sum of the first i integers (i*(i+1)/2). Then add j to get the column offset,
	# and add 1 to convert back to 1-based indexing.
	return ((i * (i + 1) ÷ 2) + j) + 1
end

# Implement some core functions of the AbstractMatrix interface
function Base.getindex(mat::SymmetricMatrix, i::Int, j::Int)
	return mat.data[index(mat, i, j)]
end

function Base.setindex!(mat::SymmetricMatrix, value::Float64, i::Int, j::Int)
	mat.data[index(mat, i, j)] = value
end

function Base.size(mat::SymmetricMatrix)
	return (mat.size, mat.size)
end

function Base.length(mat::SymmetricMatrix)
	return mat.size * (mat.size + 1) ÷ 2
end

# Scalar multiplication. Use this in case LinearAlgebra.rmul! tries to multiply
# off-diagonal elements twice. Call explicitly instead of exporting to avoid
# name conflicts.
# Turns out this wasn't actually useful. Kept in case someone needs it later.
# function rmul!(mat::SymmetricMatrix, scalar::Number)
# 	LinearAlgebra.rmul!(mat.data, scalar)
# 	return mat
# end

end
