module CSVUtil

export load_csv_matrix, find_zipped_file, read_zipped_csv, load_overlay_data

import CSV
import ZipFile

function load_csv_matrix(path_or_file::Union{String,ZipFile.ReadableFile})
	return CSV.File(path_or_file, header = false) |> CSV.Tables.matrix
end

function find_zipped_file(prefix::String, name::String, zip_reader::ZipFile.Reader)
	matching_files =
		filter(f -> startswith(f.name, prefix) && endswith(f.name, name), zip_reader.files)
	@assert length(matching_files) == 1 "Expected exactly one file matching $name with prefix $prefix, found $(length(matching_files))"
	return matching_files[1]
end

function read_zipped_csv(prefix::String, name::String, zip_reader::ZipFile.Reader)
    return load_csv_matrix(find_zipped_file(prefix, name, zip_reader))
end

function load_overlay_data(path::String, zip_reader::Union{ZipFile.Reader,Nothing}=nothing)
    if !endswith(path, "/")
        path *= "/"
    end
    if zip_reader === nothing && !contains(path, ".zip/")
        # Load directly from Filesystem
		u_vals = load_csv_matrix(path * "mu_vals.csv")[:, 1]
		T_vals = load_csv_matrix(path * "T_vals.csv")[:, 1]
        overlay_data = Dict{String,Matrix{Float64}}()
        for file in readdir(path)
            if endswith(file, ".csv") && file != "mu_vals.csv" && file != "T_vals.csv"
                observable_name = replace(file, ".csv" => "")
                data_matrix = load_csv_matrix(path * file)
                overlay_data[observable_name] = data_matrix
            end
        end
        return u_vals, T_vals, overlay_data
    elseif contains(path, ".zip/")
		# Unpack zip and recurse
        zip_path = first(split(path, ".zip/")) * ".zip"
        internal_path = join(split(path, ".zip/")[2:end], ".zip/")
		if zip_reader === nothing
			# Top-level zip file
			zip_reader = ZipFile.Reader(zip_path)
			return load_overlay_data(internal_path, zip_reader)
		else
			# Nested zip file
			nested_zip = find_zipped_file("", zip_path, zip_reader)
			nested_zip_reader = ZipFile.Reader(IOBuffer(read(nested_zip)))
			return load_overlay_data(internal_path, nested_zip_reader)
		end
	else
		# Load from ZipFile.Reader
		u_vals = read_zipped_csv(path, "mu_vals.csv", zip_reader)[:, 1]
		T_vals = read_zipped_csv(path, "T_vals.csv", zip_reader)[:, 1]
		overlay_data = Dict{String,Matrix{Float64}}()
		for file in zip_reader.files
			if startswith(file.name, path) && endswith(file.name, ".csv") &&
			   !endswith(file.name, "mu_vals.csv") && !endswith(file.name, "T_vals.csv")
				observable_name = replace(basename(file.name), ".csv" => "")
				data_matrix = read_zipped_csv(path, basename(file.name), zip_reader)
				overlay_data[observable_name] = data_matrix
			end
		end
		return u_vals, T_vals, overlay_data
	end
end

end
