module CSVUtil

export load_csv_matrix, find_zipped_file, read_zipped_csv, load_overlay_data

import CSV
import ZipFile

"""
    load_csv_matrix(path_or_file::Union{String,ZipFile.ReadableFile})

Load a CSV file from the given path or ZipFile.ReadableFile and return its contents as a Matrix{Float64}.
Assumes that the CSV file has no header row.
"""
function load_csv_matrix(path_or_file::Union{String,ZipFile.ReadableFile})
    return CSV.File(path_or_file, header = false) |> CSV.Tables.matrix
end

"""
    find_zipped_file(prefix::String, name::String, zip_reader::ZipFile.Reader)

Find a file within a ZipFile.Reader with the given name. Prefix can be used to specify
a string that must precede the name (e.g. a directory path within the zip file).
Only one matching file is allowed. If multiple or no files are found, an error is raised.

Returns the ZipFile.ReadableFile corresponding to the found file.

prefix -- The prefix string that the file name must start with.
name -- The name of the file to find. (Will be matched to the end of the file name.)
zip_reader -- The ZipFile.Reader to search within.
"""
function find_zipped_file(prefix::String, name::String, zip_reader::ZipFile.Reader)
    matching_files =
        filter(f -> startswith(f.name, prefix) && endswith(f.name, name), zip_reader.files)
    @assert length(matching_files) == 1 "Expected exactly one file matching $name with prefix $prefix, found $(length(matching_files))"
    return matching_files[1]
end

"""
    read_zipped_csv(prefix::String, name::String, zip_reader::ZipFile.Reader)

This is a convenience function that combines `find_zipped_file` and `load_csv_matrix` to read a CSV file from within a zip archive.
See those functions for details.
"""
function read_zipped_csv(prefix::String, name::String, zip_reader::ZipFile.Reader)
    return load_csv_matrix(find_zipped_file(prefix, name, zip_reader))
end

"""
    load_overlay_data(path::String, zip_reader::Union{ZipFile.Reader,Nothing}=nothing)

Load observable data from the specified path, which can be a directory on the filesystem or a path within a zip archive
(ex. "data.zip/path/to/nested.zip/path/).
Returns a tuple (u_vals, T_vals, overlay_data) where:
- u_vals: Vector{Float64} of chemical potential values loaded from "mu_vals.csv"
- T_vals: Vector{Float64} of temperature values loaded from "T_vals.csv"
- overlay_data: Dict{String, Matrix{Float64}} mapping observable names to their corresponding data matrices.

path -- The path to load data from. Can be a directory or a path within a zip archive. The last path element must be
a directory or zip archive containing the CSV files.
zip_reader -- (optional) If provided, uses the given ZipFile.Reader as the base of the path.
"""
function load_overlay_data(
    path::String,
    zip_reader::Union{ZipFile.Reader,Nothing} = nothing,
)
    # Ensure path ends with a slash so we can easily find zipped files within it by searching for `.zip/`
    if !endswith(path, "/")
        path *= "/"
    end
    if zip_reader === nothing && !contains(path, ".zip/")
        # Load directly from Filesystem

        # Values are contained in the first column
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
        split_path = split(path, ".zip/")
        zip_path = first(split_path) * ".zip"
        internal_path = join(split_path[2:end], ".zip/")
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

        # Values are contained in the first column
        u_vals = read_zipped_csv(path, "mu_vals.csv", zip_reader)[:, 1]
        T_vals = read_zipped_csv(path, "T_vals.csv", zip_reader)[:, 1]

        overlay_data = Dict{String,Matrix{Float64}}()
        for file in zip_reader.files
            if startswith(file.name, path) &&
               endswith(file.name, ".csv") &&
               !endswith(file.name, "mu_vals.csv") &&
               !endswith(file.name, "T_vals.csv")
                observable_name = replace(basename(file.name), ".csv" => "")
                data_matrix = read_zipped_csv(path, basename(file.name), zip_reader)
                overlay_data[observable_name] = data_matrix
            end
        end
        return u_vals, T_vals, overlay_data
    end
end

end
