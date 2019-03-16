using Flux, MLDataPattern
using Flux: onehotbatch
include("spectrograms.jl")

# Image size constants
img_n = 150
img_m = 100

# Definition of a dataset with training and testing partitions
struct DataSet
  X_train::Array{Float32, 2}
  Y_train::Flux.OneHotMatrix
  X_test::Array{Float32, 2}
  Y_test::Flux.OneHotMatrix
end

# Split the data
function create_data_splits(n=img_n, m=img_m)
  @info("Generating spectrograms of size $(n)x$(m)...")

  # Read labels from source CSV file
  file_to_class = gen_label_dict("train.csv")
  labels = simple_labels(file_to_class)

  # Create spectrograms with values between 0 and 1 (via sigmoid)
  spec_base = create_spectrogram_base(file_to_class, n, m)

  # Format dataset
  X = hcat(float.(reshape.(spec_base, :))...) |> gpu
  XY = vcat(X, labels')

  get_label(x) = convert(Int, x[end])

  @info("Generating train and test splits...")

  # Stratified split: 80% for training and 20% for testing
  # Encode Ys with one-hot encoding
  train_data, test_data = stratifiedobs(get_label, XY, 0.8)
  X_train = train_data[1:end-1,:]
  Y_train = onehotbatch(convert(Array{Int}, train_data[end,:]), 0:9)
  X_test = test_data[1:end-1,:]
  Y_test = onehotbatch(convert(Array{Int}, test_data[end,:]), 0:9)

  @info("Done")

  return DataSet(X_train, Y_train, X_test, Y_test)
end
