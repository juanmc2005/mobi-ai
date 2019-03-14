using Flux, Statistics, MLDataPattern
using Flux: onehotbatch, onecold, crossentropy, throttle
# using Base.Iterators: repeated

include("myplots.jl")
include("spectrograms.jl")

img_n = 150
img_m = 100

epochs = 300
eta = 0.001

mutable struct DataSet
  X_train::Array{Float32, 2}
  Y_train::Flux.OneHotMatrix
  X_test::Array{Float32, 2}
  Y_test::Flux.OneHotMatrix
end

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

# Train the model
function train_ann(d::DataSet; hidden=750, epochs=epochs, eta=eta, plotloss=true)
  @info("Constructing ANN model...")
  m = Chain(
    Dense(size(d.X_train[:,1],1), hidden, relu),
    Dense(hidden, size(d.Y_train[:,1],1)),
    softmax)

  loss(x, y) = crossentropy(m(x), y)
  accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

  dataset = [(d.X_train, d.Y_train)]
  evalcb = () -> @show(loss(d.X_train, d.Y_train))
  opt = ADAM(eta)

  # Check for NaN values so we know if we have bad data before training
  # @info("Checking for NaN values")
  # for i in 1:size(X_train,2)
  #   if (isnan(loss(X_train[:,i], Y_train[:,i])))
  #     @info("$i is NaN")
  #   end
  # end
  # @info("Done.")

  @info("Training model...")

  # Train for a specified number of epochs
  if plotloss
    plt = create_plot(0, loss(d.X_train, d.Y_train).data)
  end
  for step in 1:epochs
    @info("Epoch $step")
    Flux.train!(loss, params(m), dataset, opt)
    error = loss(d.X_train, d.Y_train).data
    @show(error)
    @show(accuracy(d.X_test, d.Y_test))
    if plotloss
      update_plot(plt, step, error)
    end
  end

  @info("Done")

  return m
end

# 250 epochs, image size 150x100, eta 0.001, model = 15000 relu | 750 sigmoid | 10 softmax => 77.6% accuracy
