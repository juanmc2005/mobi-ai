using Flux, Statistics
using Flux: onecold, crossentropy

include("myplots.jl")
include("model-commons.jl")

epochs = 300
eta = 0.001

# Train the ANN model
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
