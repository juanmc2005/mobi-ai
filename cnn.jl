using Flux, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy
using Printf, BSON

include("model-commons.jl")
include("myplots.jl")

batch_size = 100
img_n = 48
img_m = 48

function make_batch_indices(i, n, batch_size=batch_size)
  istart = (i - 1) * batch_size + 1
  iend = if (istart+batch_size-1 > n) n else istart+batch_size-1 end
  convert(Int,istart):convert(Int,iend)
end

function train_minibatches(X, Y, batch_size=batch_size, img_n=img_n, img_m=img_m)
  n = size(X,2)
  bcount = ceil(n/batch_size)
  batches = []
  for i in 1:bcount
    inds = make_batch_indices(i, n)
    push!(batches, (reshape(X[:,inds], (img_n, img_m, 1, length(inds))), Y[:,inds]))
  end
  batches
end

function test_minibatch(X, Y, n=img_n, m=img_m)
  bsize = size(X,2)
  inds = make_batch_indices(1, bsize, bsize)
  (reshape(X[:,inds], (n, m, 1, bsize)), Y[:,inds])
end

dataset = create_data_splits()
X_train = reshape(dataset.X_train, (img_n, img_m, 1, size(dataset.X_train,2)))
X_test = reshape(dataset.X_test, (img_n, img_m, 1, size(dataset.X_test,2)))
train_set = train_minibatches(dataset.X_train, dataset.Y_train)
test_set = test_minibatch(dataset.X_test, dataset.Y_test)

@info("Constructing model...")
model = Chain(
    # First convolution, operating upon a 48x48 images
    Conv((8, 8), 1=>16, pad=(1,1), stride=2, relu),

    # Second convolution, operating upon a 22x22 image
    Conv((8, 8), 16=>32, pad=(1,1), stride=2, relu),

    # Reshape 3d tensor into a 2d one, at this point it should be (9, 9, 32, N)
    # which is where we get the 2592 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(2592, 10)
) |> gpu

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

loss(x, y) = logitcrossentropy(model(x), y)

function accuracy(x, y)
  logits = model(x)
  probs = [logsoftmax(logits[:,i]) for i in 1:size(logits,2)]
  mean(onecold.(probs) .== onecold(y))
end

opt = ADAM(0.0005)

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate metrics:
    acc = accuracy(test_set...)
    err = loss(X_train, dataset.Y_train).data
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    @info(@sprintf("[%d]: Current loss: %.4f", epoch_idx, err))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to sound_cnn2.bson")
        BSON.@save "sound_cnn2.bson" model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 2 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 6
        @warn(" -> We're calling this converged.")
        break
    end
end

# Best Accuracy: 74% - File sound_cnn.bson
#   batch_size       = 100
#   spectrogram size = 48x48
#   eta              = 0.0005
#   model:
#       Conv((8, 8), 1=>16, pad=(1,1), stride=2, relu)
#       Conv((8, 8), 16=>32, pad=(1,1), stride=2, relu),
#       Dense(2592, 10)
#       softmax

# Problems : loss is NaN, tried changing crossentropy to logitcrossentropy
#            for numerical stability, and decreasing ETA, but the
#            error keeps appearing. Maybe Flux bug ?
