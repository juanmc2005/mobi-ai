using WAV, DataFrames, CSV, PlotlyJS, DSP, PyPlot
include("spectrograms.jl")

function plot_classes(classes, frequencies)
    plot(bar(;x=classes, y=frequencies))
end

function create_audio_metadata(labels)
    audiodata = DataFrame(sampleRate=Float32[], channels=Int8[], samples=Int32[], length=Float32[], class=String[])
    looked = 0
    n = length(labels)
    for (id, lbl) in labels
        println("Analyzing file $id.wav")
        data, srate = wavread("train/Train/$id.wav")
        channels = size(data,2)
        samples = size(data,1)
        len = samples / srate
        row = [srate channels samples len lbl]
        push!(audiodata, row)
        looked += 1
        println("$(convert(Int8, round(looked / n * 100)))% Analyzed")
    end
    return audiodata
end

# Read class data
df = CSV.read("train/train.csv", header=true)
labels = gen_label_dict("train/train.csv")

# Plot class distribution
classdata = by(df, :Class, nrow)
# plot_classes(classdata[:,1], classdata[:,2])

# audio_metadata = create_audio_metadata(labels)
audio_metadata = CSV.read("metadata.csv", header=true)
# describe(audio_metadata, stats=:all)
# CSV.write("metadata.csv", audio_metadata)

# Plot a single wav file
# data, fs = wavread("train/0.wav")
# trace1 = scatter(;x=1:size(y,1), y=y[:,1], mode="lines")
# trace2 = scatter(;x=1:size(y,1), y=y[:,2], mode="lines")
# plot([trace1, trace2])
spectr = create_spectrogram("train/Train/0.wav")
imshow(spectr)
# imshow(log10.(power(spec)), extent=[first(t), last(t),
#              fs*first(f), fs*last(f)], aspect="auto")
