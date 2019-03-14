using WAV, DataFrames, CSV, PlotlyJS, DSP, PyPlot

function gen_label_dict(df_labels)
    labels_dict = Dict()
    for i in 1:size(df_labels, 1)
        labels_dict[df_labels[i,1]] = df_labels[i,2]
    end
    return labels_dict
end

function plot_classes(classes, frequencies)
    plot(bar(;x=classes, y=frequencies))
end

function create_audio_metadata(labels)
    audiodata = DataFrame(sampleRate=Float32[], channels=Int8[], samples=Int32[], length=Float32[], class=String[])
    looked = 0
    n = length(labels)
    for (id, lbl) in labels
        println("Analyzing file $id.wav")
        data, srate = wavread("train/$id.wav")
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

function create_spectrogram(wavfile)
    signal, sampleRate = wavread(wavfile)
    spec = spectrogram(signal[:,1], convert(Int, round(25e-3*fs)),
                        convert(Int, 10e-3*fs); window=hanning)
    t = time(spec)
    f = freq(spec)
    return log10.(power(spec))
end

# Read class data
df = CSV.read("train.csv", header=true)
# Read label dictionary : (sound id) => class
labels = gen_label_dict(df)
delete!(labels, 8542) # Bad file

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
spectr = create_spectrogram("train/0.wav")
imshow(spectr)
# imshow(log10.(power(spec)), extent=[first(t), last(t),
#              fs*first(f), fs*last(f)], aspect="auto")
