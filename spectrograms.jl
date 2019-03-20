using DataFrames, CSV, WAV, DSP, Images

class_to_i = Dict(
    "siren" => 0,
    "street_music" => 1,
    "drilling" => 2,
    "dog_bark" => 3,
    "children_playing" => 4,
    "gun_shot" => 5,
    "engine_idling" => 6,
    "air_conditioner" => 7,
    "jackhammer" => 8,
    "car_horn" => 9
)

# Read label dictionary : (sound id) => class
function gen_label_dict(labels_filename)
    # Read class data
    df = CSV.read(labels_filename, header=true)
    labels_dict = Dict()
    for i in 1:size(df, 1)
        labels_dict[df[i,1]] = df[i,2]
    end
    delete!(labels_dict, 8542) # Bad file
    delete!(labels_dict, 3190) # Spectrogram has 0 columns (weird)
    return labels_dict
end

function simple_labels(labels_dict)
    result = Array{Int}(undef, 0)
    for id in sort(collect(keys(labels_dict)))
        push!(result, class_to_i[labels_dict[id]])
    end
    return result
end

function create_spectrogram(wavfile)
    signal, sampleRate = wavread(wavfile)
    spec = spectrogram(signal[:,1], convert(Int, round(25e-3*sampleRate)),
                        convert(Int, round(10e-3*sampleRate)); window=hanning)
    t = time(spec)
    f = freq(spec)
    return log10.(power(spec))
end

function create_spectrogram_base(labels, n=200, m=100)
    spectres = Array{Any}(undef, 0)
    iter = 0
    for id in sort(collect(keys(labels)))
        spectr = create_spectrogram("train/Train/$id.wav")
        if size(spectr, 2) == 0
            @info("$id.wav spectrogram has 0 columns. Skipping...")
        else
            push!(spectres, imresize(sigmoid.(spectr), (n, m)))
        end
        iter += 1
        if iter % 300 == 0
            @info("Created $iter spectrograms")
        end
    end
    return spectres
end
