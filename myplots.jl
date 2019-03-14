using Plots

function create_plot(x,y)
	plt = plot([x],[y],title="Error function",label=["Error"])
	sleep(0.005)
	return plt
end

function update_plot(plt,x,y)
    push!(plt, 1, x, 1.0*y)
    display(plt)
    sleep(0.005)
end
