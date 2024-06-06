import Plots as plt
using PGFPlots
using XLSX
using DataFrames
# gr()
input_file = "correlation approx-comp data after edit .xlsx"

data1 = XLSX.readtable(input_file, "PT_COR1") |> DataFrame
describe(data1)

vscodedisplay(data1)

raw_data1 = transform(data1, [1, 3, 4, 5] => ByRow((x1, x2, x3, x4) -> (Int.(x1), x2, x3, x4)) => [:sz, :Pro, :SQV, :SQC])
labels = permutedims(names(raw_data1[!, 3:5]))
xtcks = (1:8, raw_data1[1:8, 2])
# vscodedisplay(raw_data1)

function getDataBySize(df, sz)
    filter(r -> r[:sz] == sz, df)
end
function getPlotingData(df)
    df[!, 3:5] |> Matrix
end
function getPlot(df, sz)
    data1 = getDataBySize(df, sz) |> getPlotingData
    # p = plt.scatter(data1, label=:none)
    # plt.plot(p, data1, labels=labels, legend=:outertop, legendfont=(6), xticks=xtcks, legend_column=-1, title=sz, xtickfont=(6), titlefont=("Arial", 8, :bold), xrotation=45)
    plt.plot(data1, label=labels, markershape=:auto, xticks=xtcks, legend=:outertop, legend_column=-1, title="Size:$(sz)", xtickfont=(6), titlefont=("Arial", 6, :bold), xrotation=45)

end

plts1 = map([10, 15, 25, 50, 100, 200]) do sz
    getPlot(raw_data1, sz)
end

plt.pgfplotsx()

pt = plt.plot(plts1..., layout=(2, 3), size=(800, 600))

# plt.savefig(pt, "PCT.png")
plt.savefig(pt, "PCT.tex")




data2 = XLSX.readtable(input_file, "PI_COR1") |> DataFrame
describe(data2)

# vscodedisplay(data2)

raw_data2 = transform(data2, [1, 3, 4, 5] => ByRow((x1, x2, x3, x4) -> (Int.(x1), x2, x3, x4)) => [:sz, :Pro, :SQV, :SQC])
labels = permutedims(names(raw_data2[!, 3:5]))
xtcks = (1:8, raw_data2[1:8, 2])
# vscodedisplay(raw_data2)

function getDataBySize(df, sz)
    filter(r -> r[:sz] == sz, df)
end

function getPlotingData(df)
    df[!, 3:5] |> Matrix
end

function getPlot(df, sz)
    data2 = getDataBySize(df, sz) |> getPlotingData
    # p = plt.scatter(data2, label=:none)
    # plt.plot(p, data2, labels=labels, legend=:outertop, legendfont=(6), xticks=xtcks, legend_column=-1, title=sz, xtickfont=(6), titlefont=("Arial", 8, :bold), xrotation=45)
    plt.plot(data2, label=labels, markershape=:auto, xticks=xtcks, legend=:outertop, legend_column=-1, title="Size:$(sz)", xtickfont=(6), titlefont=("Arial", 6, :bold), xrotation=45)

end

plts2 = map([10, 15, 25, 50, 100, 200]) do sz
    getPlot(raw_data2, sz)
end

plt.pgfplotsx()

Pi = plt.plot(plts2..., layout=(2, 3), size=(800, 600))

plt.savefig(Pi, "PCI.tex")
# plt.savefig(Pi, "PCI.png")

data3 = XLSX.readtable(input_file, "PE_COR1") |> DataFrame
describe(data3)

# vscodedisplay(data3)

raw_data3 = transform(data3, [1, 3, 4, 5] => ByRow((x1, x2, x3, x4) -> (Int.(x1), x2, x3, x4)) => [:sz, :Pro, :SQV, :SQC])
labels = permutedims(names(raw_data3[!, 3:end]))
xtcks = (1:8, raw_data3[1:8, 2])
# vscodedisplay(raw_data3)

function getDataBySize(df, sz)
    filter(r -> r[:sz] == sz, df)
end

function getPlotingData(df)
    df[!, 3:5] |> Matrix
end

function getPlot(df, sz)
    data3 = getDataBySize(df, sz) |> getPlotingData
    # p = plt.scatter(data3, label=:none)
    # plt.plot(p, data3, labels=labels, legend=:outertop, legendfont=(6), xticks=xtcks, legend_column=-1, title=sz, xtickfont=(6), titlefont=("Arial", 8, :bold), xrotation=45)
    plt.plot(data3, label=labels, markershape=:auto, xticks=xtcks, legend=:outertop, legend_column=-1, title="Size:$(sz)", xtickfont=(6), titlefont=("Arial", 6, :bold), xrotation=45)

end

plts3 = map([10, 15, 25, 50, 100, 200]) do sz
    getPlot(raw_data3, sz)
end

plt.pgfplotsx()

pe = plt.plot(plts3..., layout=(2, 3), size=(800, 600))

plt.savefig(pe, "PCE..tex")

# plt.savefig(pe, "PCE.png")


data4 = XLSX.readtable(input_file, "CT_COR") |> DataFrame
describe(data4)

# vscodedisplay(data1)

raw_data4 = transform(data4, [1, 3, 4, 5] => ByRow((x1, x2, x3, x4) -> (Int.(x1), x2, x3, x4)) => [:sz, :Pro, :SQV, :SQC])
labels = permutedims(names(raw_data4[!, 3:5]))
xtcks = (1:8, raw_data4[1:8, 2])
# vscodedisplay(raw_data1)

function getDataBySize(df, sz)
    filter(r -> r[:sz] == sz, df)
end
function getPlotingData(df)
    df[!, 3:5] |> Matrix
end
function getPlot(df, sz)
    data4 = getDataBySize(df, sz) |> getPlotingData
    # p = plt.scatter(data4, label=:none)
    # plt.plot(p, data4, labels=labels, legend=:outertop, legendfont=(6), xticks=xtcks, legend_column=-1, title=sz, xtickfont=(6), titlefont=("Arial", 8, :bold), xrotation=45)
    plt.plot(data4, label=labels, markershape=:auto, xticks=xtcks, legend=:outertop, legend_column=-1, title="Size:$(sz)", xtickfont=(6), titlefont=("Arial", 6, :bold), xrotation=45)

end

plts4 = map([10, 15, 25, 50, 100, 200]) do sz
    getPlot(raw_data4, sz)
end

plt.pgfplotsx()

Ct = plt.plot(plts4..., layout=(2, 3), size=(800, 600))

# plt.savefig(pt, "PCT.png")
plt.savefig(Ct, "T_COM.tex")




data5 = XLSX.readtable(input_file, "CI_COR") |> DataFrame
describe(data5)

# vscodedisplay(data2)

raw_data5 = transform(data5, [1, 3, 4, 5] => ByRow((x1, x2, x3, x4) -> (Int.(x1), x2, x3, x4)) => [:sz, :Pro, :SQV, :SQC])
labels = permutedims(names(raw_data5[!, 3:5]))
xtcks = (1:8, raw_data5[1:8, 2])
# vscodedisplay(raw_data2)

function getDataBySize(df, sz)
    filter(r -> r[:sz] == sz, df)
end

function getPlotingData(df)
    df[!, 3:5] |> Matrix
end

function getPlot(df, sz)
    data5 = getDataBySize(df, sz) |> getPlotingData
    # p = plt.scatter(data5, label=:none)
    # plt.plot(p, data5, labels=labels, legend=:outertop, legendfont=(6), xticks=xtcks, legend_column=-1, title=sz, xtickfont=(6), titlefont=("Arial", 8, :bold), xrotation=45)
    plt.plot(data5, label=labels, markershape=:auto, xticks=xtcks, legend=:outertop, legend_column=-1, title="Size:$(sz)", xtickfont=(6), titlefont=("Arial", 6, :bold), xrotation=45)

end

plts5 = map([10, 15, 25, 50, 100, 200]) do sz
    getPlot(raw_data5, sz)
end

plt.pgfplotsx()

Ci = plt.plot(plts5..., layout=(2, 3), size=(800, 600))

plt.savefig(Ci, "I_COM.tex")
# plt.savefig(Pi, "PCI.png")

data6 = XLSX.readtable(input_file, "CE_COR") |> DataFrame
describe(data6)

# vscodedisplay(data3)

raw_data6 = transform(data6, [1, 3, 4, 5] => ByRow((x1, x2, x3, x4) -> (Int.(x1), x2, x3, x4)) => [:sz, :Pro, :SQV, :SQC])
labels = permutedims(names(raw_data6[!, 3:end]))
xtcks = (1:8, raw_data6[1:8, 2])
# vscodedisplay(raw_data3)

function getDataBySize(df, sz)
    filter(r -> r[:sz] == sz, df)
end

function getPlotingData(df)
    df[!, 3:5] |> Matrix
end

function getPlot(df, sz)
    data6 = getDataBySize(df, sz) |> getPlotingData
    # p = plt.scatter(data6, label=:none)
    # plt.plot(p, data6, labels=labels, legend=:outertop, legendfont=(6), xticks=xtcks, legend_column=-1, title=sz, xtickfont=(6), titlefont=("Arial", 8, :bold), xrotation=45)
    plt.plot(data6, label=labels, markershape=:auto, xticks=xtcks, legend=:outertop, legend_column=-1, title="Size:$(sz)", xtickfont=(6), titlefont=("Arial", 6, :bold), xrotation=45)

end

plts6 = map([10, 15, 25, 50, 100, 200]) do sz
    getPlot(raw_data6, sz)
end

plt.pgfplotsx()

Ce = plt.plot(plts6..., layout=(2, 3), size=(800, 600))

plt.savefig(Ce, "E_COM..tex")

# plt.savefig(Ce, "PCE.png")