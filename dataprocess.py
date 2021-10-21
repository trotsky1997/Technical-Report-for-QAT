import pandas
from pandas.core.indexes.base import Index
data = pandas.read_csv("./acc.txt",sep="\t",header=None)
datas,models,bits = [],[],[]
for i in data.iloc[:,0]:
    # print(i)
    a,b,c = i.split("-")
    datas.append(a)
    models.append(b)
    bits.append(c)
data.columns = ["a","item","value"]
data["dataset"] = datas
data["models"] = models
data["bits"] = bits

data[list(data.columns[3:])+['value']].to_csv("./acc.csv",index=False)
a = data[(data["item"] == "pre_quant")].copy()

c = data[(data["item"] == "finetuned")]
a["trained_acc"] = a['value'].values
a['fintuned_acc'] = c['value'].values

print(a)

a["finetuned and quantization loss"] = - a["trained_acc"] + a['fintuned_acc']
print(a[a.columns[3:]])

