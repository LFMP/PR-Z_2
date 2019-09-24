import sys, random

def divideK(lista, k):
	retorno = [lista[i::k] for i in range(k)]
	random.shuffle(retorno)
	return retorno

numFolds = int(sys.argv[1])
dic = {
	'03_Artrite': {
		'Unhealthy' : {},
		'Healthy': {}
	},

	'02_Tumor': {
		'Unhealthy' : {},
		'Healthy': {}
	},

	'01_Diabetes': {
		'Unhealthy' : {},
		'Healthy': {}
	}
}

foldsCount = {
	'03_Artrite': {
		'Unhealthy' : [0 for i in range(numFolds)],
		'Healthy': [0 for i in range(numFolds)]
	},

	'02_Tumor': {
		'Unhealthy' : [0 for i in range(numFolds)],
		'Healthy': [0 for i in range(numFolds)]
	},

	'01_Diabetes': {
		'Unhealthy' : [0 for i in range(numFolds)],
		'Healthy': [0 for i in range(numFolds)]
	}
}


for linha in open('./glia_cells.map.txt', 'r'):
	info = linha.split(' ')
	doenca, classe, grupo = info[0].split('/')[2:5]

	if grupo not in dic[doenca][classe].keys():
		dic[doenca][classe][grupo] = {}

	dic[doenca][classe][grupo][linha] = 0
	
#saida = open('teste{}folds.txt'.format(numFolds), 'w')
for doenca in sorted(dic.keys()):
	print(doenca)
	
	for classe in sorted(dic[doenca].keys()):
		print("  %s" % classe)
		for grupo in sorted(dic[doenca][classe].keys()):
			print("    %s : %d" % (grupo, len(dic[doenca][classe][grupo].keys())))
			amostras = divideK(list(dic[doenca][classe][grupo].keys()), numFolds)
			for i in range(len(amostras)):
				for amostra in amostras[i]:
					dic[doenca][classe][grupo][amostra] = i
					#saida.write("%d %s" % (i, amostra))
				foldsCount[doenca][classe][i] += len(amostras[i])
		print("Divisao de folds: ", foldsCount[doenca][classe], sum(foldsCount[doenca][classe]))
#saida.close()
