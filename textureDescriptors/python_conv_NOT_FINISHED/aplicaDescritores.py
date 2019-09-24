# Generated with SMOP  0.41-beta
from libsmop import *
# 

    #######################################################################
#Autor: Gustavo Zanoni Felipe
#Data: 23/04/2017
#Este algoritmo aplica os descritores: RLBP, LBP e LPQ.
#Podendo zonear e segmentar.
#######################################################################
# PARAMETROS
    
    entrada='/home/gustavozf/Documentos/UEM/Projetos/Baby/base_americana/03_Folds/x1500/audios_04/'
# aplicaDescritores.m:9
    
    out='/home/gustavozf/Documentos/UEM/Projetos/Baby/base_americana/04_Features/x1500/audios_04/'
# aplicaDescritores.m:10
    
    frequencia='32000'
# aplicaDescritores.m:11
    amplitude='130'
# aplicaDescritores.m:12
    desc=3
# aplicaDescritores.m:13
    
    tipo_zona=1
# aplicaDescritores.m:14
    
    segmentos=1
# aplicaDescritores.m:15
    
    zonas=1
# aplicaDescritores.m:16
    
    folds=5
# aplicaDescritores.m:17
    #######################################################################
    entrada=strcat(entrada,num2str(folds),'_folds/',frequencia,'Hz_',amplitude,'dB/')
# aplicaDescritores.m:19
    out=strcat(out,num2str(folds),'_folds/',frequencia,'Hz_',amplitude,'dB/')
# aplicaDescritores.m:20
    if 1 == (desc):
        descritor='rlbp'
# aplicaDescritores.m:24
    else:
        if 2 == (desc):
            descritor='lpq'
# aplicaDescritores.m:26
        else:
            if 3 == (desc):
                descritor='lbp'
# aplicaDescritores.m:28
    
    if 1 == (tipo_zona):
        str_fold='zl'
# aplicaDescritores.m:32
    else:
        if 2 == (tipo_zona):
            str_fold='mel'
# aplicaDescritores.m:34
    
    #examplo de saida: saida/lbp_mel_15z_1s
    destino=strcat(out,descritor,'_',str_fold,'_',num2str(zonas),'z_',num2str(segmentos),'s/')
# aplicaDescritores.m:38
    status,msg=mkdir(destino,nargout=2)
# aplicaDescritores.m:40
    if status:
        disp(concat(['Destino Criado: ',destino]))
    else:
        disp(concat(['Destino nao Criado: ',msg]))
    
    #can = {'Mono', 'Spectrograms1', 'Spectrograms2'};
#Caso haja canais (Canal esquerdo, direito ou esquerdo+direito (mono))
    can=cellarray(['Left_Channel','Right_Channel'])
# aplicaDescritores.m:49
    #Variar teste e treino
    tes=cellarray(['test','train'])
# aplicaDescritores.m:52
    #tes = {};
    
    #vetor com as classificacoes
#classi = {'beach', 'bus', 'cafe-restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram'};
#classi = {'Friction', 'Move', 'Pain', 'Rest'};
    classi=cellarray(['Cry_No_Pain_5s','Cry_Pain_5s'])
# aplicaDescritores.m:58
    for i in arange(1,length(can)).reshape(-1):
        for j in arange(1,length(tes)).reshape(-1):
            for z in arange(1,folds).reshape(-1):
                for k in arange(1,length(classi)).reshape(-1):
                    fold=strcat(entrada,can[i],'/',tes[j],'/fold',num2str(z),'/',classi[k],'/')
# aplicaDescritores.m:64
                    saida=strcat(destino,can[i],'/',tes[j],'/fold',num2str(z),'/',classi[k],'/')
# aplicaDescritores.m:65
                    #saida = strcat(destino, can{i},'/fold', num2str(z), '/', classi{k}, '/');
                    status,msg=mkdir(saida,nargout=2)
# aplicaDescritores.m:69
                    if status:
                        disp(concat(['Diretorio Criado: ',saida]))
                    else:
                        disp(concat(['Diretorio nao Criado: ',msg]))
                    if 1 == (tipo_zona):
                        chamaRLBP_zonas_linearNEW(fold,segmentos,zonas,str_fold,classi[k],desc)
                    else:
                        if 2 == (tipo_zona):
                            chamaRLBP_zonas_mel_Acoustic_scene_com_rotulo_1025NEW(fold,segmentos,zonas,str_fold,classi[k],desc)
                    movefile('*.txt',saida)
                    disp(saida)
    