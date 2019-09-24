# Generated with SMOP  0.41-beta
from libsmop import *
# 

    # ======================================================
# SCRIPT PARA COMPUTAR LBP DAS IMAGENS DE CARTAS
# ALGORITMO DE http://www.cse.oulu.fi/MVG/Downloads/LBPMatlab
# UNIVERSIDADE OULU - MAENPPA.
# VERS�O 0.3.2
# ======================================================
    
    
@function
def chamaRLBP_zonas_linearNEW(folder=None,segmentos=None,zonas=None,str_fold=None,classi=None,desc=None,*args,**kwargs):
    varargin = chamaRLBP_zonas_linearNEW.varargin
    nargin = chamaRLBP_zonas_linearNEW.nargin

    #clc;
    #folder = 'E:\Espectrogramas\44.1k-z=150(stereo)\Mono\test\fold1'; 
    #dirListing = dir(folder);
    lin=0
# chamaRLBP_zonas_linearNEW.m:12
    
    #zonas=5;
    #str_fold='acoustic_scene';
    cont=1
# chamaRLBP_zonas_linearNEW.m:16
    for zon in arange(0,(zonas - 1)).reshape(-1):
        for seg in arange(0,(segmentos - 1)).reshape(-1):
            #nome = strcat(str_fold,'-',num2str(zon),'-',num2str(seg),'.txt') ;
            nome_rot=strcat(str_fold,'-',classi,'-',num2str(zon),'-',num2str(seg),'.txt')
# chamaRLBP_zonas_linearNEW.m:21
            fid=fopen(nome_rot,'w')
# chamaRLBP_zonas_linearNEW.m:22
            #fid_sem_rot = fopen(nome,'w');  # NOME DO ARQUIVO DE SA�DA COM R�TULO
            #for d = 3:length(dirListing)
            #if (folder.isdir == 1)
            fileName=fullfile(folder)
# chamaRLBP_zonas_linearNEW.m:29
            fopen(folder)
            arquivos=dir(folder)
# chamaRLBP_zonas_linearNEW.m:33
            for i in arange(3,length(arquivos)).reshape(-1):
                disp(cont)
                cont=cont + 1
# chamaRLBP_zonas_linearNEW.m:37
                if (arquivos(i).isdir == 0):
                    nomeArquivo=fullfile(fileName,arquivos(i).name)
# chamaRLBP_zonas_linearNEW.m:40
                    #       'u2'   for uniform LBP
                                #       'ri'   for rotation-invariant LBP
                                #       'riu2' for uniform rotation-invariant LBP.
                    fopen(fileName)
                    lin=lin + 1
# chamaRLBP_zonas_linearNEW.m:46
                    I=imread(nomeArquivo)
# chamaRLBP_zonas_linearNEW.m:48
                    altura,largura=size(I,nargout=2)
# chamaRLBP_zonas_linearNEW.m:49
                    #fator_vertical=round(altura/zonas);
                    mapping=getmapping(8,'u2')
# chamaRLBP_zonas_linearNEW.m:52
                    linha1=round((dot(zon / zonas,altura)))
# chamaRLBP_zonas_linearNEW.m:54
                    if linha1 == 0:
                        linha1=1
# chamaRLBP_zonas_linearNEW.m:56
                    linha2=round(dot((zon + 1) / zonas,altura))
# chamaRLBP_zonas_linearNEW.m:58
                    coluna1=round((dot((seg / segmentos),largura)))
# chamaRLBP_zonas_linearNEW.m:59
                    if coluna1 == 0:
                        coluna1=1
# chamaRLBP_zonas_linearNEW.m:61
                    coluna2=round(dot(((seg + 1) / segmentos),largura))
# chamaRLBP_zonas_linearNEW.m:63
                    #coluna1=(seg*fator_horizontal)+1;
                                #linha2=((zon*fator_vertical)+fator_vertical);
                                #coluna2=((seg*fator_horizontal)+fator_horizontal);
                    if 1 == (desc):
                        H1=RLBP(I(arange(linha1,linha2),arange(coluna1,coluna2)),2,8,mapping,'nh')
# chamaRLBP_zonas_linearNEW.m:71
                    else:
                        if 2 == (desc):
                            H1=lpq(I(arange(linha1,linha2),arange(coluna1,coluna2)),7)
# chamaRLBP_zonas_linearNEW.m:73
                        else:
                            if 3 == (desc):
                                H1=lbp(I(arange(linha1,linha2),arange(coluna1,coluna2)),2,8,mapping,'nh')
# chamaRLBP_zonas_linearNEW.m:75
                    #Salvando em arquivo....
                    fprintf(fid,'%f ',H1)
                    fprintf(fid,'%s ',arquivos(i).name)
                    fprintf(fid,'\n')
                    arquivoSaida[lin,arange()]=H1
# chamaRLBP_zonas_linearNEW.m:88
                    clear('I','mapping','H1')
            #end 
            #end
            #save (nome, arquivoSaida, '-ascii'); # NOME DO ARQUIVO DE SA�DA SEM OS R�TULOS.
            fclose(fid)
    
    return
    
if __name__ == '__main__':
    pass
    