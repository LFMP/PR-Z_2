%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Autor: Gustavo Zanoni Felipe
%Data: 23/04/2017
%Este algoritmo aplica os descritores: RLBP, LBP e LPQ.
%Podendo zonear e segmentar.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETROS

entrada = '/home/gustavozf/Documentos/UEM/Projetos/Baby/base_americana/03_Folds/x1500/audios_04/'; %Diretorio dos Espectrogramas
out = '/home/gustavozf/Documentos/UEM/Projetos/Baby/base_americana/04_Features/x1500/audios_04/'; %Saida com os features
frequencia = '32000';
amplitude = '130';
desc = 3; %1 = RLBP, 2 = LPQ, 3 = LBP
tipo_zona = 1; %1 = Zonas Lineares / 2 = Escala Mel
segmentos = 1; %Numero de segmentos
zonas = 1; %Numero de zonas. PS.: Escala Mel = 15 Zonas!
folds = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
entrada = strcat(entrada,num2str(folds),'_folds/', frequencia,'Hz_',amplitude,'dB/');
out = strcat(out,num2str(folds),'_folds/' , frequencia,'Hz_',amplitude,'dB/');

switch (desc)
    case 1
        descritor = 'rlbp';
    case 2
        descritor = 'lpq';
    case 3
        descritor = 'lbp';
end
switch (tipo_zona)
    case 1 
        str_fold =  'zl';
    case 2
        str_fold =  'mel';
end

%examplo de saida: saida/lbp_mel_15z_1s
destino = strcat(out,descritor, '_', str_fold, '_', num2str(zonas), 'z_', num2str(segmentos), 's/');

[status, msg] = mkdir(destino);
if status %faz o diretorio da saida
    disp(['Destino Criado: ', destino]);
else
    disp(['Destino nao Criado: ', msg]);
end

%can = {'Mono', 'Spectrograms1', 'Spectrograms2'};
%Caso haja canais (Canal esquerdo, direito ou esquerdo+direito (mono))
can = {'Left_Channel', 'Right_Channel'};

%Variar teste e treino
tes = {'test', 'train'};
%tes = {};

%vetor com as classificacoes
%classi = {'beach', 'bus', 'cafe-restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram'};
%classi = {'Friction', 'Move', 'Pain', 'Rest'};
classi = {'Cry_No_Pain_5s', 'Cry_Pain_5s'};

for i = 1 : length(can) %varia entre os canais
    for j = 1 : length(tes) %varia teste e treino
        for z = 1 : folds %varia pelo numero de folds
            for k = 1:length(classi) %varia as classificacoes
            fold = strcat(entrada, can{i}, '/', tes{j}, '/fold', num2str(z), '/', classi{k}, '/');
            saida = strcat(destino, can{i},'/', tes{j}, '/fold', num2str(z), '/', classi{k}, '/');
            %fold = strcat(entrada, can{i}, '/fold', num2str(z), '/', classi{k}, '/');
            %saida = strcat(destino, can{i},'/fold', num2str(z), '/', classi{k}, '/');
            
            [status, msg] = mkdir(saida);
            if status %faz o diretorio da saida
                disp(['Diretorio Criado: ', saida]);
            else
                disp(['Diretorio nao Criado: ', msg]);
            end
                
            switch (tipo_zona)
                case 1 
                    chamaRLBP_zonas_linearNEW(fold, segmentos, zonas, str_fold, classi{k}, desc);
                case 2
                    chamaRLBP_zonas_mel_Acoustic_scene_com_rotulo_1025NEW(fold, segmentos, zonas, str_fold, classi{k}, desc);
            end
            movefile('*.txt', saida);
            disp(saida);
            end
        end
    end
end