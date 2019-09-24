% ======================================================
% SCRIPT PARA COMPUTAR LBP DAS IMAGENS DE CARTAS
% ALGORITMO DE http://www.cse.oulu.fi/MVG/Downloads/LBPMatlab
% UNIVERSIDADE OULU - MAENPPA.
% VERS�O 0.3.2
% ======================================================

function chamaRLBP_zonas_linearNEW(folder, segmentos, zonas, str_fold, classi, desc)
    %clc;
    %folder = 'E:\Espectrogramas\44.1k-z=150(stereo)\Mono\test\fold1'; 
    %dirListing = dir(folder);
    lin  = 0;
    %segmentos=1;
    %zonas=5;
    %str_fold='acoustic_scene';
    cont = 1;

    for zon=0:(zonas-1)
        for seg=0:(segmentos-1)
            %nome = strcat(str_fold,'-',num2str(zon),'-',num2str(seg),'.txt') ;
            nome_rot = strcat(classi,'.',str_fold, '.','.',num2str(zon),'.',num2str(seg),'.features') ;
            fid = fopen(nome_rot,'w');  % NOME DO ARQUIVO DE SA�DA COM R�TULO
            %fid_sem_rot = fopen(nome,'w');  % NOME DO ARQUIVO DE SA�DA COM R�TULO

            %for d = 3:length(dirListing)

            %if (folder.isdir == 1)

                fileName = fullfile(folder); % Abre a pasta....

                fopen(folder);

                arquivos = dir(folder); % carrega as imagens....

                    for i = 3 : length(arquivos) %come�a em 3 pois os dois primeiros s�o: . e ..
                        disp(cont);
                        cont = cont + 1;
                        if (arquivos(i).isdir == 0)

                                 nomeArquivo = fullfile(fileName,arquivos(i).name); 
                                %  Possible values for MAPPINGTYPE are
                                %       'u2'   for uniform LBP
                                %       'ri'   for rotation-invariant LBP
                                %       'riu2' for uniform rotation-invariant LBP.
                                fopen(fileName);%nomeArquivo)
                                lin = lin + 1;   

                                I = imread(nomeArquivo); % L� imagem.
                                [altura, largura]=size(I);
                                %fator_horizontal=round(largura/segmentos);
                                %fator_vertical=round(altura/zonas);
                                mapping=getmapping(8,'u2');

                                linha1=round((zon/zonas*altura));
                                if linha1==0
                                     linha1=1;
                                end
                                linha2=round((zon+1)/zonas*altura);
                                coluna1=round(((seg/segmentos)*largura));
                                if coluna1==0
                                      coluna1=1;                            
                                end
                                coluna2=round(((seg+1)/segmentos)*largura);
                                %linha1=(zon*fator_vertical)+1;
                                %coluna1=(seg*fator_horizontal)+1;
                                %linha2=((zon*fator_vertical)+fator_vertical);
                                %coluna2=((seg*fator_horizontal)+fator_horizontal);

                                switch(desc)
                                    case 1
                                        H1 = RLBP(I(linha1:linha2,coluna1:coluna2),2,8,mapping,'nh'); % nh = histograma normalizado.... 8 vizinhos
                                    case 2
                                        H1 = lpq(I(linha1:linha2,coluna1:coluna2), 11);
                                    case 3
                                        H1 = lbp(I(linha1:linha2,coluna1:coluna2),2,8,mapping,'nh');
                                end
                                %Salvando em arquivo....

                                fprintf(fid, '%f ', H1 ); %k, VetorDissimilaridade(lin,k));
                                fprintf(fid, '%s ', arquivos(i).name);

                                %fprintf(fid_sem_rot, '%f ', H1 );


                                fprintf(fid,'\n');
                                %fprintf(fid_sem_rot,'\n');

                                arquivoSaida(lin,:) = H1;
                                clear I mapping H1;
                        end

                    end

            %end 
            %end

        %save (nome, arquivoSaida, '-ascii'); % NOME DO ARQUIVO DE SA�DA SEM OS R�TULOS.

        fclose ( fid );
        %fclose ( fid_sem_rot );

        end
    end
end