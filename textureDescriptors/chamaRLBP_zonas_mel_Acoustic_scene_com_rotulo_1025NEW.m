% ======================================================
% SCRIPT PARA COMPUTAR LBP DAS IMAGENS DE CARTAS
% ALGORITMO DE http://www.cse.oulu.fi/MVG/Downloads/LBPMatlab
% UNIVERSIDADE OULU - MAENPPA.
% VERS�O 0.3.2
% ======================================================

function chamaRLBP_zonas_mel_Acoustic_scene_com_rotulo_1025NEW(folder, segmentos, zonas, str_fold, classi, desc)
    clc;
    %folder = uigetdir; 
    %dirListing = dir(folder);
    lin  = 0;
    %segmentos=1;
    %zonas=15;
    %str_fold='acoustic_scene_Mel';
    cont =1;

    for zon=0:(zonas-1)
        for seg=0:(segmentos-1)
            %nome = strcat(str_fold,'-',num2str(zon),'-',num2str(seg),'-sem_rot.txt') ;
            nome_rot = strcat(str_fold,'-', classi, '-',num2str(zon),'-',num2str(seg),'.txt') ;
            fid = fopen(nome_rot,'w');  % NOME DO ARQUIVO DE SA�DA COM R�TULO
            %fid_sem_rot = fopen(nome,'w');  % NOME DO ARQUIVO DE SA�DA SEM R�TULO

            %for d = 3:length(dirListing)

            %if (dirListing(d).isdir == 1)

                fileName = fullfile(folder); % Abre a pasta....

                fopen(fileName);

               arquivos = dir(folder); % carrega as imagens....

                    for i = 3 : length(arquivos) %come�a em 3 pois os dois primeiros s�o: . e ..
                        disp(cont);
                        cont = cont +1;
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

                                mapping=getmapping(8,'u2');

                                %linha1=round((zon/zonas*altura));
                                %if linha1==0
                                %     linha1=1;
                                %end;
                                %linha2=round((zon+1)/zonas*altura);

                                coluna1=round(((seg/segmentos)*largura));
                                if coluna1==0
                                      coluna1=1;                            
                                end;
                                coluna2=round(((seg+1)/segmentos)*largura);

                                switch (zon)
                                    case 0
                                       linha1= 1;
                                       linha2= 265;
                                    case 1
                                       linha1= 265;
                                       linha2= 305;
                                    case 2
                                       linha1= 305;
                                       linha2= 336;
                                    case 3
                                       linha1= 336;
                                       linha2= 381;
                                    case 4
                                       linha1= 381;
                                       linha2= 404;
                                    case 5
                                       linha1= 404;
                                       linha2= 417;
                                    case 6
                                       linha1= 417;
                                       linha2= 448;                      
                                    case 7
                                       linha1= 448;
                                       linha2= 481;
                                    case 8
                                       linha1= 481;
                                       linha2= 485;
                                    case 9
                                       linha1= 485;
                                       linha2= 491; 
                                    case 10
                                       linha1= 491;
                                       linha2= 500;
                                    case 11
                                       linha1= 500;
                                       linha2= 507;
                                    case 12
                                       linha1= 506;
                                       linha2= 510;
                                    case 13
                                       linha1= 507;
                                       linha2= 512;
                                    case 14
                                       linha1= 509;
                                       linha2= 513;
                                end

                                switch(desc)
                                    case 1
                                        H1=RLBP(I(linha1:linha2,coluna1:coluna2),2,8,mapping,'nh'); % nh = histograma normalizado.... 8 vizinhos
                                    case 2
                                        H1 = lpq(I(linha1:linha2,coluna1:coluna2),3);
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