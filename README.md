# Fiap PosTech - Tech Challenge Fase 4

### Aplicação que utiliza técnicas de análise em vídeo para reconhecimento facial, análise de expressões emocionais e detecção de atividades.

## Bibliotecas utilizadas
    opencv-python
    numpy
    tqdm 
    mediapipe
    face_recognition
    deepface
    tensorflow

## Arquivos do repositório

### Após a instalação das bibliotecas, execute os módulos na sequência descrita abaixo:

### `01_extracao_faces.py`
#### Percorre o vídeo `Unlocking Facial Recognition_ Diverse Activities Analysis.mp4` na diretório raiz analisando, recortando e salvando no diretório `faces_extraidas` as faces detectadas a cada 10 frames.

### `02_reconhecimento_faces.py`
#### Carrega as imagens extraídas pelo arquivo anterior e através da biblioteca face_recognition identifica e compara com as faces encontradas no vídeo origial e gera o vídeo `video_faces.mp4`

### `03_detectar_emocoes_atividades_resumo.py`
#### Utilizando as bibliotecas mediapip e deepface esse módulo realiza detecção de atividades em no vídeo definido na variável `video_entrada_path` gerando um vídeo de saída chamado `video_atividades.mp4` que será utilizado para a análise de emoções com o DeepFace. 

### Os dois arquivos finais serão gerados, o `video_emocoes.mp4` contendo as análises executads e o arquivo `resumo_final.txt`, onde estão os detalhes como totalizadores, categorizadores, contadores de faces, atividades, movimentos anomalos e emoções encontradas durante o vídeo.