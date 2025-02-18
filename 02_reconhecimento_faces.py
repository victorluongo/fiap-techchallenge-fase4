import cv2 
import os
import numpy as np
import face_recognition
from tqdm import tqdm

def escrever_texto(frame, text, x, y):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (242, 48, 100)
    bg_color = (255, 255, 255)
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def preprocessar_imagem(image, scale=3, alpha=1.5, beta=50):

    # ajusta brilho e contraste
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # return adjusted_image
    
    # redimensiona a imagem
    resized_image = cv2.resize(adjusted_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized_image

def carregar_imagens_da_pasta(folder):

    # listas para armazenar as codificações e nomes
    known_face_names = []
    known_face_encodings = []
    
    # contador de imagens sem rostos
    images_without_faces = 0  

    # verifica se a pasta existe
    if not os.path.exists(folder):
        print(f"Pasta {folder} não encontrada.")
        return known_face_encodings, known_face_names

    # verifica todos os arquivos na pasta
    # for filename in os.listdir(folder):
    for filename in tqdm(os.listdir(folder), desc='Carregando imagens'):        
        
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder, filename)
        
            # carrega a imagem (a função retorna a imagem em RGB)
            image = face_recognition.load_image_file(image_path)

            # aplica o pré-processamento
            image = preprocessar_imagem(image, scale=3, alpha=1.5, beta=50)

            # Detecta faces (usando o modelo "cnn" para maior precisão)
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) == 0:
                # print(f"Sem rostos detectados em: {filename}")
                images_without_faces += 1
                continue

            # print(f"Processando imagem: {filename}")
            # Para cada face detectada, armazena a codificação e utiliza o nome do arquivo (sem extensão)
            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0][:-2].replace("_", " "))

    # print(f"Imagens processadas: {len(known_face_names)}")
    # print(f"Imagens sem rostos detectados: {images_without_faces}")
    return known_face_encodings, known_face_names

def main():
    
    # carrega as imagens de faces previamente extraídas e obtém as codificações
    image_folder = 'faces_extraidas'
    known_face_encodings, known_face_names = carregar_imagens_da_pasta(image_folder)
    # print("Nomes conhecidos:", known_face_names)

    # abrir o vídeo de entrada ou camera
    video_path = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"  # ou outro caminho de vídeo
    
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Erro ao abrir o vídeo:", video_path)
        return

    # caminho para salvar o vídeo processado
    video_faces_path = "video_faces.mp4"

    # Obter propriedades do vídeo
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Criar um objeto VideoWriter para salvar o vídeo com reconhecimento
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(video_faces_path, fourcc, fps, (frame_width, frame_height))

    for _ in tqdm(range(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))), desc='Identificando faces'):
        
        # Captura um frame do vídeo
        ret, frame = video_capture.read()
        if not ret:
            break 

        # Redimensiona para 1/2 do tamanho para acelerar a detecção
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Converte de BGR (padrão OpenCV) para RGB (padrão do face_recognition)
        # rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detecta as localizações das faces no frame pequeno usando o modelo "cnn"
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1, model="cnn")
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame) 
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="small")

        face_names = []  # Lista para armazenar os nomes identificados
        for face_encoding in face_encodings:
            
            # Compara com as codificações conhecidas; pode-se ajustar o parâmetro "tolerance" se necessário
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Desconhecido"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)

        # Para cada face detectada, desenha um retângulo e exibe o nome
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            # Reajusta as coordenadas para o tamanho original (já que o frame foi reduzido)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            top += -25
            right += 25
            bottom += 25
            left += -25
            
            cv2.rectangle(frame, (left, top), (right, bottom), (242, 48, 100), 1)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (242, 48, 100), cv2.FILLED)
            
            cv2.putText(frame, f"{name} ", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # escreve o frame processado no vídeo de saída
        out.write(frame)

        # (opcional) exibe o frame em tempo real – descomente se desejar visualizar
        # cv2.imshow('Reconhecimento de Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # libera os recursos e finaliza
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Vídeo com reconhecimento salvo em: {video_faces_path}")

if __name__ == "__main__":
    main()
