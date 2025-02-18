import cv2
import os
import face_recognition
from tqdm import tqdm

def extrair_faces_do_video(video_path, output_folder):

    # cria a pasta de saída se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # dicionario imagens por pessoa
    face_data = {}
    
    # lista de rostos únicos
    known_encodings = []    
    
    # contador de frames
    frame_count = 0

    # abre o vídeo
    cap = cv2.VideoCapture(video_path)
    
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc='Extraindo faces'):
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # processa a cada 10 quadros para variação de ângulo
        if frame_count % 10 != 0:  
            continue

        # converter imagem de BGR (OpenCV) para RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # rechonecimento de face no frame
        face_locations = face_recognition.face_locations(img=rgb_frame, number_of_times_to_upsample=1, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="small")

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
            
            # encontrar ID da pessoa
            if True in matches:
                index = matches.index(True)  
            else:
                
                # atribui ID, armazena face e cria lista
                index = len(known_encodings)  
                known_encodings.append(encoding)
                face_data[index] = []

            # limita apenas 5 imagens por pessoa
            if len(face_data[index]) < 9:
                top, right, bottom, left = location
                face_image = frame[top:bottom, left:right]
                
                # arquivo de imagem: Pessoa_{ID}_{número}.jpg
                filename = os.path.join(output_folder, f"Pessoa_{index+1}_{len(face_data[index])+1}.jpg")
                cv2.imwrite(filename, face_image)
                face_data[index].append(filename)
                 
    # libera os recursos e finaliza                                                
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Extração concluída e arquivos de faces salvos em {output_folder}.")

if __name__ == "__main__":
    
    video_entrada = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
    pasta_saida = "faces_extraidas"
    
    extrair_faces_do_video(video_entrada, pasta_saida)
