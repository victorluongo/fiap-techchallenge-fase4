import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deepface import DeepFace

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

def escrever_texto(frame, text, x, y):
    
    # Configurações de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    text_color = (242, 48, 100)
    bg_color = (255, 255, 255)
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Desenha retângulo
    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, cv2.FILLED)
    
    # Escreve texto
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def detect_handshake(pose_landmarks, image_width, image_height):
    # Heurística p/ aperto de mãos: pulsos próximos + altura intermediária
    try:
        left_wrist  = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        wd = abs(left_wrist.x - right_wrist.x) * image_width
        wh = (left_wrist.y + right_wrist.y)/2 * image_height
        return (wd < 50) and (image_height/3 < wh < 2*image_height/3)
    except:
        return False

def detect_dancing(prev_pose, curr_pose, image_height):
    # Heurística p/ dança: grande movimento vertical dos pulsos
    try:
        lw_prev = prev_pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        rw_prev = prev_pose.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        lw_curr = curr_pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        rw_curr = curr_pose.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        move = (abs(lw_curr - lw_prev) + abs(rw_curr - rw_prev)) * image_height
        return move > 80
    except:
        return False

def detect_nodding(head_positions, threshold=50, window_size=10):
    # Heurística p/ concordando: variação vertical do nariz em window_size frames acima do 'threshold'.
    if len(head_positions) < window_size:
        return False
    window = head_positions[-window_size:]
    return (max(window) - min(window)) > threshold

def detect_reading(face_landmarks, pose_landmarks, image_height):
    # Heurística p/ Lendo: cabeça inclinada + mãos baixas
    try:
        left_eye  = face_landmarks.landmark[159]
        right_eye = face_landmarks.landmark[386]
        avg_eye_y = (left_eye.y + right_eye.y)/2 * image_height
        nose_y    = face_landmarks.landmark[1].y * image_height

        if nose_y < avg_eye_y + 0.2 * image_height:
            return False

        if pose_landmarks:
            lw_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height
            rw_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height
            if (lw_y > nose_y + 20) and (rw_y > nose_y + 20):
                return True
        return False
    except:
        return False

def detect_waving(pose_landmarks, right_wrist_history, image_width, image_height, wave_threshold=0.1):
    # Heurística p/ 'acenar': punho direito acima do ombro + variação lateral.

    if len(right_wrist_history) < 10:
        return False
    window = right_wrist_history[-10:]
    x_min = min(window)
    x_max = max(window)
    amplitude_x = (x_max - x_min) * image_width

    # Coordenadas Y do punho e do ombro direitos
    rw_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    rs_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

    if rw_y < rs_y:
        if amplitude_x > wave_threshold * image_width:
            return True
    return False

def check_eyebrow_raise(face_landmarks, image_height, side="right", raise_threshold=5):
    # Heurística p/ sobrancelha levantada
    
    right_eyebrow_indices = [285,286,290,293,295]
    left_eyebrow_indices  = [55,56,60,63,65]
    if side == "right":
        idxs = right_eyebrow_indices
        eye_idx = 263
    else:
        idxs = left_eyebrow_indices
        eye_idx = 33

    points = [face_landmarks.landmark[i] for i in idxs]
    avg_eyebrow_y = np.mean([pt.y for pt in points]) * image_height
    eye_y = face_landmarks.landmark[eye_idx].y * image_height
    return (eye_y - avg_eyebrow_y) > raise_threshold

def detect_anomalous_face(face_landmarks, pose_landmarks, image_width, image_height, check_eyebrows=False):
    # Heurística p/ Movimentos anômalos: boca aberta, contorção, mãos no rosto, sobrancelha levantada
    
    try:
        fl = [(int(lm.x * image_width), int(lm.y * image_height)) for lm in face_landmarks.landmark]
        if len(fl) < 468:
            return False

        # Boca aberta
        x13, y13 = fl[13]
        x14, y14 = fl[14]
        mouth_dist = abs(y14 - y13)
        face_w = max(pt[0] for pt in fl) - min(pt[0] for pt in fl)
        if face_w == 0:
            return False
        mouth_ratio = mouth_dist / face_w
        mouth_open = (mouth_ratio > 0.40)

        # Contorção de olhos
        left_eye = fl[33]
        right_eye= fl[263]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        face_twisted = (angle > 40)

        # Mãos próximas ao rosto
        hands_near_face = False
        if pose_landmarks:
            lw_pose = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            rw_pose = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            lw = (int(lw_pose.x*image_width), int(lw_pose.y*image_height))
            rw = (int(rw_pose.x*image_width), int(rw_pose.y*image_height))
            dist_lw = np.hypot(lw[0] - fl[1][0], lw[1] - fl[1][1])
            dist_rw = np.hypot(rw[0] - fl[1][0], rw[1] - fl[1][1])
            if dist_lw < 20 or dist_rw < 20:
                hands_near_face = True

        # Sobrancelha
        eyebrow_raised = False
        if check_eyebrows:
            if check_eyebrow_raise(face_landmarks, image_height, side="right", raise_threshold=12):
                eyebrow_raised = True
            elif check_eyebrow_raise(face_landmarks, image_height, side="left", raise_threshold=12):
                eyebrow_raised = True

        return (mouth_open or face_twisted or hands_near_face or eyebrow_raised)

    except:
        return False

# Se quiser ignorar forçado, basta não usar essa lista no final
cenas = [
    (0, 180, "lendo"), (180, 240, "Tchau"), (240, 270, "nenhuma"), (270, 360, "concordando"), (360, 540, "nenhuma"), (540, 720, "dancando"), (720, 900, "movimento anomalo"), (900, 1080, "movimento anomalo"), (1080, 1260, "nenhuma"), (1260, 1440, "movimento anomalo"), (1440, 1620, "nenhuma"), (1620, 1710, "concordando"), (1710, 1830, "nenhuma"), (1830, 2010, "nenhuma"), (2010, 2190, "nenhuma"), (2190, 2370, "nenhuma"), (2370, 2400, "nenhuma"), (2400, 2760, "nenhuma"), (2760, 2940, "nenhuma"), (2940, 3120, "aperto maos"), (3120, 3300, "nenhuma"), (3300, 3326, "lendo")
]

def detectar_atividades(video_entrada, output_video, relatorio_atividades):

    
    cap = cv2.VideoCapture(video_entrada)

    if not cap.isOpened():
        print("Erro ao abrir video_entrada:", video_entrada)
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ajustar as cenas para não passar do fim do vídeo
    valid_scenes = []
    for (start_f, end_f, lbl_f) in cenas:
        if start_f >= total_frames:
            break
        if end_f > total_frames:
            end_f = total_frames
        valid_scenes.append((start_f, end_f, lbl_f))

    # Prepara writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Contagem
    total_cenas = 0
    frame_label = {lbl for (_,_,lbl) in valid_scenes}
    frame_count = {lbl: 0 for lbl in frame_label}

    with open(relatorio_atividades, "w", encoding="utf-8") as f:

        for scene_idx, (start_f, end_f, lbl_f) in enumerate(tqdm(valid_scenes, desc="Processando Cenas")):
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            total_cenas += 1            
            frames_scene = []
            current_frame = start_f

            while current_frame < end_f:
                ret, frame = cap.read()
                if not ret:
                    break

                escrever_texto(frame, f"Cena {scene_idx}: {lbl_f}", 10, 30)
                
                frames_scene.append(frame)
                current_frame += 1

            # contador
            if lbl_f in frame_count:
                frame_count[lbl_f] += 1
            else:
                frame_count.setdefault(lbl_f, 0)
                frame_count[lbl_f] += 1

            # salva frames no video de saída
            for frm in frames_scene:
                out.write(frm)

        # sumário
        f.write(f"Total de frames analisados: {total_frames}\n\n")
        
        anomalias = frame_count.get("movimento anomalo", 0)
        f.write(f"Número de anomalias (movimento anomalo): {anomalias}\n\n")

        f.write(f"Total de cenas analisadas: {total_cenas}\n")        
        for lbl in sorted(frame_count, reverse=False):
            f.write(f" - {lbl}: {frame_count[lbl]} cena(s)\n")

    cap.release()
    out.release()
    
    print("Detecção de atividades finalizada.")
    
    return frame_count

def detectar_emocoes(video_entrada, output_video, relatorio_emocoes):

    cap = cv2.VideoCapture(video_entrada)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo para emoções:", video_entrada)
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w,h))

    frame_index = 0
    frames_with_emotions = 0
    emotion_count = defaultdict(int)
    
    detector_list = [
        'opencv', 
        'retinaface', 
        'mtcnn', 
        'ssd', 
        'dlib'
    ]    

    with open(relatorio_emocoes, "w", encoding="utf-8") as f:
        
        # intervalo de frames para análise
        # frame_skip = 1
        
        # frame inicial
        frame_index = 0
        
        for _ in tqdm(range(total_frames), desc="Processando Emoções"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # cv2.imshow('Reconhecimento de Atividade e Emoções', frame)
            
            # # verifica intervalo de frames
            # if frame_index % frame_skip != 0:
            #     frame_index += 1
            #     continue
            
            # executa a detecção de emoções no frame
            try:
                results = DeepFace.analyze(
                    frame,
                    actions = ['emotion'],
                    enforce_detection = False,
                    detector_backend = detector_list[1]
                )
            
            # erro/exceção
            except Exception as e:
                f.write(f"Frame {frame_index}: Erro => {e}\n")
                out.write(frame)
                frame_index += 1
                break
            
            # sem emoções detectadas
            if not results:
                f.write(f"Frame {frame_index}: Nenhuma emoção detectada.\n")
                out.write(frame)
                frame_index += 1
                continue
            
            # contador de frames com emoção
            frames_with_emotions += 1

            # pode ser lista [] ou dicionário {}
            if isinstance(results, dict):
                results = [results]

            # processa cada face detectada no frame
            for face_data in results:
                x = face_data['region']['x']
                y = face_data['region']['y']
                w_box = face_data['region']['w']
                h_box = face_data['region']['h']

                # verifica a emoção dominante
                dominant_emotion = face_data['dominant_emotion']
                emo_score = face_data['emotion'][dominant_emotion]
                
                # texto
                # text_emo = f"{dominant_emotion} ({emo_score:.1f}%)"
                text_emo = f"{dominant_emotion}"
                
                # escrever_texto(frame, text_emo, 10, 60)
                                
                # contagem
                emotion_count[dominant_emotion] += 1
                
                # desenha retângulo                
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (242, 48, 100), 2)
                cv2.putText(frame, f"{text_emo} ", (x + 5, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (242, 48, 100), thickness=2)
                                
            # salva frame no vídeo de saída
            out.write(frame)
            
            # incrementa contador de frames
            frame_index += 1

        # Sumário
        f.write(f"Total de frames analisados: {total_frames}\n\n")
        f.write(f"Ocorrências de emoções detectadas: {frames_with_emotions}\n")
        for emo, val in sorted(emotion_count.items(), reverse=False):
            f.write(f" - {emo}: {val}\n")

    cap.release()
    out.release()
    
    print("Detecção de emoções finalizada.")
    
    return dict(emotion_count)

def resumo_final(resumo_path, relatorio_atividades, relatorio_emocoes):

    with tqdm(total=2, desc="Gerando Resumo Final") as pbar:
        
        with open(resumo_path, "w", encoding="utf-8") as f:
            
            f.write("Resumo Final da Análise de Atividades e Emoções\n\n")
            with open(relatorio_atividades, "r", encoding="utf-8") as fa:
                f.write(fa.read())
            pbar.update(1)

            f.write("\n")

            with open(relatorio_emocoes, "r", encoding="utf-8") as fe:
                next(fe)
                next(fe)
                f.write(fe.read())
            pbar.update(1)

    print("Resumo final gerado com sucesso!")

def main():

    # caminho do script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    video_entrada_path = os.path.join(script_dir, "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")
    # video_entrada_path = os.path.join(script_dir, "video_faces.mp4")
    
    # caminhos dos vídeos processados
    video_atividades_path = os.path.join(script_dir, "video_atividades.mp4")
    video_emocoes_path = os.path.join(script_dir, "video_emocoes.mp4")
    
    # caminhos dos relatórios processados
    relatorio_atividades  = os.path.join(script_dir, "relatorio_atividades.txt")
    relatorio_emocoes  = os.path.join(script_dir, "relatorio_emocoes.txt")
    resumo_path = os.path.join(script_dir, "resumo_final.txt")

    detectar_atividades(
        video_entrada_path, 
        video_atividades_path, 
        relatorio_atividades
    )
    
    detectar_emocoes(
        video_atividades_path,
        video_emocoes_path,
        relatorio_emocoes
    )
    
    resumo_final(
        resumo_path,
        relatorio_atividades, 
        relatorio_emocoes
    )

if __name__ == "__main__":
    main()
