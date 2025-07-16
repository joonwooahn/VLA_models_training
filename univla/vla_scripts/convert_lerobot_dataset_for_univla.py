import os
import cv2
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# 사용자 설정 영역 (이전과 동일, 경로 확인)
# ==============================================================================

# base_input_dir과 output_dir은 shell script에서 동적으로 설정됩니다
base_input_dir = None
output_dir = None

VIDEO_KEY = 'observation.images.sideview'

STATE_COLUMN_NAME = 'observation.state'
ACTION_COLUMN_NAME = 'action'
INSTRUCTION_COLUMN_NAME = 'language_instruction'
# ==============================================================================

def process_episode(parquet_file, video_dir_template, output_root):
    # try-except 블록을 제거하여 오류 발생 시 전체 Traceback을 확인합니다.
    
    df = pd.read_parquet(parquet_file)
    if df.empty:
        print(f"File {os.path.basename(parquet_file)} is empty, skipping.")
        return

    # --- 가장 안정적인 변환 방식 적용 ---
    states_list = df[STATE_COLUMN_NAME].tolist()
    actions_list = df[ACTION_COLUMN_NAME].tolist()
    instruction = df[INSTRUCTION_COLUMN_NAME].iloc[0]

    # tolist()로 얻은 리스트를 np.stack을 사용해 안정적으로 변환합니다.
    states = np.stack(states_list).astype(np.float32)
    actions = np.stack(actions_list).astype(np.float32)
    # ---

    episode_name = os.path.basename(parquet_file).replace('.parquet', '')
    video_caps = []
    video_file = os.path.join(video_dir_template, VIDEO_KEY, f"{episode_name}.mp4")
    if not os.path.exists(video_file):
        print(f"Warning: Video file not found {video_file}, skipping episode.")
        return
    video_caps.append(cv2.VideoCapture(video_file))
    
    episode_output_path = os.path.join(output_root, episode_name)
    os.makedirs(episode_output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 이미지를 바로 저장합니다. (hconcat 제거)
        cv2.imwrite(os.path.join(episode_output_path, f"frame_{frame_count + 1:03d}.png"), frame)
        frame_count += 1
    cap.release()
    
    if len(actions) != frame_count:
        print(f"Warning: Action length ({len(actions)}) and frame count ({frame_count}) differ in {episode_name}.")

    np.save(os.path.join(episode_output_path, "state.npy"), states)
    np.save(os.path.join(episode_output_path, "action.npy"), actions)
    with open(os.path.join(episode_output_path, "instruction.txt"), "w") as f:
        f.write(instruction)

def main():
    # 입력 데이터의 마지막 폴더명을 그대로 사용하되, 명시적으로 설정
    input_folder_name = os.path.basename(base_input_dir)
    final_output_dir = os.path.join(output_dir, input_folder_name)
    
    print(f"입력 폴더명: {input_folder_name}")
    print(f"출력 디렉토리: {final_output_dir}")
    video_dir_template = os.path.join(base_input_dir, 'videos/chunk-000')
    data_dir = os.path.join(base_input_dir, 'data/chunk-000')

    # 필요한 비디오 경로가 존재하는지 확인
    if not os.path.isdir(os.path.join(video_dir_template, VIDEO_KEY)):
         print(f"오류: 비디오 디렉토리 '{os.path.join(video_dir_template, VIDEO_KEY)}'를 찾을 수 없습니다.")
         return

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    
    if not parquet_files:
        print(f"오류: {data_dir} 에서 .parquet 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(parquet_files)}개의 에피소드를 변환합니다.")
    print(f"사용할 비디오: {VIDEO_KEY}")
    print(f"변환된 데이터 저장 위치: '{final_output_dir}'")

    for parquet_file in tqdm(parquet_files):
        process_episode(parquet_file, video_dir_template, final_output_dir)

    print("\n데이터 변환 완료!")

if __name__ == '__main__':
    main()
